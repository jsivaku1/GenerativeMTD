# app.py
# Main Flask application file. Handles web routes, data processing, and model training orchestration.

import os
import json
import uuid
import time
import pandas as pd
import numpy as np
import traceback
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, render_template, url_for, send_from_directory, Response, stream_with_context, jsonify
from werkzeug.utils import secure_filename

# Local imports for the data pipeline, models, and utilities
from data_pipeline import DataPipeline
from GenerativeMTD import GenerativeMTD
from kNNMTD import kNNMTD
from mtd_utils import stat_tests, predictive_model, regression_model, analyze_columns, unsupervised_clustering_utility

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
LOG_FILE = 'app.log'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Logging Setup ---
handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# --- Routes ---

@app.route('/')
def index():
    """Renders the main upload and configuration page."""
    return render_template('index.html')

@app.route('/info', methods=['POST'])
def get_file_info():
    """Analyzes the uploaded CSV to identify column types for the UI."""
    file = request.files.get('dataset')
    if file and file.filename != '':
        try:
            df = pd.read_csv(file)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return jsonify(analyze_columns(df))
        except Exception as e:
            app.logger.error(f"Error analyzing file: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Could not process CSV. Please ensure it is a valid CSV file. Error: {e}"}), 500
    return jsonify({"error": "No file uploaded"}), 400

@app.route('/generate', methods=['POST'])
def generate():
    """Handles the synthetic data generation request."""
    file = request.files.get('dataset')
    if not file or file.filename == '':
        return Response(json.dumps({'status': 'error', 'message': 'Invalid file provided'}), mimetype='application/json', status=400)

    try:
        opts = {
            "n_samples": int(request.form.get('n_samples')),
            "pseudo_n_obs": int(request.form.get('pseudo_n_obs')),
            "epochs": int(request.form.get('epochs')),
            "batch_size": int(request.form.get('batch_size')),
            "embedding_dim": int(request.form.get('embedding_dim')),
            "lr": float(request.form.get('lr')),
            "seed": 42
        }
        class_col = request.form.get('class_col')
        task_type = request.form.get('task_type')
        if class_col == 'none':
            class_col = None
            task_type = 'unsupervised'

    except (ValueError, TypeError) as e:
        app.logger.error(f"Invalid parameter error: {e}")
        return Response(json.dumps({'status': 'error', 'message': f'Invalid parameter provided: {e}'}), mimetype='application/json', status=400)

    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    real_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_real.csv")
    file.seek(0)
    file.save(real_filepath)
    app.logger.info(f"--- Starting Task {task_id} for file {filename} ---")
    app.logger.info(f"Parameters: {opts}, Target: {class_col}, Task: {task_type}")

    def training_stream():
        start_time = time.time()
        logs = []

        def progress_callback(log_data):
            nonlocal logs
            if log_data.get('status') == 'training':
                logs.append(log_data)
            yield f"data: {json.dumps(log_data)}\n\n"

        try:
            yield from progress_callback({'status': 'Loading data and preparing pipeline...'})
            real_df = pd.read_csv(real_filepath)
            real_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            pipeline = DataPipeline()
            pipeline.fit(real_df)
            real_df_imputed = pipeline.inverse_transform(pipeline.transform(real_df))
            app.logger.info(f"Task {task_id}: Data pipeline fitted successfully.")

            # --- Optimal k Selection for kNNMTD ---
            yield from progress_callback({'status': 'Optimizing k for kNNMTD...'})
            k_options = [3, 5, 7, 10]
            best_k, best_pcd = 3, float('inf')
            for k_val in k_options:
                if k_val >= len(real_df_imputed): continue
                try:
                    temp_knnmtd = kNNMTD(n_obs=50, k=k_val, random_state=opts['seed'])
                    temp_pseudo = next(temp_knnmtd.fit_generate(real_df_imputed))
                    current_pcd = stat_tests(real_df_imputed, temp_pseudo)['pcd']
                    if current_pcd is not None and not np.isnan(current_pcd) and current_pcd < best_pcd:
                        best_pcd = current_pcd
                        best_k = k_val
                except Exception as e:
                    app.logger.warning(f"Could not test k={k_val}. Error: {e}")
            opts['k'] = best_k
            yield from progress_callback({'status': f'Best k found: {best_k}. Generating pseudo-real data...'})
            
            knnmtd_generator = kNNMTD(n_obs=opts['pseudo_n_obs'], k=opts['k'], random_state=opts['seed'])
            pseudo_real_df = next(knnmtd_generator.fit_generate(real_df_imputed))
            app.logger.info(f"Task {task_id}: Generated {len(pseudo_real_df)} pseudo-real samples.")
            
            yield from progress_callback({'status': 'Evaluating pseudo-real data...'})
            initial_stats = stat_tests(real_df_imputed, pseudo_real_df)
            initial_ml = {}
            if task_type == 'unsupervised':
                initial_ml = unsupervised_clustering_utility(real_df_imputed, pseudo_real_df)
            elif class_col:
                eval_func = predictive_model if task_type == 'classification' else regression_model
                for mode in ['TSTR', 'TRTS', 'TRTR', 'TSTS']:
                    try:
                        initial_ml[mode] = eval_func(real_df_imputed, pseudo_real_df, class_col, mode)
                    except Exception as e:
                        app.logger.warning(f"Could not compute initial ML utility for mode {mode}: {e}")
                        initial_ml[mode] = (np.nan, np.nan, np.nan)

            yield from progress_callback({'status': 'Initializing GenerativeMTD model...'})
            model = GenerativeMTD(real_df_imputed, pipeline, opts)
            
            yield from progress_callback({'status': 'Starting model training...'})
            yield from model.train_vae(pseudo_real_df, callback=progress_callback)
            app.logger.info(f"Task {task_id}: Model training completed.")

            yield from progress_callback({'status': 'Generating final synthetic data...'})
            synthetic_df = model.sample(opts['n_samples'])
            app.logger.info(f"Task {task_id}: Generated {len(synthetic_df)} final synthetic samples.")

            yield from progress_callback({'status': 'Evaluating final synthetic data...'})
            final_stats = stat_tests(real_df_imputed, synthetic_df)
            final_ml = {}
            if task_type == 'unsupervised':
                final_ml = unsupervised_clustering_utility(real_df_imputed, synthetic_df)
            elif class_col:
                eval_func = predictive_model if task_type == 'classification' else regression_model
                for mode in ['TSTR', 'TRTS', 'TRTR', 'TSTS']:
                    try:
                        final_ml[mode] = eval_func(real_df_imputed, synthetic_df, class_col, mode)
                    except Exception as e:
                        app.logger.error(f"Final ML evaluation failed for mode {mode}: {e}")
                        final_ml[mode] = (np.nan, np.nan, np.nan)
            
            yield from progress_callback({'status': 'Finalizing results...'})
            generated_filename = f"{task_id}_synthetic.csv"
            synthetic_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], generated_filename), index=False)
            
            results_payload = {
                'generated_filename': generated_filename,
                'input_options': {**opts, 'Filename': filename, 'Runtime (s)': f"{time.time() - start_time:.2f}", 'Target': class_col or "None"},
                'initial_metrics': {k.upper(): f"{v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else "N/A" for k, v in initial_stats.items()},
                'final_metrics': {k.upper(): f"{v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else "N/A" for k, v in final_stats.items()},
                'initial_ml_utility_scores': initial_ml,
                'final_ml_utility_scores': final_ml,
                'training_logs': logs,
                'task_type': task_type
            }
            
            with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_results.json"), 'w') as f:
                json.dump(results_payload, f)
            
            yield from progress_callback({'status': 'complete', 'task_id': task_id})
            app.logger.info(f"--- Task {task_id} completed in {time.time() - start_time:.2f} seconds ---")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"An unexpected error occurred: {str(e)}"
            app.logger.error(f"Task {task_id}: FAILED. {error_message}\n{tb_str}")
            yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
    
    return Response(stream_with_context(training_stream()), mimetype='text/event-stream')

@app.route('/results/<task_id>')
def results(task_id):
    """Displays the results page for a completed task."""
    try:
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_results.json")
        with open(results_path, 'r') as f:
            data = json.load(f)
        pipeline_stages = [
            ("K-Value Optimization", "Automatically selected the best k for kNNMTD."),
            ("Pseudo-Real Generation", "Generated intermediate data via kNNMTD."),
            ("GenerativeMTD Training", "Trained model with Sinkhorn & MMD losses."),
            ("Final Generation & Evaluation", "Sampled final data and computed all metrics.")
        ]
        return render_template('results.html', **data, pipeline_stages=pipeline_stages)
    except FileNotFoundError:
        return render_template('error.html', error_message="Results for this task were not found."), 404
    except Exception as e:
        app.logger.error(f"Failed to load results for task {task_id}: {e}\n{traceback.format_exc()}")
        return render_template('error.html', error_message=f"Could not load results: {e}"), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Provides a download link for the generated synthetic CSV."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
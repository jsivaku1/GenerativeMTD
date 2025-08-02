import os
import json
import uuid
import time
import pandas as pd
import traceback
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, render_template, url_for, send_from_directory, Response, stream_with_context, jsonify
from werkzeug.utils import secure_filename
import numpy as np

# --- Import local modules ---
from data_pipeline import DataPipeline
from GenerativeMTD import GenerativeMTD
from mtd_utils import stat_tests, predictive_model, find_target_column, analyze_columns

# --- App Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
LOG_FILE = 'app.log'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(log_formatter)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addHandler(handler)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/info', methods=['POST'])
def get_file_info():
    """Analyzes the uploaded CSV and returns column information."""
    if 'dataset' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['dataset']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)
            column_info = analyze_columns(df)
            return jsonify(column_info)
        except Exception as e:
            return jsonify({"error": f"Could not process CSV file: {e}"}), 500
    return jsonify({"error": "Invalid file type"}), 400


@app.route('/generate', methods=['POST'])
def generate():
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['dataset']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no file selected'}), 400

    try:
        opts = {
            "n_samples": int(request.form.get('n_samples', 1000)),
            "pseudo_n_obs": int(request.form.get('pseudo_n_obs', 500)),
            "epochs": int(request.form.get('epochs', 100)),
            "batch_size": int(request.form.get('batch_size', 500)),
            "embedding_dim": int(request.form.get('embedding_dim', 128)),
            "lr": float(request.form.get('lr', 2e-4)),
            "seed": 42
        }
        class_col_name = request.form.get('class_col', 'none')
        if class_col_name == 'none':
            class_col_name = None

    except (ValueError, TypeError) as e:
        app.logger.error(f"Invalid parameter value: {e}")
        return jsonify({'error': f'Invalid parameter value: {e}'}), 400
        
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    real_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_real.csv")
    file.seek(0)
    file.save(real_filepath)
    
    def training_stream():
        start_time = time.time()
        logs = []
        
        def progress_callback(log):
            nonlocal logs
            logs.append(log)
            yield f"data: {json.dumps(log)}\n\n"

        try:
            app.logger.info(f"--- Starting Task {task_id} for file {filename} ---")
            
            yield f"data: {json.dumps({'status': 'Loading data and preparing pipeline...'})}\n\n"
            real_df = pd.read_csv(real_filepath)
            pipeline = DataPipeline()
            pipeline.fit(real_df)
            
            yield f"data: {json.dumps({'status': 'Optimizing k for kNN-MTD...'})}\n\n"
            max_k = len(real_df) - 1
            if class_col_name and class_col_name in real_df.columns:
                min_class_size = real_df[class_col_name].value_counts().min()
                max_k = min(max_k, min_class_size)
            
            k_options = [k for k in [2, 3, 5] if k <= max_k]
            if not k_options: k_options = [min(2, max_k)] if max_k > 1 else []

            best_k, best_pcd = -1, float('inf')
            imputed_transformed = pipeline.transform(real_df)
            real_df_imputed = pipeline.inverse_transform(imputed_transformed)
            
            if k_options:
                for k_val in k_options:
                    temp_opts = opts.copy()
                    temp_opts['k'] = k_val
                    model_test = GenerativeMTD(real_df, temp_opts, device='cpu')
                    pseudo_df_test = model_test.generate_pseudo_real_data(pipeline)
                    current_pcd = stat_tests(real_df_imputed, pseudo_df_test)['pcd']
                    
                    if not np.isnan(current_pcd) and current_pcd < best_pcd:
                        best_pcd = current_pcd
                        best_k = k_val
            
            if best_k == -1: best_k = 3 # Fallback k
            opts['k'] = best_k
            yield f"data: {json.dumps({'status': f'Best k found: {best_k}. Generating pseudo-real data...'})}\n\n"

            model = GenerativeMTD(real_df, opt=opts, device='cpu')
            pseudo_real_df = model.generate_pseudo_real_data(pipeline)

            # --- Calculate Initial ML Utility on Pseudo-Real Data ---
            initial_ml_utility_scores = {}
            target_col = class_col_name or find_target_column(real_df_imputed)
            if target_col:
                for mode in ['TSTR', 'TRTS', 'TRTR', 'TSTS']:
                    acc, f1 = predictive_model(real_df_imputed, pseudo_real_df, target_col, mode=mode)
                    initial_ml_utility_scores[f'{mode}_Accuracy'] = f"{acc*100:.2f}%"

            yield f"data: {json.dumps({'status': 'Starting VAE model training...'})}\n\n"
            yield from model.train_vae(pseudo_real_df, pipeline, callback=progress_callback)
            
            yield f"data: {json.dumps({'status': 'Generating final synthetic data...'})}\n\n"
            synthetic_df = model.sample(opts['n_samples'], pipeline)
            
            yield f"data: {json.dumps({'status': 'Calculating final metrics...'})}\n\n"
            
            final_stats = stat_tests(real_df_imputed, synthetic_df)
            
            # --- Calculate Final ML Utility on Synthetic Data ---
            final_ml_utility_scores = {}
            if target_col:
                for mode in ['TSTR', 'TRTS', 'TRTR', 'TSTS']:
                    acc, f1 = predictive_model(real_df_imputed, synthetic_df, target_col, mode=mode)
                    final_ml_utility_scores[f'{mode}_Accuracy'] = f"{acc*100:.2f}%"
            
            generated_filename = f"{task_id}_synthetic.csv"
            synthetic_filepath = os.path.join(app.config['UPLOAD_FOLDER'], generated_filename)
            synthetic_df.to_csv(synthetic_filepath, index=False)
            
            total_runtime = f"{time.time() - start_time:.2f} seconds"
            
            initial_metrics = logs[0] if logs else {}
            final_metrics_display = {k: f"{v:.4f}" if v is not None else "N/A" for k, v in final_stats.items()}
            initial_metrics_display = {
                'PCD': f"{initial_metrics.get('pcd', 0):.4f}",
                'NNDR': f"{initial_metrics.get('nndr', 0):.4f}",
                'DCR': f"{initial_metrics.get('dcr', 0):.4f}"
            }

            results_payload = {
                'generated_filename': generated_filename,
                'input_options': {**opts, 'Original Filename': filename, 'Total Runtime': total_runtime, 'Task Target': target_col or "Unsupervised"},
                'initial_metrics': initial_metrics_display,
                'final_metrics': final_metrics_display,
                'initial_ml_utility_scores': initial_ml_utility_scores,
                'final_ml_utility_scores': final_ml_utility_scores,
                'training_logs': logs
            }
            
            results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_results.json")
            with open(results_filepath, 'w') as f:
                json.dump(results_payload, f)

            yield f"data: {json.dumps({'status': 'complete', 'task_id': task_id})}\n\n"

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"An unexpected error occurred: {str(e)}"
            app.logger.error(f"Task {task_id}: FAILED. {error_message}\nTraceback:\n{tb_str}")
            yield f"data: {json.dumps({'status': 'error', 'message': error_message})}\n\n"
    
    return Response(stream_with_context(training_stream()), mimetype='text/event-stream')

@app.route('/results/<task_id>')
def results(task_id):
    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_results.json")
    try:
        with open(results_filepath, 'r') as f:
            results_data = json.load(f)
        
        pipeline_stages = [
            ("K-Optimization", "Automatically selected the best k for kNN-MTD."),
            ("Pseudo-Real Generation", "Generated intermediate data using kNN-MTD."),
            ("VAE Training", "Trained a VAE with live metric tracking."),
            ("Final Generation", "Sampled new data and converted it to the original format.")
        ]
        
        return render_template('results.html', **results_data, pipeline_stages=pipeline_stages)
    except FileNotFoundError:
        return render_template('error.html', error_message="Results for this task could not be found."), 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
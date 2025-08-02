# GenerativeMTD: A Deep Synthetic Data Generation Framework

This repository contains the source code for **GenerativeMTD**, a web-based application for generating high-fidelity synthetic tabular data. The application is a direct implementation of the research paper, "GenerativeMTD: A Deep Synthetic Data Generation Framework for Small Datasets."

## 📜 Overview

-   **The Challenge**: Deep learning-based synthetic data generation frameworks for small tabular data are limited. Training deep models on small datasets often leads to overfitting, poor generalization, and privacy risks.
-   **Our Solution**: GenerativeMTD provides a deep learning-based framework specifically designed for synthetic data generation from small datasets.
-   **Core Idea**: Instead of using the real dataset for training, GenerativeMTD first employs the **kNNMTD** algorithm to create a larger, intermediate "pseudo-real" dataset. This pseudo-real data preserves the statistical properties of the original data while expanding the sample size.
-   **Translation to Reality**: The VAE-GAN model is then trained on this pseudo-real data, learning to translate it into a final synthetic dataset that is statistically similar and privacy-preserving.

The following illustration shows how the algorithm generates artificial samples.

<div align="center">
<br/>
<p align="center">
<img align="center" width=90% src="https://github.com/jsivaku1/GenerativeMTD/blob/main/genMTD.jpg"></img>
</p>
</div>

## 🌟 Key Features

-   **Advanced Generative Model**: Implements a VAE-GAN architecture with sophisticated loss functions, including Sinkhorn Divergence and Maximum Mean Discrepancy (MMD), to ensure high statistical similarity between real and synthetic data.
-   **Small Dataset Specialization**: Utilizes the novel **kNNMTD** algorithm to generate an intermediate "pseudo-real" dataset, enabling deep learning models to be trained effectively even on small initial datasets.
-   **Automatic Hyperparameter Tuning**: The application automatically optimizes the `k` value for the kNNMTD algorithm to produce the best possible pseudo-real data.
-   **Comprehensive Evaluation**: Provides a rich set of metrics to evaluate the quality of the generated data, including:
    -   **Statistical & Privacy Metrics**: Pairwise Correlation Difference (PCD), Distance to Closest Record (DCR), and Nearest Neighbor Distance Ratio (NNDR).
    -   **Machine Learning Utility**: A full comparison of model performance (Accuracy, F1, AUC for classification; R², RMSE, MAE for regression) across TSTR, TRTS, TRTR, and TSTS scenarios.
    -   **Unsupervised Utility**: Clustering metrics (Silhouette Score, Calinski-Harabasz) to evaluate structural preservation.
-   **Interactive Web Interface**: An intuitive UI for uploading data, configuring parameters, and visualizing the training process and results in real-time.

## ⚙️ Technical Details

The GenerativeMTD framework operates in a two-stage process:

1.  **Pseudo-Real Data Generation (kNNMTD)**: For each data point in the small real dataset, we find its k-nearest neighbors. Based on the statistical properties (min, max, variance, skew) of this local neighborhood, we define a plausible domain and generate a large number of "pseudo-real" samples. This step effectively enriches the dataset, providing enough data for the deep learning model to train on without directly memorizing the original, sensitive data points.

2.  **Translation and Refinement (GenerativeMTD)**: A VAE-GAN model is trained using the pseudo-real data as input. The model's goal is not just to reconstruct the pseudo-real data, but to "translate" it to be as close as possible to the *real* data's distribution. This is achieved through a composite loss function:
    -   **Sinkhorn Divergence**: Measures the distance between the latent space distributions of the real and generated data.
    -   **MMD + Cross-Entropy Loss**: Acts as a reconstruction loss, ensuring the generated data points are statistically similar to the real ones.
    -   **Adversarial Loss**: A critic network provides feedback to the generator, pushing the generated data to be indistinguishable from the real data.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   Flask
-   pandas
-   numpy
-   scikit-learn
-   torch
-   dython

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd GenerativeMTD
    ```
2.  Install the required Python packages:
    ```bash
    pip install Flask pandas numpy scikit-learn torch dython
    ```

### Running the Application

1.  Navigate to the project's root directory.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to `http://127.0.0.1:5000`.

## 🛠️ How to Use

1.  **Upload Data**: Click the "Upload Dataset" button and select a `.csv` file from your local machine.
2.  **Select Target Column**: (Optional) If you want to evaluate the data for a specific machine learning task, select the target column. The application will automatically detect whether it's a classification or regression task. If you select "None," unsupervised clustering metrics will be calculated.
3.  **Configure Parameters**: Adjust the generation parameters as needed, such as the number of synthetic samples to generate and the number of training epochs.
4.  **Generate Data**: Click the "Generate Data" button to start the process.
5.  **View Results**: You will be automatically redirected to a results page where you can view the training plots, comparison metrics, and download your new synthetic dataset.

## 📁 Project Structure

-   `app.py`: The main Flask application that handles web routes and orchestrates the data generation process.
-   `GenerativeMTD.py`: The core implementation of the VAE-GAN model.
-   `kNNMTD.py`: The implementation of the k-Nearest Neighbor Mega-Trend Diffusion algorithm.
-   `data_pipeline.py`: A robust pipeline for data cleaning, imputation, and transformation.
-   `mtd_utils.py`: Contains helper functions for loss calculations and all evaluation metrics.
-   `templates/`: Contains the HTML files for the web interface (`index.html`, `results.html`, etc.).
-   `static/`: Contains the CSS stylesheet (`style.css`).

##  Citing this Work

If you use this framework in your research, please cite both of the following papers which form the basis of this implementation.

1.  **For the main framework:**
    -   Sivakumar, Jayanth, et al. "GenerativeMTD: A deep synthetic data generation framework for small datasets." *Knowledge-Based Systems* 280 (2023): 110956.

    ```bibtex
    @article{sivakumar2023generativemtd,
      title={GenerativeMTD: A deep synthetic data generation framework for small datasets},
      author={Sivakumar, Jayanth and Ramamurthy, Karthik and Radhakrishnan, Menaka and Won, Daehan},
      journal={Knowledge-Based Systems},
      volume={280},
      pages={110956},
      year={2023},
      publisher={Elsevier}
    }
    ```

2.  **For the pseudo-real data generation algorithm:**
    -   Sivakumar, Jayanth, et al. "Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors." *Knowledge-Based Systems* 235 (2022): 107687.

    ```bibtex
    @article{sivakumar2022synthetic,
      title={Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors},
      author={Sivakumar, Jayanth and Ramamurthy, Karthik and Radhakrishnan, Menaka and Won, Daehan},
      journal={Knowledge-Based Systems},
      volume={235},
      pages={107687},
      year={2022},
      publisher={Elsevier}
    }
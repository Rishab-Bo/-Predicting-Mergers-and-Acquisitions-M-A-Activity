# Predicting Mergers & Acquisitions (M&A) Signals from SEC Filings

## Project Overview

This project implements an end-to-end financial data analytics (FDA) pipeline to predict M&A activity. The model leverages a multimodal approach, integrating unstructured textual data and structured financial data extracted from public SEC EDGAR filings. The primary objective is to build and evaluate a suite of machine learning models that can identify filings containing strong signals of potential M&A events, acting as a powerful screening tool for financial analysts.

This project directly addresses the objectives and feedback outlined in the **Task ID FDA-8** course assignment.

### Key Features
- **Large-Scale Data Acquisition:** Fetches and processes thousands of 10-K, 10-Q, and 8-K filings for over 100 publicly traded companies.
- **Ground Truth Generation:** Implements a robust process to scan historical 8-K filings for actual M&A announcements, creating a reliable, real-world target variable for supervised learning.
- **Interactive Dashboard:** Includes a web-based dashboard built with Streamlit for EDA, model performance comparison, and scoring new filings.
- **Comparative Modeling:** Trains, evaluates, and compares four distinct ML models: a Logistic Regression baseline, an advanced XGBoost model, the novel TabPFN transformer, and a complex LSTM time-series model.
- **Automated Prediction Pipelines:** Delivers saved, production-ready models and reusable pipeline functions.

## Project Structure
```
M-A_Prediction/
│
├── data/
│   ├── ma_secondary_dataset.csv     # The main, large dataset of filing features
│   └── real_ma_events.csv           # The smaller, ground-truth dataset of M&A announcements
│
├── outputs/
│   └── models/                      # Saved model artifacts and scaler
│
├── images/                          # (Optional) Folder for README images
│
├── companies_list.csv               # Input list of company tickers to analyze
├── Final_Dataset_Creation.ipynb     # Notebook for Deliverable 1: Data Extraction
├── Deliverable_2_EDA_and_Preprocessing.ipynb # Notebook for EDA and Ground Truth Labeling
├── Deliverable_2_Advanced_Modeling_and_Pipelines.ipynb # Notebook for ML Modeling
├── Find_Real_MA_Events.py           # Script to generate the ground-truth M&A event list
├── dashboard.py                     # The Streamlit interactive dashboard application
├── requirements.txt                 # A list of all required Python packages
└── README.md                        # This documentation file
```

## Setup and Installation

**1. Create a Virtual Environment (Recommended):**
```bash
# Navigate to the project directory
cd path/to/M-A_Prediction

# Create a virtual environment
python -m venv BDA

# Activate the environment (Windows)
BDA\Scripts\activate
```

**2. Install Dependencies:**
Install all required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## Execution Workflow

The project is divided into a data generation phase and an interactive analysis phase.

### Phase 1: Data and Model Generation (Run Once)

This phase generates the datasets and trains the models. It only needs to be run once.

**Step 1: Create the Main Feature Dataset (Deliverable 1)**
- **Run:** `Final_Dataset_Creation.ipynb`
- **Action:** This notebook connects to the SEC EDGAR APIs to download and process thousands of filings. **This step will take several hours to complete.**

**Step 2: Find Real M&A Events (Ground Truth)**
- **Run:** `Find_Real_MA_Events.py`
- **Action:** This script scans the filing history for all companies to find actual M&A announcements. Remember to set your email in the `SEC_EMAIL` variable inside the script.

**Step 3: Train Models and Create Pipelines (Deliverable 2)**
- **Run:** `Deliverable_2_Advanced_Modeling_and_Pipelines.ipynb` from top to bottom.
- **Action:** This notebook performs the final data labeling, trains all four machine learning models, and saves the final model artifacts to the `outputs/models/` directory.

### Phase 2: Interactive Dashboard

After completing Phase 1, you can launch the interactive dashboard to explore the results and score new filings.

- **Run this command in your terminal:**
  ```bash
  streamlit run dashboard.py
  ```
- **Action:** This will launch a local web server and open the interactive dashboard in your browser. You can use the controls in the sidebar to switch between models and explore the different tabs.

---

## Methodology and Results Summary

### Model Performance
Four models were trained to predict the `real_target` variable. The key evaluation metrics are for the minority class ("Pre-M&A").

*(Image: Screenshot of the Model Comparison Graph from the notebook/dashboard would be placed here)*

| Model                   | Accuracy | Precision (Pre-M&A) | Recall (Pre-M&A) | F1-Score (Pre-M&A) |
| :---------------------- | :------- | :------------------ | :--------------- | :----------------- |
| Logistic Regression     | 61.05%   | 0.17                | **0.63**         | 0.27               |
| **XGBoost Classifier**  | **88.41%** | **0.48**            | 0.32             | **0.38**           |
| TabPFN (subsample)      | 90.22%   | 0.83                | 0.16             | 0.27               |
| LSTM (Time Series)      | 90.64%   | 0.00                | 0.00             | 0.00               |

#### Analysis of Results:
- **XGBoost** provided the best overall performance with the highest F1-Score, indicating a strong balance between identifying true signals and not missing them. It is the most reliable and practical model for this task.
- **Logistic Regression** served as a good baseline, excelling at recall but suffering from very low precision.
- **TabPFN** was overly conservative, achieving high precision but missing the vast majority of true M&A signals.
- **LSTM** failed on this task. Its high accuracy was misleading, as it learned to only predict the majority class, resulting in 0% recall for the rare "Pre-M&A" signals.

### Automated Prediction Pipeline
The project culminates in the **Score New Filing** tab of the Streamlit dashboard, which serves as the user interface for the automated pipeline. Users can input features for a hypothetical filing, select a model, and receive an instant prediction and probability score, demonstrating a fully operational end-to-end system.
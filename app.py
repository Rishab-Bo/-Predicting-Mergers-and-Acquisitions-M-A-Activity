import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf

st.set_page_config(layout="wide", page_title="M&A Signal Prediction Dashboard")

DATA_PATH = "data/ma_secondary_dataset.csv"
EVENTS_PATH = "data/real_ma_events.csv"
MODELS_DIR = "outputs/models"

TABULAR_FEATURES = [
    'ma_mentions_in_filing',
    'ma_sentiment_in_filing',
    'company_current_ratio',
    'company_debt_to_equity'
]
SEQUENCE_LENGTH = 4

@st.cache_data
def load_and_label_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Please run the `Final_Dataset_Creation.ipynb` notebook first.")
        return None
    
    df = pd.read_csv(DATA_PATH)
    df['filing_date'] = pd.to_datetime(df['filing_date'])

    if not os.path.exists(EVENTS_PATH):
        st.warning(f"Ground truth file not found at {EVENTS_PATH}. The 'real_target' will be all zeros. Please run `Find_Real_MA_Events.py`.")
        df['real_target'] = 0
    else:
        events_df = pd.read_csv(EVENTS_PATH)
        events_df['event_date'] = pd.to_datetime(events_df['event_date'])
        
        df['real_target'] = 0
        
        for _, event in events_df.iterrows():
            cik = event['cik']
            event_date = event['event_date']
            one_year_prior = event_date - pd.Timedelta(days=365)
            
            mask = (
                (df['cik'] == cik) &
                (df['filing_date'] >= one_year_prior) &
                (df['filing_date'] < event_date)
            )
            df.loc[mask, 'real_target'] = 1
            
    return df

@st.cache_resource
def load_models_and_scaler():
    """Loads all saved models (joblib and keras) and the scaler."""
    models = {}
    
    sklearn_models = {
        "Logistic Regression": "ma_prediction_lr_model.joblib",
        "XGBoost": "ma_prediction_xgb_model.joblib",
        "TabPFN (subsample)": "ma_prediction_tabpfn_model.joblib"
    }
    for name, fname in sklearn_models.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None
    
    lstm_path = os.path.join(MODELS_DIR, "ma_prediction_lstm_model.keras")
    if os.path.exists(lstm_path):
        models["LSTM"] = tf.keras.models.load_model(lstm_path)
    else:
        models["LSTM"] = None

    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return models, scaler

def plot_confusion_matrix(y_true, y_pred):
    """Generates a matplotlib confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=['Normal', 'Pre-M&A'], yticklabels=['Normal', 'Pre-M&A'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_roc_curve(y_true, y_proba):
    """Generates a matplotlib ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    return fig

st.title("📊 M&A Signal Prediction Dashboard")
st.markdown("An interactive dashboard to explore the dataset and evaluate machine learning models for predicting M&A signals from SEC filings.")

with st.spinner("Loading and preparing data... This may take a moment."):
    df = load_and_label_data()
    models, scaler = load_models_and_scaler()

if df is not None:
    st.sidebar.header("⚙️ Controls")
    available_models = [name for name, model in models.items() if model is not None]
    if not available_models:
        st.sidebar.error("No models were found in the `outputs/models` directory.")
        model_choice = None
    else:
        model_choice = st.sidebar.selectbox("Choose a Model to Evaluate", options=available_models)
    
    if model_choice and model_choice != "TabPFN (subsample)":
        threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    else:
        threshold = 0.5


    tabs = st.tabs(["📈 Overview", "🔍 Exploratory Data Analysis (EDA)", "🤖 Model Evaluation", "💡 Feature Importance", "✍️ Score New Filing"])

    with tabs[0]:
        st.subheader("Project Dataset Overview")
        total_filings = len(df)
        positive_filings = df["real_target"].sum()
        negative_filings = total_filings - positive_filings
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Filings Analyzed", total_filings)
        c2.metric("Pre-M&A Filings (Positive Class)", int(positive_filings))
        c3.metric("Normal Filings (Negative Class)", int(negative_filings))

        st.markdown("### Class Distribution")
        fig = px.pie(df, names="real_target", title=f"Class Distribution ({positive_filings/total_filings:.2%} Positive)", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Sample Data")
        st.dataframe(df.head(10))

    with tabs[1]:
        st.subheader("Exploratory Data Analysis")
        st.markdown("Compare feature distributions between the two classes.")
        feature_to_explore = st.selectbox("Select a Feature to Explore", TABULAR_FEATURES)
        
        fig = px.box(df, x="real_target", y=feature_to_explore, color="real_target",
                     title=f"Distribution of '{feature_to_explore}' by Class",
                     labels={"real_target": "Class (1 = Pre-M&A)"})
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader(f"Evaluation: {model_choice}")
        
        if model_choice and scaler:
            mdl = models[model_choice]
            
            X_eval = df[TABULAR_FEATURES].fillna(0)
            y_eval = df["real_target"]
            X_eval_scaled = scaler.transform(X_eval)
            
            if model_choice == "LSTM":
                st.info("LSTM evaluation requires creating sequences from the full dataset. This might take a moment.")
                X_sequences, y_sequences = [], []
                df_sorted = df.sort_values(by=['cik', 'filing_date'])
                for cik, group in df_sorted.groupby('cik'):
                    features_scaled_group = scaler.transform(group[TABULAR_FEATURES].fillna(0))
                    targets = group['real_target'].values
                    if len(features_scaled_group) > SEQUENCE_LENGTH:
                        for i in range(len(features_scaled_group) - SEQUENCE_LENGTH):
                            X_sequences.append(features_scaled_group[i:i + SEQUENCE_LENGTH])
                            y_sequences.append(targets[i + SEQUENCE_LENGTH])
                X_eval = np.array(X_sequences)
                y_eval = np.array(y_sequences)
                
                probs = mdl.predict(X_eval).flatten()
            elif model_choice == "TabPFN (subsample)":
                 probs = mdl.predict_proba(X_eval)[:, 1]
            else: 
                probs = mdl.predict_proba(X_eval_scaled)[:, 1]
                
            preds = (probs >= threshold).astype(int)

            st.markdown("### Classification Report")
            report = classification_report(y_eval, preds, target_names=['Normal', 'Pre-M&A'], output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Confusion Matrix")
                st.pyplot(plot_confusion_matrix(y_eval, preds))
            with col2:
                st.markdown("### ROC Curve & AUC")
                st.pyplot(plot_roc_curve(y_eval, probs))

    with tabs[3]:
        st.subheader("Feature Importance (for XGBoost Model)")
        xgb_model = models.get("XGBoost")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": TABULAR_FEATURES,
                "importance": xgb_model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            fig = px.bar(imp, x="importance", y="feature", orientation='h', title="XGBoost Feature Importances")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("XGBoost model not available or feature importances could not be calculated.")

    with tabs[4]:
        st.subheader("Score a New Hypothetical Filing")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Input Features")
            mentions = st.slider("M&A Mentions in Filing", 0, 50, 12)
            sentiment = st.slider("M&A Sentiment in Filing", -1.0, 1.0, 0.72, 0.01)
            current_ratio = st.slider("Company Current Ratio", 0.0, 5.0, 1.8, 0.1)
            debt_to_equity = st.slider("Company Debt-to-Equity", 0.0, 5.0, 0.6, 0.1)
            
            new_filing_data = {
                'ma_mentions_in_filing': mentions,
                'ma_sentiment_in_filing': sentiment,
                'company_current_ratio': current_ratio,
                'company_debt_to_equity': debt_to_equity
            }
        
        if st.button("Predict M&A Signal", type="primary"):
            if model_choice and scaler:
                mdl = models[model_choice]
                
                if model_choice == "LSTM":
                    with col2:
                        st.warning("LSTM requires a sequence of historical data for prediction, not a single filing. This tab is for tabular models only.")
                else:
                    X_new = pd.DataFrame([new_filing_data])
                    if model_choice != "TabPFN (subsample)":
                        X_new_prepared = scaler.transform(X_new)
                    else: 
                        X_new_prepared = X_new
                    
                    prob = mdl.predict_proba(X_new_prepared)[:, 1][0]
                    pred = int(prob >= threshold)
                    label = "Pre-M&A Signal" if pred == 1 else "Normal Signal"

                    with col2:
                        st.markdown("#### Prediction Result")
                        if label == "Pre-M&A Signal":
                            st.error(f"**Prediction: {label}**")
                        else:
                            st.success(f"**Prediction: {label}**")
                        
                        st.metric(label="Probability of being a Pre-M&A Signal", value=f"{prob:.2%}")
                        st.progress(prob)
            else:
                st.error("Model or scaler not available for prediction.")
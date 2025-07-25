# 📦 Streamlit App: Portfolio-Grade Credit Card Fraud Detection

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import numpy as np
import os

# --- Load model and scaler ---
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Page setup ---
st.set_page_config(page_title="💳 Credit Card Fraud Detection Dashboard", layout="wide")
st.title("🚨 Portfolio-Grade Credit Card Fraud Detection App")

# --- Upload section ---
uploaded_file = st.file_uploader("📂 Upload a transaction CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show file info
    st.sidebar.header("📁 File Info")
    st.sidebar.write({
        "Filename": uploaded_file.name,
        "Size (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
        "Rows": len(df)
    })

    # Drop unneeded columns if they exist
    for col in ['Class', 'Time']: 
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scale Amount
    if 'Amount' in df.columns:
        df['Amount'] = scaler.transform(df[['Amount']])

    # Predictions
    y_pred = model.predict(df)
    y_proba = model.predict_proba(df)[:, 1]

    df['Fraud_Prediction'] = y_pred
    df['Fraud_Score'] = y_proba.round(3)

    def risk_action(score):
        if score > 0.85:
            return "🔴 Block"
        elif score > 0.6:
            return "🟠 Monitor"
        else:
            return "🟢 Safe"

    df['Action'] = df['Fraud_Score'].apply(risk_action)

    # --- Sidebar Stats ---
    fraud_count = df['Fraud_Prediction'].sum()
    total = len(df)
    fraud_rate = round((fraud_count / total) * 100, 2)

    st.sidebar.header("📊 Fraud Stats")
    st.sidebar.metric("Total", total)
    st.sidebar.metric("Frauds", fraud_count)
    st.sidebar.metric("Fraud Rate (%)", fraud_rate)

    # --- Data View Filter ---
    st.subheader("🔍 View Transactions")
    view = st.radio("Choose view:", ["All", "Only Fraud", "Only Non-Fraud"], horizontal=True)
    view_df = df if view == "All" else df[df['Fraud_Prediction'] == int(view == "Only Fraud")]
    st.dataframe(view_df.style.applymap(
        lambda v: 'background-color: #ffcccc' if v == 1 else '', subset=['Fraud_Prediction']
    ))

    # --- Visualizations ---
    st.subheader("📊 Fraud Distribution")
    pie_fig = px.pie(
        names=["Non-Fraud", "Fraud"],
        values=[total - fraud_count, fraud_count],
        color_discrete_sequence=["green", "red"]
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("📈 Fraud Amounts by Transaction")
    frauds = df[df['Fraud_Prediction'] == 1]
    if not frauds.empty:
        frauds['Transaction_ID'] = frauds.index
        fig_line = px.scatter(
            frauds, x='Transaction_ID', y='Amount', color='Fraud_Score',
            color_continuous_scale='reds', title="Fraud Amounts & Risk Scores"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # --- SHAP Explainability ---
    st.subheader("🧠 SHAP Explainability")
    try:
        sample = df.drop(columns=['Fraud_Prediction', 'Fraud_Score', 'Action'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample[:100])

        st.write("**Top Feature Contributions (first 100 rows)**")
        shap.summary_plot(shap_values[1], sample[:100], plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')

        st.write("**SHAP Force Plot (Row 0)**")
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0], sample.iloc[0], matplotlib=True)
        st.pyplot(force_plot.figure)

    except Exception as e:
        st.warning("SHAP explainability couldn't run. Reason: " + str(e))

    # --- Feedback Section ---
    st.subheader("💬 Feedback")
    feedback = st.radio("Was the prediction correct?", ["Yes", "No"])
    if feedback == "No":
        reason = st.text_input("Tell us what went wrong:")
        if reason:
            st.success("Thanks! Feedback saved (not really, this is a demo 😄)")

    # --- Download ---
    st.subheader("⬇️ Download Results")
    st.download_button(
        "Download CSV", df.to_csv(index=False).encode('utf-8'),
        file_name="fraud_predictions.csv", mime="text/csv"
    )

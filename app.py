import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.message import EmailMessage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --------------------------
# ✉️ Email Alert Function
# --------------------------
def send_fraud_alert(fraud_count):
    SENDER_EMAIL = "amanwasiullah@gmail.com"
    RECEIVER_EMAIL = "amanwasiullah@gmail.com"
    APP_PASSWORD = "ealwngeehhaldfnr"  # ✅ Replace with your App Password

    msg = EmailMessage()
    msg['Subject'] = "🚨 Fraud Alert Detected!"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    body = f"⚠️ {fraud_count} fraudulent transactions detected by your model."
    msg.set_content(body)

    try:
        with smtpllib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        st.success("📨 Email sent successfully.")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")

# --------------------------
# 🔁 Load Model and Scaler
# --------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------
# 🌐 Streamlit UI
# --------------------------
st.set_page_config(page_title="💳 Fraud Detector", page_icon="🔍")
st.title("💳 Credit Card Fraud Detection App")

# Input type selector
input_mode = st.radio("Choose input method:", ["📂 Upload CSV File", "✍️ Manual Input"])

# --------------------------
# 📂 CSV Upload Mode
# --------------------------
if input_mode == "📂 Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        if 'Time' in data.columns:
            data.drop(['Time'], axis=1, inplace=True)

        if 'Amount' in data.columns and 'NormalizedAmount' not in data.columns:
            data['NormalizedAmount'] = scaler.transform(data[['Amount']])
            data.drop(['Amount'], axis=1, inplace=True)

        if 'Class' in data.columns:
            actual = data['Class']
            data.drop('Class', axis=1, inplace=True)
        else:
            actual = None

        st.subheader("📊 Data Preview")
        st.dataframe(data.head())

        if st.button("🔍 Predict Fraud"):
            try:
                expected_cols = ['V' + str(i) for i in range(1, 29)] + ['NormalizedAmount']
                missing_cols = [col for col in expected_cols if col not in data.columns]

                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    data = data[expected_cols]
                    predictions = model.predict(data)
                    data['Prediction'] = predictions

                    st.session_state.fraud_count = int((predictions == 1).sum())
                    st.session_state.legit_count = int((predictions == 0).sum())

                    st.success(f"✅ Prediction done. Fraud: {st.session_state.fraud_count}, Legit: {st.session_state.legit_count}")

                    # Charts
                    fig1, ax1 = plt.subplots()
                    ax1.pie([st.session_state.legit_count, st.session_state.fraud_count],
                            labels=["Legit", "Fraud"],
                            autopct='%1.1f%%',
                            colors=["#66b3ff", "#ff6666"], startangle=90)
                    ax1.axis('equal')
                    st.pyplot(fig1)

                    st.bar_chart(pd.Series([st.session_state.legit_count, st.session_state.fraud_count], index=["Legit", "Fraud"]))

                    if actual is not None:
                        st.subheader("📊 Confusion Matrix")
                        cm = confusion_matrix(actual, predictions)
                        fig2, ax2 = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=['Legit', 'Fraud'],
                                    yticklabels=['Legit', 'Fraud'])
                        st.pyplot(fig2)

                        st.text("📈 Classification Report:")
                        st.text(classification_report(actual, predictions, digits=4))

                    st.subheader("📋 Prediction Results")
                    st.dataframe(data)

            except Exception as e:
                st.error(f"❌ Error: {e}")

        if "fraud_count" in st.session_state and st.session_state.fraud_count > 0:
            if st.button("📧 Send Email Alert"):
                send_fraud_alert(st.session_state.fraud_count)

# --------------------------
# ✍️ Manual Input Mode
# --------------------------
else:
    st.subheader("✍️ Enter Transaction Details Manually")

    inputs = []
    for i in range(1, 29):
        val = st.number_input(f"Feature V{i}", value=0.0, step=0.01)
        inputs.append(val)

    amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=0.1)
    norm_amount = scaler.transform(np.array([[amount]]))[0][0]
    inputs.append(norm_amount)

    if st.button("🧠 Predict This Transaction"):
        try:
            prediction = model.predict([inputs])[0]
            st.success("✅ Prediction: FRAUD" if prediction == 1 else "✅ Prediction: LEGIT")
        except Exception as e:
            st.error(f"❌ Error: {e}")

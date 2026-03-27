# ==========================================
# 🏦 Loan Approval Prediction (Advanced UI)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Loan Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.markdown("Predict whether a loan will be approved based on applicant details.")

# ------------------------------
# Create Dataset
# ------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)

    data = {
        "income": np.random.randint(20000, 100000, 300),
        "loan_amount": np.random.randint(5000, 50000, 300),
        "credit_history": np.random.randint(0, 2, 300),
        "education": np.random.choice(["Graduate", "Not Graduate"], 300),
    }

    df = pd.DataFrame(data)

    df["loan_status"] = (
        (df["income"] > 40000) &
        (df["credit_history"] == 1)
    ).astype(int)

    return df

df = load_data()

# ------------------------------
# Dataset Preview
# ------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ------------------------------
# Visualization
# ------------------------------
st.subheader("📈 Data Visualization")

# ------------------------------
# Preprocessing
# ------------------------------
le = LabelEncoder()
df["education"] = le.fit_transform(df["education"])

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Train Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ------------------------------
# Accuracy
# ------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")
st.success(f"{accuracy*100:.2f}%")

# ------------------------------
# User Input
# ------------------------------
st.subheader("🧠 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    income = st.slider("Income (₹)", 20000, 100000, 50000)
    loan_amount = st.slider("Loan Amount (₹)", 5000, 50000, 20000)

with col2:
    credit_history = st.selectbox("Credit History", [0, 1])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

education_encoded = le.transform([education])[0]

input_data = np.array([[income, loan_amount, credit_history, education_encoded]])
input_scaled = scaler.transform(input_data)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Check Loan Status"):
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("🚀 Developed by Harshit Banyal")

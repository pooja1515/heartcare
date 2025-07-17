import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# App title
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Predictor")

# Sidebar for user input
st.sidebar.header("🧑 User Info")
username = st.sidebar.text_input("Enter your name", value="Anonymous")

st.sidebar.header("📊 Input Health Metrics")

age = st.sidebar.slider("Age", 30, 80, 50)
sex_male = st.sidebar.radio("Sex", ["Male", "Female"])
cigsPerDay = st.sidebar.slider("Cigarettes per Day", 0, 40, 5)
totChol = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
sysBP = st.sidebar.slider("Systolic Blood Pressure", 90, 200, 120)
glucose = st.sidebar.slider("Glucose", 60, 300, 100)

model_choice = st.sidebar.selectbox("🧠 Select Model", [
    "Logistic Regression", "Random Forest", "SVM", "KNN"
])

# Prepare input
X_input = pd.DataFrame([{
    "age": age,
    "Sex_male": 1 if sex_male == "Male" else 0,
    "cigsPerDay": cigsPerDay,
    "totChol": totChol,
    "sysBP": sysBP,
    "glucose": glucose
}])

# Dummy training data (to simulate real model training)
df_train = pd.DataFrame({
    "age": np.random.randint(30, 80, 300),
    "Sex_male": np.random.randint(0, 2, 300),
    "cigsPerDay": np.random.randint(0, 40, 300),
    "totChol": np.random.randint(100, 400, 300),
    "sysBP": np.random.randint(90, 200, 300),
    "glucose": np.random.randint(60, 300, 300),
    "TenYearCHD": np.random.randint(0, 2, 300)
})

X = df_train.drop("TenYearCHD", axis=1)
y = df_train["TenYearCHD"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_input_scaled = scaler.transform(X_input)

# Train models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "SVM": SVC(probability=True, class_weight="balanced"),
    "KNN": KNeighborsClassifier()
}

model = models[model_choice]
model.fit(X_scaled, y)

# Predict on user input
if st.button("📍 Submit for Prediction"):
    prediction = model.predict(X_input_scaled)[0]
    try:
        prediction_proba = model.predict_proba(X_input_scaled)[0][1]
    except:
        prediction_proba = None

    st.success(f"✅ Prediction: {'At Risk' if prediction == 1 else 'Not at Risk'}")
    if prediction_proba is not None:
        st.info(f"📊 Risk Probability: {prediction_proba:.2f}")

    # Store predictions in session state
    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = pd.DataFrame(columns=[
            "Timestamp", "Username", "Age", "Sex_male", "Cigarettes_per_Day",
            "Total_Cholesterol", "Systolic_BP", "Glucose",
            "Model", "Prediction", "Risk_Probability"
        ])

    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Username": username,
        "Age": age,
        "Sex_male": 1 if sex_male == "Male" else 0,
        "Cigarettes_per_Day": cigsPerDay,
        "Total_Cholesterol": totChol,
        "Systolic_BP": sysBP,
        "Glucose": glucose,
        "Model": model_choice,
        "Prediction": prediction,
        "Risk_Probability": prediction_proba if prediction_proba is not None else "N/A"
    }

    st.session_state.prediction_log = pd.concat(
        [st.session_state.prediction_log, pd.DataFrame([new_row])],
        ignore_index=True
    )

# Show and download log
if "prediction_log" in st.session_state and not st.session_state.prediction_log.empty:
    st.subheader("📝 Prediction Log")
    st.dataframe(st.session_state.prediction_log)

    csv = st.session_state.prediction_log.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Log as CSV",
        data=csv,
        file_name='heart_predictions.csv',
        mime='text/csv'
    )

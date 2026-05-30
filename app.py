import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# -----------------------
# LOAD DATA
# -----------------------

df = pd.read_csv("mango_quality_dataset.csv")

X = df[['Weight_g', 'Color_Index', 'Sugar_Content_Brix', 'Firmness']]
y = df['Grade']

# -----------------------
# TRAIN TEST SPLIT
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# SCALING
# -----------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# SVM
# -----------------------

svm_model = SVC()

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

svm_f1 = f1_score(y_test, svm_pred, average='weighted')

# -----------------------
# KNN
# -----------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_f1 = f1_score(y_test, knn_pred, average='weighted')

# -----------------------
# MLP
# -----------------------

mlp_model = MLPClassifier(
    hidden_layer_sizes=(20,),
    max_iter=500,
    random_state=42
)

mlp_model.fit(X_train_scaled, y_train)

mlp_pred = mlp_model.predict(X_test_scaled)

mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')

# -----------------------
# PAGE TITLE
# -----------------------

st.title("Export Quality Mango Grading System")

st.write(
    """
    This project grades mangoes into Grade A, Grade B,
    Grade C and Reject categories using Machine Learning.
    """
)

# -----------------------
# DATASET PREVIEW
# -----------------------

st.header("Dataset Preview")

st.dataframe(df.head())

# -----------------------
# DATASET STATISTICS
# -----------------------

st.header("Dataset Statistics")

st.dataframe(df.describe())

# -----------------------
# MODEL COMPARISON
# -----------------------

results = pd.DataFrame({
    "Model": ["SVM", "KNN", "MLP"],
    "Weighted F1 Score": [svm_f1, knn_f1, mlp_f1]
})

st.header("Model Comparison")

st.dataframe(results)

# -----------------------
# GRAPH
# -----------------------

st.header("Performance Graph")

fig, ax = plt.subplots()

ax.bar(results["Model"], results["Weighted F1 Score"])

ax.set_ylabel("Weighted F1 Score")

ax.set_title("Model Performance Comparison")

st.pyplot(fig)

# -----------------------
# BEST MODEL
# -----------------------

st.header("Mango Grade Prediction")

weight = st.number_input("Weight (g)", 100.0, 500.0, 300.0)

color = st.number_input("Color Index", 0.0, 100.0, 70.0)

sugar = st.number_input("Sugar Content (Brix)", 0.0, 30.0, 15.0)

firmness = st.number_input("Firmness", 0.0, 10.0, 7.0)

if st.button("Predict Grade"):

    sample = pd.DataFrame(
        [[weight, color, sugar, firmness]],
        columns=[
            "Weight_g",
            "Color_Index",
            "Sugar_Content_Brix",
            "Firmness"
        ]
    )

    sample_scaled = scaler.transform(sample)

    prediction = mlp_model.predict(sample_scaled)

    st.success(f"Predicted Grade: {prediction[0]}")
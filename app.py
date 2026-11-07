# This script imports and evaluates various machine learning classifiers on a test dataset
# Import Libraries Here
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# Streamlit application for displaying evaluation metrics of different classifiers
st.title("Dry Bean Classification Using 5 Different Classifiers")
st.write("This application evaluates the performance of Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine classifiers on user uploaded test dataset. Additionally, it displays the evaluation metrics for each classifier based on the uploaded test dataset.")

# File uploader for test dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Test Dataset:")
    st.dataframe(test_data)

    # Separate features and target variable from the test dataset
    X_test = test_data.drop(columns=["Class"])
    Y_test = test_data["Class"]

    # Load trained models
    log_reg_model = joblib.load("model/logistic_regression_model.joblib")
    dt_model = joblib.load("model/decision_tree_model.joblib")
    knn_model = joblib.load("model/knn_classifier_model.joblib")
    nb_model = joblib.load("model/naive_bayes_classifier_model.joblib")
    svm_model = joblib.load("model/svm_classifier_model.joblib")

    # Setup a dropdown for selecting the classifier
    classifier_option = st.selectbox(
        "Select Classifier",
        ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine", "All")
    )

    # Use a dictionary to map classifier names to their respective models
    classifiers = {
        "Logistic Regression": log_reg_model,
        "Decision Tree": dt_model,
        "K-Nearest Neighbors": knn_model,
        "Naive Bayes": nb_model,
        "Support Vector Machine": svm_model
    }

    # Function to evaluate and display metrics for a given classifier
    def evaluate_classifier(model, X_test, Y_test):
        Y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
        st.write("Precision:", precision_score(Y_test, Y_pred, average="weighted"))
        st.write("Recall:", recall_score(Y_test, Y_pred, average="weighted"))
        st.write("F1 Score:", f1_score(Y_test, Y_pred, average="weighted"))
        st.write("ROC AUC Score:", roc_auc_score(Y_test, model.predict_proba(X_test), multi_class="ovr"))
        st.write("Matthews Correlation Coefficient:", matthews_corrcoef(Y_test, Y_pred))

    # Evaluate the selected classifier(s)
    if classifier_option == "All":
        # Header for comparison of all models
        st.subheader("Comparison of all 5 models")

        # Empty list to store values
        all_metrics = []

        for name, model in classifiers.items():
            Y_pred = model.predict(X_test) # Run model prediction on test data
            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(Y_test, Y_pred),
                "Precision": precision_score(Y_test, Y_pred, average="weighted"),
                "Recall": recall_score(Y_test, Y_pred, average="weighted"),
                "F1 Score": f1_score(Y_test, Y_pred, average="weighted"),
                "ROC AUC Score": roc_auc_score(Y_test, model.predict_proba(X_test), multi_class="ovr"),
                "Matthews Correlation Coefficient": matthews_corrcoef(Y_test, Y_pred)
            }
            all_metrics.append(metrics)

        st.table(pd.DataFrame(all_metrics))      
    else:
        st.subheader(classifier_option)
        evaluate_classifier(classifiers[classifier_option], X_test, Y_test)

    
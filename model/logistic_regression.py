#Import Libraries Here
from sklearn import linear_model
import joblib
from load_dataset import X_train_scaled, Y_train, X_test_scaled, Y_test
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

#Initialize Logistic Regression Model with specified parameters
log_reg = linear_model.LogisticRegression(
  C = 1.0, # Inverse of regularization strength
  max_iter=2000 # Maximum number of iterations for the solver
)

#Train the model 
log_reg.fit(X_train_scaled, Y_train)

#Get the directory of the current running script
script_dir = Path(__file__).resolve().parent

#Save the trained model to a file
joblib.dump(log_reg, script_dir / 'logistic_regression_model.joblib')

#Indicate that the model has been saved successfully
print("Logistic Regression model saved as 'logistic_regression_model.joblib'")

#Run the model on the test dataset and store the predictions as well as probabilities
Y_pred = log_reg.predict(X_test_scaled)
Y_prob = log_reg.predict_proba(X_test_scaled)

#Evaluate the model's metrics on the test set
log_reg_accuracy = accuracy_score(Y_test, Y_pred) # Accuracy score
log_reg_precision = precision_score(Y_test, Y_pred, average="weighted") # Weighted precision score
log_reg_recall = recall_score(Y_test, Y_pred, average="weighted") # Weighted recall score
log_reg_f1 = f1_score(Y_test, Y_pred, average="weighted") # Weighted F1 score
log_reg_roc_auc = roc_auc_score(Y_test, Y_prob, multi_class="ovr") # ROC AUC score
log_reg_mcc = matthews_corrcoef(Y_test, Y_pred) # Matthews correlation coefficient

#Print the evaluation metrics for the Logistic Regression model
print("\n")
print("Logistic Regression Model Evaluation:")
print("Accuracy:", "%.4f" % log_reg_accuracy)
print("Precision:", "%.4f" % log_reg_precision)
print("Recall:", "%.4f" % log_reg_recall)
print("F1 Score:", "%.4f" % log_reg_f1)
print("ROC AUC Score:", "%.4f" % log_reg_roc_auc)
print("Matthews Correlation Coefficient:", "%.4f" % log_reg_mcc)

#Create a dict to store the evaluation metrics
log_reg_metrics = {
    "Accuracy": log_reg_accuracy,
    "Precision": log_reg_precision,
    "Recall": log_reg_recall,
    "F1 Score": log_reg_f1,
    "ROC AUC Score": log_reg_roc_auc,
    "Matthews Correlation Coefficient": log_reg_mcc
}
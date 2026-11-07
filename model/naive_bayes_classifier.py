#Import Libraries Here
from sklearn import naive_bayes
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

#Initialize Naive Bayes Classifier with specified parameters
nb_clf = naive_bayes.GaussianNB(var_smoothing=1e-9)

#Train the Naive Bayes Classifier
nb_clf.fit(X_train_scaled, Y_train)

#Get the directory of the current running script
script_dir = Path(__file__).resolve().parent

#Save the trained Naive Bayes Classifier to a file
joblib.dump(nb_clf, script_dir / 'naive_bayes_classifier_model.joblib')

#Indicate that the model has been saved successfully
print("Naive Bayes Classifier model saved as 'naive_bayes_classifier_model.joblib'")

#Run the model on the test dataset and store the predictions as well as probabilities
Y_pred = nb_clf.predict(X_test_scaled)
Y_prob = nb_clf.predict_proba(X_test_scaled)

#Evaluate the model's metrics on the test set
nb_accuracy = accuracy_score(Y_test, Y_pred) # Accuracy score
nb_precision = precision_score(Y_test, Y_pred, average="weighted") # Weighted precision score
nb_recall = recall_score(Y_test, Y_pred, average="weighted") # Weighted recall score
nb_f1 = f1_score(Y_test, Y_pred, average="weighted") # Weighted F1 score
nb_roc_auc = roc_auc_score(Y_test, Y_prob, multi_class="ovr") # ROC AUC score
nb_mcc = matthews_corrcoef(Y_test, Y_pred) # Matthews correlation coefficient

#Print the evaluation metrics for the Naive Bayes Classifier
print("\n")
print("Naive Bayes Classifier Model Evaluation:")
print("Accuracy:", "%.4f" % nb_accuracy)
print("Precision:", "%.4f" % nb_precision)
print("Recall:", "%.4f" % nb_recall)
print("F1 Score:", "%.4f" % nb_f1)
print("ROC AUC Score:", "%.4f" % nb_roc_auc)
print("Matthews Correlation Coefficient:", "%.4f" % nb_mcc)

#Create a dict to store the evaluation metrics
nb_metrics = {
    "Accuracy": nb_accuracy,
    "Precision": nb_precision,
    "Recall": nb_recall,
    "F1 Score": nb_f1,
    "ROC AUC Score": nb_roc_auc,
    "Matthews Correlation Coefficient": nb_mcc
}
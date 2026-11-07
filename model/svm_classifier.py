#Import Libraries Here
from sklearn import svm
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

#Initialize Support Vector Machine Classifier with specified parameters
svm_clf = svm.SVC(
    C=1.0, # Regularization parameter
    kernel='rbf', # Kernel type
    probability=True # Enable probability estimates
)

#Train the Support Vector Machine Classifier
svm_clf.fit(X_train_scaled, Y_train)

#Get the directory of the current running script
script_dir = Path(__file__).resolve().parent

#Save the trained Support Vector Machine Classifier to a file
joblib.dump(svm_clf, script_dir / 'svm_classifier_model.joblib')

#Indicate that the model has been saved successfully
print("Support Vector Machine Classifier model saved as 'svm_classifier_model.joblib'")

#Run the model on the test dataset and store the predictions as well as probabilities
Y_pred = svm_clf.predict(X_test_scaled)
Y_prob = svm_clf.predict_proba(X_test_scaled)

#Evaluate the model's metrics on the test set
svm_accuracy = accuracy_score(Y_test, Y_pred) # Accuracy score
svm_precision = precision_score(Y_test, Y_pred, average="weighted") # Weighted precision score
svm_recall = recall_score(Y_test, Y_pred, average="weighted") # Weighted recall score
svm_f1 = f1_score(Y_test, Y_pred, average="weighted") # Weighted F1 score
svm_roc_auc = roc_auc_score(Y_test, Y_prob, multi_class="ovr") # ROC AUC score
svm_mcc = matthews_corrcoef(Y_test, Y_pred) # Matthews correlation coefficient

#Print the evaluation metrics for the Support Vector Machine Classifier
print("\n")
print("Support Vector Machine Classifier Model Evaluation:")
print("Accuracy:", "%.4f" % svm_accuracy)
print("Precision:", "%.4f" % svm_precision)
print("Recall:", "%.4f" % svm_recall)
print("F1 Score:", "%.4f" % svm_f1)
print("ROC AUC Score:", "%.4f" % svm_roc_auc)
print("Matthews Correlation Coefficient:", "%.4f" % svm_mcc)
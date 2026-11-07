#Import Libraries Here
from sklearn import neighbors
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

#Initialize K-Nearest Neighbors Classifier with specified parameters
knn_clf = neighbors.KNeighborsClassifier(
  n_neighbors=5, #Number of neighbors to use
  weights='uniform', #Weight function used in prediction
  algorithm='auto' #Algorithm used to compute the nearest neighbors
)

#Train the K-Nearest Neighbors Classifier
knn_clf.fit(X_train_scaled, Y_train)

#Get the directory of the current running script
script_dir = Path(__file__).resolve().parent

#Save the trained K-Nearest Neighbors Classifier to a file
joblib.dump(knn_clf, script_dir / 'knn_classifier_model.joblib')

#Indicate that the model has been saved successfully
print("K-Nearest Neighbors Classifier model saved as 'knn_classifier_model.joblib'")

#Run the model on the test dataset and store the predictions as well as probabilities
Y_pred = knn_clf.predict(X_test_scaled)
Y_prob = knn_clf.predict_proba(X_test_scaled)

#Evaluate the model's metrics on the test set
knn_accuracy = accuracy_score(Y_test, Y_pred) # Accuracy score
knn_precision = precision_score(Y_test, Y_pred, average="weighted") # Weighted precision score
knn_recall = recall_score(Y_test, Y_pred, average="weighted") # Weighted recall score
knn_f1 = f1_score(Y_test, Y_pred, average="weighted") # Weighted F1 score
knn_roc_auc = roc_auc_score(Y_test, Y_prob, multi_class="ovr") # ROC AUC score
knn_mcc = matthews_corrcoef(Y_test, Y_pred) # Matthews correlation coefficient

#Print the evaluation metrics for the K-Nearest Neighbors Classifier
print("\n")
print("K-Nearest Neighbors Classifier Model Evaluation:")
print("Accuracy:", "%.4f" % knn_accuracy)
print("Precision:", "%.4f" % knn_precision)
print("Recall:", "%.4f" % knn_recall)
print("F1 Score:", "%.4f" % knn_f1)
print("ROC AUC Score:", "%.4f" % knn_roc_auc)
print("Matthews Correlation Coefficient:", "%.4f" % knn_mcc)
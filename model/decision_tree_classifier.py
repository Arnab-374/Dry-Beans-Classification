#Import Libriaries Here
from load_dataset import X_train_scaled, Y_train, X_test_scaled, Y_test
from sklearn import tree
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

#Initialize Decision Tree Classifier with specified parameters
decision_tree = tree.DecisionTreeClassifier(
  criterion="entropy", # Use entropy as the splitting criterion
  max_depth=15, # Maximum depth of the tree
  random_state=42, # Seed for reproducibility
  min_samples_split=20, # Minimum number of samples required to split an internal node
  min_samples_leaf=10 # Minimum number of samples required to be at a leaf node
)

#Train the Decision Tree Classifier
decision_tree.fit(X_train_scaled, Y_train)

#Get the directory of the current running script
script_dir = Path(__file__).resolve().parent

#Save the trained model to a file
joblib.dump(decision_tree, script_dir / 'decision_tree_model.joblib')

#Indicate that the model has been saved successfully
print("Decision Tree model saved as 'decision_tree_model.joblib'")

#Run the model on the test dataset and store the predictions as well as probabilities
Y_pred = decision_tree.predict(X_test_scaled)
Y_prob = decision_tree.predict_proba(X_test_scaled)

#Evaluate the model's metrics on the test set
dt_accuracy = accuracy_score(Y_test, Y_pred) # Accuracy score
dt_precision = precision_score(Y_test, Y_pred, average="weighted") # Weighted precision score
dt_recall = recall_score(Y_test, Y_pred, average="weighted") # Weighted recall score
dt_f1 = f1_score(Y_test, Y_pred, average="weighted") # Weighted F1 score
dt_roc_auc = roc_auc_score(Y_test, Y_prob, multi_class="ovr") # ROC AUC score
dt_mcc = matthews_corrcoef(Y_test, Y_pred) # Matthews correlation coefficient

#Print the evaluation metrics for the Decision Tree model
print("\n")
print("Decision Tree Model Evaluation:")
print("Accuracy:", "%.4f" % dt_accuracy)
print("Precision:", "%.4f" % dt_precision)
print("Recall:", "%.4f" % dt_recall)
print("F1 Score:", "%.4f" % dt_f1)
print("ROC AUC Score:", "%.4f" % dt_roc_auc)
print("Matthews Correlation Coefficient:", "%.4f" % dt_mcc)

#Create a dict to store the evaluation metrics
dt_metrics = {
    "Accuracy": dt_accuracy,
    "Precision": dt_precision,
    "Recall": dt_recall,
    "F1 Score": dt_f1,
    "ROC AUC Score": dt_roc_auc,
    "Matthews Correlation Coefficient": dt_mcc
}
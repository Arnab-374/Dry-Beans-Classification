#import Libraries
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Get the directory of the current running script
script_dir = Path(__file__).resolve().parent

#Get the directory of where the data is stored
data_path = str(script_dir.parent) + "\\original_data.csv"

#Fetch the dataset using pandas.read_excel
dataset = pd.read_csv(data_path)

#Display first 5 rows of the dataset
print("First 5 Rows ->")
print(dataset.head(5))

print("\n")
#Display dataset shape
print("Shape")
print(dataset.shape)

print("\n")
#Display dataset summary
print("Summary")
print(dataset.info())

print("\n")
#Display dataset null value count
print("Null Value Count")
print(dataset.isnull().sum())

#Convert the dataset into X_data and Y_data - X_data are all the features while Y_data is the original Class
X_data = dataset.drop(columns = ["Class"])
Y_data = dataset["Class"]

#Check the shapre of X_data and Y_data
print("\n")
print("X_data Shape")
print(X_data.shape)
print("\n")
print("Y_data Shape")
print(Y_data.shape)

#Split the X_data and Y_data into train and test data. In order to train the models effectively, use stratified sampling method to maintain class proportions
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, stratify= Y_data, random_state=42)

#Preprocess the X_train and X_test data using StandardScaler to make all the input features in the 0 to 1 scale
scaler = StandardScaler() #Initialize the Scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Showcase the scaled data
print("\n")
print("X_train_data Scaled")
print(X_train_scaled[:5])
print("\n")
print("X_test_data Scaled")
print(X_test_scaled[:5])
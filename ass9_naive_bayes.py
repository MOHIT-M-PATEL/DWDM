# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (Contains glucose, bloodpressure, diabetes)
data = pd.read_csv('Naive-Bayes-Classification-Data.csv')

# Splitting data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one are features
y = data.iloc[:, -1]   # The last column is the target class

# Display frequency tables for each feature with respect to the target class
def generate_frequency_tables(df, target):
    # For each feature in the dataframe
    for col in df.columns:
        print(f"\nFrequency Table for {col} with respect to {target.name}:\n")
        # Creating a crosstab (frequency table)
        freq_table = pd.crosstab(df[col], target)
        print(freq_table)

# Call the function to generate frequency tables
generate_frequency_tables(X, y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the Naive Bayes classifier
nb_classifier = GaussianNB()

# Training the classifier
nb_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluating the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Printing the results
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
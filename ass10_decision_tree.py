# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('decision_tree_data.csv')

# Preview the dataset
print(data.head())

# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False)  # Updated to use sparse_output
X_encoded = encoder.fit_transform(data.iloc[:, :-1])  # encode features
y = data.iloc[:, -1].values  # target class

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Initializing the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Training the classifier
dt_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluating the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Printing the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Plotting the Decision Tree with improved layout
plt.figure(figsize=(20, 10))  # Larger figure for better clarity
plot_tree(dt_classifier, 
          filled=True, 
          feature_names=encoder.get_feature_names_out(), 
          class_names=np.unique(y).astype(str), 
          rounded=True, 
          proportion=True,  # Proportional node sizes
          fontsize=10)      # Slightly larger font size for better visibility
plt.title("Decision Tree Visualization", fontsize=16)

# Apply tight layout for better organization
plt.tight_layout()

# Save the improved plot
plt.savefig("decision_tree_visualization_improved.png")

# Display the plot
plt.show()


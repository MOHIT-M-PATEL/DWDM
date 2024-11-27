# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('decision_tree_data.csv')

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

# Printing concise results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)

# Plotting the Decision Tree with necessary details only
plt.figure(figsize=(16, 8))  # Moderate size for clarity
plot_tree(dt_classifier, 
          filled=True, 
          feature_names=encoder.get_feature_names_out(), 
          class_names=np.unique(y).astype(str), 
          rounded=True, 
          fontsize=8)  # Reduced font size for minimal output
plt.title("Decision Tree Visualization", fontsize=14)

# Save the plot
plt.tight_layout()
plt.savefig("decision_tree_visualization.png")

# Display the plot
plt.show()

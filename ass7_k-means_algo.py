import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load the dataset
df = pd.read_csv("kmeans_dataset.csv")
# Select features for clustering
features = df[["Years of Experience", "Work Hours/Week"]]
# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# Define the number of clusters
num_clusters = 3 # Adjust as needed
# Create KMeans model
kmeans = KMeans(n_clusters=num_clusters)
# Fit the model
kmeans.fit(scaled_features)
# Get the predicted cluster labels
df['Predicted Cluster'] = kmeans.labels_
# Get the cluster centers (in scaled space)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
# Plot the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Years of Experience'], df['Work Hours/Week'],
 c=df['Predicted Cluster'], s=30, cmap='viridis', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', 
label='Centroids')
plt.title('K-means Clustering Results')
plt.xlabel('Years of Experience')
plt.ylabel('Work Hours/Week')
plt.legend()
plt.grid()
plt.show()

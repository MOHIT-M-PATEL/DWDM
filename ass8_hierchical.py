import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Load the dataset from a CSV file

data = pd.read_csv('customer_data.csv')  

# Converting the data into a numpy array for processing
X = data.values

# Plotting the Dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.xlabel("Data points")
plt.ylabel("Euclidean distances")
plt.show()

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')

# Fitting the model
y_hc = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')

plt.title("Clusters of data points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

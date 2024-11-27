import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import quote

# Encode special characters in password
password = quote("mysqlroot@1234567")  # Replace with your actual password
# Establish connection to the MySQL database using SQLAlchemy
engine = create_engine(f"mysql+mysqlconnector://root:{password}@localhost/student_data_db")

# Define the query to fetch data from the MySQL table
query = """
SELECT G1, G2, G3, activities, studytime, freetime, absences
FROM students; 
"""

# Load dataset from MySQL into a pandas DataFrame
student_data = pd.read_sql(query, con=engine)

# Handle categorical column 'activities' (yes/no -> 1/0)
student_data['activities'] = student_data['activities'].apply(lambda x: 1 if x == 'yes' else 0)

# Scale the numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(student_data)

# Calculate Inertia for the Elbow chart
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow chart
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-', markersize=8)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Calculate Silhouette Score for each possible cluster number (from 2 to 10)
silhouette_scores = {}
K = range(2, 11)  # Minimum 2 clusters for silhouette score calculation

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores[k] = score

# Display Silhouette Scores
print("Silhouette Scores for different clusters:")
for k, score in silhouette_scores.items():
    print(f"{k} clusters: {score:.4f}")

# Take user input for the preferred number of clusters
num_clusters = int(input("Enter the preferred number of clusters based on Silhouette Score: "))

# Perform K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_data)
student_data['Cluster'] = kmeans.labels_

# Visualize the clusters (using G1 and G2 for example)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=student_data['G1'], y=student_data['G2'], hue=student_data['Cluster'], palette='Set1', s=100)
plt.title(f'K-Means Clustering of Students with {num_clusters} Clusters')
plt.xlabel('G1 - First Period Grade')
plt.ylabel('G2 - Second Period Grade')
plt.show()

# Take user input for a new data point to determine which cluster it belongs to
print("\nEnter new student data to find its cluster:")
g1 = float(input("G1 (First Period Grade): "))
g2 = float(input("G2 (Second Period Grade): "))
g3 = float(input("G3 (Final Grade): "))
activities = 1 if input("Participates in Activities? (yes/no): ").lower() == "yes" else 0
studytime = float(input("Study Time: "))
freetime = float(input("Free Time: "))
absences = int(input("Number of Absences: "))

# Scale the input data in DataFrame format to match the feature names
new_data = pd.DataFrame([[g1, g2, g3, activities, studytime, freetime, absences]],
                        columns=['G1', 'G2', 'G3', 'activities', 'studytime', 'freetime', 'absences'])
scaled_new_data = scaler.transform(new_data)

# Predict the cluster for the new data point
cluster_label = kmeans.predict(scaled_new_data)[0]
print(f"The new student data point belongs to cluster: {cluster_label}")

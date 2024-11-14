import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Data generation (assuming data is ready and loaded into the df variable)
# Age, income, and loan amount are defined in integers
data = {
    'age': np.random.randint(20, 60, 100),
    'income': np.random.randint(30000000, 100000000, 100),
    'loan_amount': np.random.randint(5000000, 40000000, 100),
    'transaction_count': np.random.randint(10, 100, 100)
}
df = pd.DataFrame(data)

# 2. Data preprocessing
# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Determining the optimal number of clusters using silhouette score
silhouette_scores = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

# Select the number of clusters with the highest silhouette score
best_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {best_k}")

# 4. Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

# 5. Analyzing the results
# To calculate Lift, identify the cluster with the highest average loan amount as the target cluster
lift_values = df.groupby('cluster')['loan_amount'].mean()
target_cluster = lift_values.idxmax()
print(f"Target cluster for high loan likelihood: {target_cluster}")
print("Lift values for each cluster:")
print(lift_values / df['loan_amount'].mean())

# 6. Displaying results as integers in a formatted table
result = df.groupby('cluster').mean().round(0).astype(int)
print(result.to_string(index=True, justify="center", col_space=10, header=True))

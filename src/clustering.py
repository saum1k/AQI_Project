import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("outputs/cleaned_aqi.csv")

# features for clustering
features = ['PM2.5','PM10','NO2','SO2']

# scale data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[features])

# apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = kmeans.fit_predict(scaled_data)

# save dataset with cluster labels
df.to_csv("outputs/models/city_clusters.csv", index=False)

# visualize clusters
plt.figure()

plt.scatter(df['PM2.5'], df['PM10'], c=df['Cluster'])

plt.title("City Clusters based on Pollution")

plt.xlabel("PM2.5")

plt.ylabel("PM10")

plt.savefig("outputs/graphs/clusters.png")

plt.show()

print("Clustering completed")
print("Clustered dataset saved in outputs/models")
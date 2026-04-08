import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("outputs/cleaned_aqi.csv")

# features for PCA
features = ['PM2.5','PM10','NO2','SO2']

# scale data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[features])

# apply PCA
pca = PCA(n_components=2)

principal_components = pca.fit_transform(scaled_data)

# create dataframe
pca_df = pd.DataFrame(data=principal_components,
                      columns=['PC1','PC2'])

# plot PCA result
plt.figure()

plt.scatter(pca_df['PC1'], pca_df['PC2'])

plt.title("PCA Visualization")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.savefig("outputs/graphs/pca_plot.png")

plt.show()

print("PCA completed and graph saved")
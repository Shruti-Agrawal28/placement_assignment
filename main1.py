from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("data.csv")
data = pd.read_csv("data_2genre.csv")
data["label"].unique()
data["label"] = data["label"].replace(1, "pop")
data["label"] = data["label"].replace(2, "classical")
df = pd.concat([df, data], ignore_index=True)
le = LabelEncoder()
df["filename"] = le.fit_transform(df["filename"])
df["label"] = le.fit_transform(df["label"])
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_df)
df['cluster_label'] = kmeans.labels_
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(df)
cluster_stats = df.groupby('cluster_label').mean()
print(cluster_stats)

X = df.drop("cluster_label", axis=1)
y = df["cluster_label"]
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform K-means clustering on the training set
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predict cluster labels for the test set
y_pred = rfc.predict(X_test)

# Evaluate the performance of the clustering algorithm
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


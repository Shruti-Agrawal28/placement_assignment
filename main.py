import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

#loading the dataset
df = pd.read_csv("data.csv")
data = pd.read_csv("data_2genre.csv")

data["label"].unique()
data["label"] = data["label"].replace(1, "pop")
data["label"] = data["label"].replace(2, "classical")
df = pd.concat([df, data], ignore_index=True)
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])


# Splitting the dataset into features (X) and labels (y)
X = df.drop(["filename", "label"], axis=1) # Replace with the actual feature columns
y = df['label']  # Replace with the actual label column

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
k = 5  # Replace with your desired number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train_scaled, y_train)

# Predict labels for the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
report = classification_report(y_test, y_pred)

# Display the classification report in Streamlit
st.text("Classification Report:")
st.text(report)


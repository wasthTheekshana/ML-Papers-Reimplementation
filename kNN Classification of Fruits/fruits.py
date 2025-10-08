import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Sample dataset: Fruits (weight in grams, sugar content g/100g)
data = {
    "Weight": [150, 170, 140, 130, 180, 120, 200, 160],
    "Sugar":  [10, 12,  9,   8,   14,  7,   15, 11],
    "Label":  ["Apple", "Apple", "Apple", "Apple",
               "Banana", "Banana", "Banana", "Banana"]
}

df = pd.DataFrame(data)

X = df[["Weight", "Sugar"]].values
y = df["Label"].values

# Split into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train kNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = knn.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Predictions:", y_pred)

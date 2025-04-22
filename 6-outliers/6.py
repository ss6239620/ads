import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("6.csv")  # Update with your actual file path if needed

# Select numeric columns only (DBSCAN requires numerical input)
X = df.select_dtypes(include=['int64', 'float64'])

# Standardize features (important for DBSCAN performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # You may need to tune eps/min_samples
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add DBSCAN labels to the original DataFrame
df["DBSCAN_Label"] = dbscan_labels

# Separate inliers and outliers
inliers = df[df["DBSCAN_Label"] != -1]
outliers = df[df["DBSCAN_Label"] == -1]

# Plot Before Outlier Detection
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', label='All Data')
plt.title("Before Outlier Detection")
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend()

# Plot After Outlier Detection
plt.subplot(1, 2, 2)
plt.scatter(inliers.iloc[:, 0], inliers.iloc[:, 1], c='green', label='Inliers')
plt.scatter(outliers.iloc[:, 0], outliers.iloc[:, 1], c='red', label='Outliers')
plt.title("After Outlier Detection with DBSCAN")
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend()

plt.tight_layout()
plt.show()

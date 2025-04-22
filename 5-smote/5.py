import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("5.csv")

# Separate features and target
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

# Plot class distribution before SMOTE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title('Before SMOTE')
plt.xlabel('Diabetes')
plt.ylabel('Count')

print("Original class distribution:", Counter(y))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Resampled class distribution:", Counter(y_resampled))

# Plot class distribution after SMOTE
plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled)
plt.title('After SMOTE')
plt.xlabel('Diabetes')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

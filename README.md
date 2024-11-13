import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("First 10 rows of the Iris dataset:")
print(df.head(10))

print("\nDataset Summary:")
print(f"Number of instances: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"Target classes: {np.unique(df['target'])")

for column in df.columns[:-1]:
    print(f"{column} - Data type: {df[column].dtype}, Min: {df[column].min()}, Max: {df[column].max()}")


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

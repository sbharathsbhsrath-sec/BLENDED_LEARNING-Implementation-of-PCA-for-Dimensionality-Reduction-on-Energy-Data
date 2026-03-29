# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Loading and Selection
The program uses pandas to read the dataset HeightsWeights.csv and selects two features: Height (Inches) and Weight (Pounds) for analysis. 
2. Data Standardization
The StandardScaler from scikit-learn is used to scale the features so that they have a mean of 0 and standard deviation of 1, which is important before applying PCA.
3.Applying PCA
PCA (Principal Component Analysis) is applied with 2 components to transform the original features into principal components (PC1 and PC2) that capture the maximum variance in the dataset. 
4.Visualization of Results
The transformed data is plotted using Matplotlib and Seaborn in a scatter plot, showing how the data points are distributed based on the new principal components. 

## Program:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("HeightsWeights.csv")
print(data.head())
print(data.columns)
X = data[['Height(Inches)', 'Weight(Pounds)']]  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print("\nName: bharath S")
print("Reg No: 212225230031\n")
print("Explained Variance Ratio for each Principal Component:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Heights and Weights Dataset")
plt.show()

```

## Output:
<img width="959" height="812" alt="image" src="https://github.com/user-attachments/assets/84354046-7bb1-4496-92a2-b208a2774fa7" />



## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import numpy as np

df = pd.read_csv('CustomerData.csv')
df.head()

df.loc[df['Age'] > 100, 'Age'] = np.nan

df['Age'] = df['Age'].fillna(int(df['Age'].mean()))
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Country'] = df['Country'].fillna(df['Country'].mode()[0])
df['Employed'] = df['Employed'].fillna(df['Employed'].mode()[0])
df['Income'] = df['Income'].fillna(df['Income'].mean())
df['ItemsPurchased(monthly)'] = df['ItemsPurchased(monthly)'].fillna(int(df['ItemsPurchased(monthly)'].mean()))
df['ProductType'] = df['ProductType'].fillna(df['ProductType'].mode()[0])
df['PaymentType'] = df['PaymentType'].fillna(df['PaymentType'].mode()[0])
df['Mode'] = df['Mode'].fillna(df['Mode'].mode()[0])

df[['Gender', 'Country', 'Employed', 'ProductType', 'PaymentType', 'Mode']] = df[['Gender', 'Country', 'Employed', 'ProductType', 'PaymentType', 'Mode']].apply(LabelEncoder().fit_transform)
df.head()

X = df[['Age','Gender','Income','ItemsPurchased(monthly)']]
y = df['Mode']

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
X_normalized = normalize(X_scaled)

X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2']

#Hierarchical Clustering
plt.figure(figsize =(8, 8)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))

#AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean')
plt.figure() 
plt.scatter(X_principal['P1'], X_principal['P2'], c = agg.fit_predict(X_principal)) 
plt.show()

#Density Based Clustering
dbscan = DBSCAN(eps=0.1, min_samples=3)
plt.scatter(X_principal['P1'], X_principal['P2'], c = dbscan.fit_predict(X_principal))
plt.figure()
plt.show()

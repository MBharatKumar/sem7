#KMeans
import pandas as pd
from sklearn import datasets

from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = pd.DataFrame(iris.target, columns= ["Targets"])

model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

import numpy as np
color_map = np.array(['red','green','blue'])
import matplotlib.pyplot as plt
plt.scatter(X.petal_length, X.petal_width, c = color_map[y.Targets], s=40)

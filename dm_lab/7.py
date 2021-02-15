#KMeans
import pandas as pd
from sklearn import datasets

from sklearn.cluster import KMeans

x = pd.DataFrame(datasets.load_iris().data, columns = datasets.load_iris().feature_names)
y = pd.DataFrame(datasets.load_iris().target, columns = ['Targets'])

model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

import numpy as np
color_map = np.array(['red','green','blue'])
import matplotlib.pyplot as plt
plt.scatter(x['petal length (cm)'], x['petal width (cm)'], c = color_map[y.Targets], s=40)
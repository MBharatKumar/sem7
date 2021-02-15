#dbscan
import pandas as pd
from sklearn import datasets
from sklearn.cluster import DBSCAN

x = pd.DataFrame(datasets.load_iris().data, columns = datasets.load_iris().feature_names)
y = pd.DataFrame(datasets.load_iris().target, columns = ['Targets'])

Model = DBSCAN(eps=0.1, min_samples=3)
Model.fit(x)

import numpy as np
color_map = np.array(['red','green','blue'])
import matplotlib.pyplot as plt
plt.scatter(x['petal length (cm)'], x['petal width (cm)'], c = color_map[y.Targets], s=40)

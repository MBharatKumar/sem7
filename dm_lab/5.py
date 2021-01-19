# Decision Tree
import pandas as pd
from sklearn import datasets
import numpy as np

irisdata = pd.DataFrame(datasets.load_iris().data, columns = datasets.load_iris().feature_names )
irisdata

iristarget = pd.DataFrame(datasets.load_iris().target)
iristarget

from sklearn import tree
model = tree.DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(irisdata, iristarget, test_size=0.4,random_state=42)

iris_model = model.fit(X_train, y_train)

pred = iris_model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, iris_model.predict(X_test))

from sklearn import tree
tree.plot_tree(iris_model)

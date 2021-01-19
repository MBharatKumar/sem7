# Random Forest
import pandas as pd
from sklearn import datasets

irisdata = pd.DataFrame(datasets.load_iris().data, columns = datasets.load_iris().feature_names)

iristarget = pd.DataFrame(datasets.load_iris().target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(irisdata, iristarget, test_size=0.4, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

clf = model.fit(X_train, y_train)
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('CustomerData.csv')
sns.histplot(data = df["age"])

sns.boxplot(x = df['age'])

#Removing outliers in age as person with age < 17 does not have a stable earning
df.drop(df[df['age'] < 17].index, inplace = True)

sns.histplot(data = df["age"])

sns.scatterplot(data = df, x = df.age, y = df['annual income (lakhs)'])
df.reset_index(drop=True, inplace = True)

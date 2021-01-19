import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('CustomerData.csv')
df.head()

#Handle missing values
df.isnull().sum()
df.gender.fillna(df.gender.mode()[0], inplace = True)
df.category.fillna(df.category.mode()[0], inplace = True)
df.age.fillna(int(df.age.mean()), inplace = True)
df['annual income (lakhs)'].fillna(df['annual income (lakhs)'].mean(), inplace = True)
df.isnull().sum()

#Encode categorical values to numerical values
encoder = preprocessing.LabelEncoder()
df[['category', 'purchase type ', 'gender']] = df[['category', 'purchase type ', 'gender']].apply(encoder.fit_transform)
df.head()

#Binning ages into child, teenager, adult, middle aged, old aged
df['age group'] = pd.cut(x = df['age'], bins = [0, 12, 19, 30, 50, 100], labels = ["child", "teenager", "adult", "middle aged", "old aged"], include_lowest = True)
df.head()

#Applying MinMax scaler on spending scores (reduces values between 0-1)
scaler = MinMaxScaler()
df[['spending score', 'items purchased (monthly)', 'annual income (lakhs)']] = scaler.fit_transform(df[['spending score', 'items purchased (monthly)', 'annual income (lakhs)']])
df.head()

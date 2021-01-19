#Grouping people based on gender, age, category

#Hierarchical Clustering (Agglomerative clustering)

from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv('CustomerData.csv')
dataframe = df[['gender', 'age', 'category']]

agg = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean')
agg.fit_predict(dataframe)

dataframe['cluster'] = cluster.labels_

dataframe.head()

sns.scatterplot(data = dataframe, x = dataframe.cluster, y = dataframe.age)

sns.scatterplot(data = dataframe, x = dataframe.cluster, y = dataframe.gender.replace({0 : "Male", 1 : "Female"}))

sns.scatterplot(data = dataframe, x = dataframe.cluster, y = dataframe.category)

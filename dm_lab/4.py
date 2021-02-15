#FP Growth

#!pip install pyfpgrowth
import pyfpgrowth

import pandas as pd
df1 = pd.read_csv('association_analysis.csv')
df1.head()

df.drop(['tid'], axis = 1, inplace = True)

records = []
for i in range(len(df1)):
    record = []
    for j in range(len(df1.columns)):
        if df1.values[i, j]:
            record.append(df1.columns[j])
    records.append(record)

records[:3]

itemsets = pyfpgrowth.find_frequent_patterns(records, 0.03)
itemsets

pyfpgrowth.generate_association_rules(itemsets, 0.7)

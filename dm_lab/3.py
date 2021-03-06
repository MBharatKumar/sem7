df1 = pd.read_csv('association_analysis.csv')
df1.head()

df1.drop(['tid'], axis = 1, inplace = True)
# !pip install apyori
from apyori import apriori
#Converting dataframe to a list of lists containing items

records = []
for i in range(len(df1)):
    record = []
    for j in range(len(df1.columns)):
        if df1.values[i, j]:
            record.append(df1.columns[j])
    records.append(record)

records[:3]

min_sup = 0.03
min_confidence = 0.7

#Apriori
rules = apriori(records, min_support = min_sup, min_confidence = min_confidence)
rules = list(rules)
rules[0]

for rule in rules:
    items = [i for i in rule[0]]
    print("Rule : ",items, "Support :", rule[1], "Confidence : ", rule[2][0][2])

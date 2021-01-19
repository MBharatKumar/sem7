#FP Growth

#!pip install pyfpgrowth
import pyfpgrowth

itemsets = pyfpgrowth.find_frequent_patterns(records, 0.03)
itemsets

pyfpgrowth.generate_association_rules(itemsets, 0.7)

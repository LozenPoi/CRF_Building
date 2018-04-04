import numpy as np
from collections import Counter
import editdistance
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer as CV
import re

# Vectorize a set of string by n-grams.
def string_vectorize(input_list):
    vc = CV(analyzer='char_wb', ngram_range=(3, 4), min_df=1, token_pattern='[a-z]{2,}')
    name = []
    for i in input_list:
        s = re.findall('(?i)[a-z]{2,}', i)
        name.append(' '.join(s))
    vc.fit(name)
    vec = vc.transform(name).toarray()
    # print(name)
    # print(vec)
    dictionary = vc.get_feature_names()
    return vec, dictionary


my_str = ['aaaaa', 'bbbbb']
vec, dict = string_vectorize(my_str)
print(vec[:1])
print(dict)

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



# test_prob_list = []
# pattern_list = []
# for i in range(len_test):
#     # Calculate the marginal probabilities on the testing set using the current CRF.
#     y_sequence = crf.tagger_.tag(X_test[i])
#     margin_list = []
#     for j in range(len_ptname):
#         margin_list.append(crf.tagger_.marginal(y_sequence[j], j))
#     test_prob_list.append(margin_list)
#     # Calculate gradient on the marginal probabilities, and find the sharp changes and corresponding patterns.
#     grad = np.convolve(margin_list, [-1, 1], 'valid')
#     sharp_idx_down, = np.where(grad > 0.3)
#     sharp_idx_up, = np.where(grad < -0.3)
#     sharp_idx_down = (sharp_idx_down + 1).tolist()
#     sharp_idx_up = sharp_idx_up.tolist()
#     for j in range(len(sharp_idx_down)):
#         if(j<len(sharp_idx_up)):
#             pattern_list.append(test_string[i][sharp_idx_down[j]:sharp_idx_up[j]+1])
#         else:
#             pattern_list.append(test_string[i][sharp_idx_down[j]:])
#             break
#
# # Count the most frequent patterns.
# pattern_map = Counter(pattern_list)
# query_pattern = list(pattern_map.keys())[-1]
# #query_freq = list(pattern_map.values())[-1]
# #print(query_pattern, query_freq)

# test_prob_list = []
# pattern_list = []
# for i in range(len_test):
#     # Calculate the marginal probabilities on the testing set using the current CRF.
#     y_sequence = crf.tagger_.tag(X_test[i])
#     margin_list = [crf.tagger_.marginal(y_sequence[j], j) for j in range(len_ptname)]
#     test_prob_list.append(margin_list)
#     print(margin_list)

a = {1: [3,1]}
a[1].append(4)
print(a[1])





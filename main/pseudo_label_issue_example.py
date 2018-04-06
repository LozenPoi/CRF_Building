import os
import pickle
import numpy as np
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import multiprocessing
from collections import Counter
import editdistance
from sklearn.metrics.pairwise import cosine_similarity

import utils.utils as utils


# Define the feature dictionary.
def word2features(sent, i):
    word = sent[i][0]
    # Number of cumulative digits.
    cum_dig = 0
    for k in range(i+1):
        if sent[k][0].isdigit():
            cum_dig = cum_dig + 1
        else:
            cum_dig = 0
    features = {
        'word': word,
        'word.isdigit()': word.isdigit(),
        'first_digit': cum_dig == 1,
        'second_digit': cum_dig == 2,
        'third_digit': cum_dig == 3,
    }
    # for previous character
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:isdigit()': word1.isdigit(),
        })
    else:
        features['BOS'] = True
    # for next character
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:isdigit()': word1.isdigit(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]


with open("../dataset/filtered_dataset.bin", "rb") as my_dataset:
    dataset = pickle.load(my_dataset)
with open("../dataset/filtered_string.bin", "rb") as my_string:
    strings = pickle.load(my_string)

# Use "SODA4R731__ASO" to train a CRF.
X_train_current = [sent2features(dataset[1])]
Y_train_current = [sent2labels(dataset[1])]
crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
crf.fit(X_train_current, Y_train_current)

# Calculate entropy for each character of "SODA4337A_RVAV".
crf.tagger_.set(sent2features(dataset[0]))
entropy_seq = []
label_list = crf.tagger_.labels()
for j in range(14):
    marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
    entropy_seq.append(scipy.stats.entropy(marginal_prob))
print(strings[0])
y_predict = crf.tagger_.tag(sent2features(dataset[0]))
y_truth = sent2labels(dataset[0])
print('prediction: ', y_predict)
print('truth: ', y_truth)
print('entropy: ', entropy_seq)

mean_entropy = np.mean(entropy_seq)
std_entropy = np.std(entropy_seq)
z_score = [(entropy_seq[i]-mean_entropy)/std_entropy for i in range(14)]
print('z_score: ', z_score)

# Only manual label the part larger than 0.5.
y_label = y_predict[:]
for i in range(14):
    if z_score[i] > 0.5:
        y_label[i] = y_truth[i]
print('partial labeled: ', y_label)

# Train the crf again.
X_train_current.append(sent2features(dataset[0]))
Y_train_current.append(y_label)
crf.fit(X_train_current, Y_train_current)

# Try to predict "SODA3R683_RVAV".
prediction = crf.predict([sent2features(dataset[7])])
print('SODA3R683_RVAV prediction: ', prediction)

crf.tagger_.set(sent2features(dataset[7]))
entropy_seq = []
label_list = crf.tagger_.labels()
for j in range(14):
    marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
    entropy_seq.append(scipy.stats.entropy(marginal_prob))
print('entropy: ', entropy_seq)

mean_entropy = np.mean(entropy_seq)
std_entropy = np.std(entropy_seq)
z_score = [(entropy_seq[i]-mean_entropy)/std_entropy for i in range(14)]
print('z_score: ', z_score)




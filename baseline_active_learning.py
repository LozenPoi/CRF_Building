import editdistance
import pickle

import utils

from sklearn.cluster import KMeans

with open("filtered_dataset.bin", "rb") as my_dataset:
    dataset = pickle.load(my_dataset)
with open("filtered_string.bin", "rb") as my_string:
    string = pickle.load(my_string)

#print(len(dataset))
#print(dataset[1335-1])
print(len(string))
print(string[1])

train_set = dataset[:1235]
test_set = dataset[1235:]

# Define feature dictionary.
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

    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:isdigit()': word1.isdigit(),
        })
    else:
        features['BOS'] = True

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

X_train = [sent2features(s) for s in train_set]
y_train = [sent2labels(s) for s in train_set]
X_test = [sent2features(s) for s in test_set]
y_test = [sent2labels(s) for s in test_set]

# Calculate the average Edit Distance from each training string to the test set.

#print(editdistance.eval('banana', 'cccccccccccccccccc'))

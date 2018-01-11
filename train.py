#import matplotlib.pyplot as plt
#plt.style.use('ggplot')

from itertools import chain

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import pickle

filepath = 'all_soda_labelledManually.txt'

# Load converted data.
with open("vectorized.bin", "rb") as vector_file:
    vector = pickle.load(vector_file)
with open("pos.bin", "rb") as pos_file:
    pos = pickle.load(pos_file)
with open("dict.bin", "rb") as dict_file:
    dict = pickle.load(dict_file)

# Convert the training and testing data with the format fitting CRFsuite.
#print(len(dict))
#print(vector.shape)
#print(pos.shape)
with open(filepath, 'r') as fp:
    count = 0
    line = 'initial'
    dataset = []
    while line:
        line = fp.readline()
        # The point name is ended by the first space.
        idx = 0
        sent = []
        for i in line:
            if i == ' ':
                break
            else:
                if pos[count,idx] == 0:
                    sent.append((i, 'none', dict[vector[count,idx].astype(int)]))
                elif pos[count,idx] == 1:
                    sent.append((i, 'name', dict[vector[count,idx].astype(int)]))
                else:
                    sent.append((i, 'id', dict[vector[count,idx].astype(int)]))
                idx = idx + 1
        dataset.append(sent)
        count = count + 1
        #print("Line {}: {}".format(count, sent))
    dataset = dataset[:-1]
    #print(len(dataset))
fp.close()

# Define feature dictionary.
def word2features(sent, i):
    word = sent[i][0]
    label = sent[i][2]

    features = {
        'word': word,
        'word.isdigit()': word.isdigit(),
        #'label': label,
    }
    if i > 0:
        word1 = sent[i-1][0]
        label1 = sent[i-1][2]
        features.update({
            '-1:word': word1,
            '-1:isdigit()': word1.isdigit()
            #'-1:label': label1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        label1 = sent[i+1][2]
        features.update({
            '+1:word': word1,
            '+1:isdigit()': word1.isdigit()
            #'+1:label1': label1,
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

# Train a CRF.
train_set = dataset[:1000]
test_set = dataset[1000:]
#print(sent2features(train_set[0])[13])
X_train = [sent2features(s) for s in train_set]
y_train = [sent2labels(s) for s in train_set]
X_test = [sent2features(s) for s in test_set]
y_test = [sent2labels(s) for s in test_set]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Some evaluations.
labels = list(crf.classes_)
#print(labels)
y_pred = crf.predict(X_test)
#print(len(y_pred))
#print(len(y_test))
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3))

# This is the metric for measuring the sequential labeling accuracy on phrase level.
# Author: Zheng Luo.
# Date: 01/11/2018.

def phrase_acc(y_test, y_pred):
    # y_test is the ground-truth label, y_pred is the predicted label.
    if len(y_test) == len(y_pred):
        len_label = len(y_test)
    else:
        raise ValueError('Prediction has different length than groung truth.')
    for i in range(len_label):
        # Compare labels for each sentence/string.
        j = 0
        for label in y_test[i]:


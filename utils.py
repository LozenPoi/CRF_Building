# This is the metric for measuring the sequential labeling accuracy on phrase level.
# Author: Zheng Luo.
# Date: 01/11/2018.

def phrase_acc(y_test, y_pred):
    # y_test is the ground-truth label, y_pred is the predicted label.
    if len(y_test) == len(y_pred):
        len_label = len(y_test)
    else:
        raise ValueError('Prediction has different length than ground truth.')

    # Consider the accuracy in phrase level.
    len_str = len(y_test[0])
    phrase_count = 0
    phrase_correct = 0
    out_count = 0
    out_correct = 0
    for i in range(len_label):
        # Segment each phrase in ground truth label.
        correct_flag = False
        for j in range(len_str):
            if y_test[i][j][0] == 'b':
                phrase_count = phrase_count + 1
                if correct_flag:
                    phrase_correct = phrase_correct + 1
                if y_test[i][j] == y_pred[i][j]:
                    correct_flag = True
                else:
                    correct_flag = False
            elif y_test[i][j][0] == 'i':
                if y_test[i][j] != y_pred[i][j]:
                    correct_flag = False
            elif y_test[i][j][0] == 'o':
                out_count = out_count + 1
                if correct_flag:
                    phrase_correct = phrase_correct + 1
                    correct_flag = False
                if y_test[i][j] == y_pred[i][j]:
                    out_correct = out_correct + 1
        if correct_flag:
            phrase_correct = phrase_correct + 1

    return phrase_count, phrase_correct, out_count, out_correct

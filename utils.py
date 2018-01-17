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
    phrase_count = 0    # Total number of phrases.
    phrase_correct = 0  # Number of correctly classified phrases.
    out_count = 0   # Total number of out-of-phrases (characters labeled as "o").
    out_correct = 0 # Number of correctly classified out-of-phrases (characters labeled as "o").

    for i in range(len_label):
        # Compare ground-truth labels and predicted labels character by character.
        # Once a mismatch found within a phrase or the phrase has been counted as correctly classified, the flag will be
        # changed to "False".
        correct_flag = False

        for j in range(len_str):

            # If the character is a beginning-of-phrase.
            if y_test[i][j][0] == 'b':
                phrase_count = phrase_count + 1
                if y_test[i][j] == y_pred[i][j]:
                    if correct_flag:
                        phrase_correct = phrase_correct + 1
                    correct_flag = True
                else:
                    if correct_flag:
                        if y_pred[i][j][2:] != y_pred[i][j-1][2:]:  # special case
                            phrase_correct = phrase_correct + 1
                    correct_flag = False

            # If the character is an inside-of-phrase.
            elif y_test[i][j][0] == 'i':
                if y_test[i][j] != y_pred[i][j]:
                    correct_flag = False

            # If the character is an out-of-phrase.
            elif y_test[i][j][0] == 'o':
                out_count = out_count + 1
                if y_test[i][j] == y_pred[i][j]:
                    out_correct = out_correct + 1
                    if correct_flag:
                        phrase_correct = phrase_correct + 1
                        correct_flag = False
                else:
                    if correct_flag:
                        if y_pred[i][j][2:] != y_pred[i][j-1][2:]:  # special case
                            phrase_correct = phrase_correct + 1
                        correct_flag = False

        # For the case where the phrase is at the end of a string.
        if correct_flag:
            phrase_correct = phrase_correct + 1

    return phrase_count, phrase_correct, out_count, out_correct

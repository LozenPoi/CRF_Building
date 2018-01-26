import pickle
import numpy as np
import editdistance
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import RepeatedKFold
import threading
from joblib import Parallel, delayed
import multiprocessing

import utils

# Calculate average Edit distance from each element of new_sample_set to current_set.
def avr_edit_distance(current_set, new_sample_set):
    len_current = len(current_set)
    len_new = len(new_sample_set)
    distance = np.zeros(len_new)
    for k in range(len_new):
        for j in range(len_current):
            distance[k] = distance[k] + editdistance.eval(new_sample_set[k], current_set[j])
        distance[k] = distance[k]/len_current
    return distance

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

# Active learning using edit distance with cross validation.
def cv_edit_active_learn(train_idx, test_idx, dataset, strings, max_samples_batch, num_fold, batch_size, cv_round):

    phrase_acc = np.zeros([max_samples_batch, num_fold])
    out_acc = np.zeros([max_samples_batch, num_fold])

    # Define training set and testing set.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    #test_string = [strings[i] for i in test_idx]
    X_train = [sent2features(s) for s in train_set]
    y_train = [sent2labels(s) for s in train_set]
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]

    # Define an initial actual training set from the training pool.
    X_train_current = X_train[:2]
    y_train_current = y_train[:2]
    X_train_new = X_train[2:]
    y_train_new = y_train[2:]
    train_string_current = train_string[:2]
    train_string_new = train_string[2:]

    for num_training in range(max_samples_batch):

        # Calculate average distance from new sample candidates to current training set.
        distance = avr_edit_distance(train_string_current, train_string_new)
        sort_idx = np.argsort(-distance, kind='mergesort').tolist()
        # update strings
        string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
        for i in string_to_remove:
            train_string_current.append(i)
            train_string_new.remove(i)
        # update training features
        feature_to_remove = [X_train_new[i] for i in sort_idx[:batch_size]]
        for i in feature_to_remove:
            X_train_current.append(i)
            X_train_new.remove(i)
        # update training labels
        label_to_remove = [y_train_new[i] for i in sort_idx[:batch_size]]
        for i in label_to_remove:
            y_train_current.append(i)
            y_train_new.remove(i)

        # # define fixed parameters and parameters to search
        # crf = sklearn_crfsuite.CRF(
        #     algorithm='lbfgs',
        #     max_iterations=100,
        #     all_possible_transitions=True
        # )
        # params_space = {
        #     'c1': scipy.stats.expon(scale=0.5),
        #     'c2': scipy.stats.expon(scale=0.05),
        # }
        #
        # # search
        # rs = RandomizedSearchCV(crf, params_space,
        #                         cv=2,
        #                         verbose=1,
        #                         n_jobs=-1,
        #                         n_iter=5)
        # rs.fit(X_train_current, y_train_current)
        #
        # print('best params:', rs.best_params_)
        # print('best CV score:', rs.best_score_)
        # crf = rs.best_estimator_

        # Train the CRF.
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train_current, y_train_current)

        # Use the estimator.
        y_pred = crf.predict(X_test)
        phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
        print(phrase_count, phrase_correct, out_count, out_correct)
        phrase_acc[num_training, cv_round] = phrase_correct / phrase_count
        out_acc[num_training, cv_round] = out_correct / out_count

    return phrase_acc, out_acc

# This is the main function.
if __name__ == '__main__':

    with open("filtered_dataset.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("filtered_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)

    # Randomly select test set and training pool in the way of cross validation.
    num_fold = 10
    kf = RepeatedKFold(n_splits=num_fold, n_repeats=1, random_state=666)

    # Define a loop for plotting figures.
    max_samples_batch = 50
    batch_size = 1

    num_cores = multiprocessing.cpu_count()
    phrase_acc, out_acc = Parallel(n_jobs=num_cores)(delayed(cv_edit_active_learn)(i) for i in inputs)

    # cv_round = 0
    # for train_idx, test_idx in kf.split(dataset):

        # # Define training set and testing set.
        # train_set = [dataset[i] for i in train_idx]
        # test_set = [dataset[i] for i in test_idx]
        # train_string = [strings[i] for i in train_idx]
        # #test_string = [strings[i] for i in test_idx]
        # X_train = [sent2features(s) for s in train_set]
        # y_train = [sent2labels(s) for s in train_set]
        # X_test = [sent2features(s) for s in test_set]
        # y_test = [sent2labels(s) for s in test_set]
        #
        # # Define an initial actual training set from the training pool.
        # X_train_current = X_train[:2]
        # y_train_current = y_train[:2]
        # X_train_new = X_train[2:]
        # y_train_new = y_train[2:]
        # train_string_current = train_string[:2]
        # train_string_new = train_string[2:]
        #
        # for num_training in range(max_samples_batch):
        #
        #     # Calculate average distance from new sample candidates to current training set.
        #     distance = avr_edit_distance(train_string_current, train_string_new)
        #     sort_idx = np.argsort(-distance, kind='mergesort').tolist()
        #     # update strings
        #     string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
        #     for i in string_to_remove:
        #         train_string_current.append(i)
        #         train_string_new.remove(i)
        #     # update training features
        #     feature_to_remove = [X_train_new[i] for i in sort_idx[:batch_size]]
        #     for i in feature_to_remove:
        #         X_train_current.append(i)
        #         X_train_new.remove(i)
        #     # update training labels
        #     label_to_remove = [y_train_new[i] for i in sort_idx[:batch_size]]
        #     for i in label_to_remove:
        #         y_train_current.append(i)
        #         y_train_new.remove(i)
        #
        #     # # define fixed parameters and parameters to search
        #     # crf = sklearn_crfsuite.CRF(
        #     #     algorithm='lbfgs',
        #     #     max_iterations=100,
        #     #     all_possible_transitions=True
        #     # )
        #     # params_space = {
        #     #     'c1': scipy.stats.expon(scale=0.5),
        #     #     'c2': scipy.stats.expon(scale=0.05),
        #     # }
        #     #
        #     # # search
        #     # rs = RandomizedSearchCV(crf, params_space,
        #     #                         cv=2,
        #     #                         verbose=1,
        #     #                         n_jobs=-1,
        #     #                         n_iter=5)
        #     # rs.fit(X_train_current, y_train_current)
        #     #
        #     # print('best params:', rs.best_params_)
        #     # print('best CV score:', rs.best_score_)
        #     # crf = rs.best_estimator_
        #
        #     # Train the CRF.
        #     crf = sklearn_crfsuite.CRF(
        #         algorithm='lbfgs',
        #         c1=0.1,
        #         c2=0.1,
        #         max_iterations=100,
        #         all_possible_transitions=True
        #     )
        #     crf.fit(X_train_current, y_train_current)
        #
        #     # Use the estimator.
        #     y_pred = crf.predict(X_test)
        #     phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
        #     print(phrase_count, phrase_correct, out_count, out_correct)
        #     phrase_acc[num_training,cv_round] = phrase_correct/phrase_count
        #     out_acc[num_training,cv_round] = out_correct/out_count

        # cv_round += 1

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    out_acc_av = np.sum(out_acc, axis=0)/num_fold
    plt.plot(np.arange(3,max_samples_batch*batch_size+3,batch_size),phrase_acc,'r',
             np.arange(3,max_samples_batch*batch_size+3,batch_size),out_acc,'b')
    plt.xlabel('number of training samples')
    plt.ylabel('testing accuracy')
    plt.legend(['phrase accuracy', 'out-of-phrase accuracy'])
    plt.show()

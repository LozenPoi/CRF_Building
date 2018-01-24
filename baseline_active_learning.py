import pickle
import numpy as np
import editdistance
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import random

import utils

#from sklearn.cluster import KMeans

if __name__ == '__main__':

    with open("filtered_dataset.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("filtered_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)

    # print(len(dataset))
    # print(dataset[1])
    # print(len(strings))
    # print(strings[1:3])

    # train_set = dataset[:1235]
    # test_set = dataset[1235:]
    # train_string = strings[:1235]
    # test_string = strings[1235:]

    # Randomly select 100 samples to be the testing set.
    total_len = len(dataset)
    select_idx = []
    train_set = []
    test_set = []
    train_string = []
    test_string = []
    testing_size = 100
    random.seed(666)
    for i in range(testing_size):
        select_idx.append(random.randint(0,total_len))
    for i in range(total_len):
        if i in select_idx:
            test_set.append(dataset[i])
            test_string.append(strings[i])
        else:
            train_set.append(dataset[i])
            train_string.append(strings[i])
    # print(train_set[1:3])
    # print(train_string[1:3])

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
    len_train = len(train_string)
    len_test = len(test_string)
    train_distance = np.zeros(len_train)
    for i in range(len_train):
        for j in range(len_test):
            train_distance[i] = train_distance[i] + editdistance.eval(train_string[i],test_string[j])
        train_distance[i] = train_distance[i]/len_test

    sort_idx = np.argsort(train_distance, kind='mergesort').tolist()

    # Define a loop for plotting figures.
    max_samples = 200
    phrase_acc = np.zeros(max_samples)
    out_acc = np.zeros(max_samples)
    for num_training in range(3,max_samples+3):

        # Train a CRF using strings with the top shortest distance.
        X_train_tmp = [X_train[i] for i in sort_idx[:num_training]]
        y_train_tmp = [y_train[i] for i in sort_idx[:num_training]]

        # define fixed parameters and parameters to search
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        # search
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=30)
        rs.fit(X_train_tmp, y_train_tmp)

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)

        # # Train the CRF.
        # crf = sklearn_crfsuite.CRF(
        #     algorithm='lbfgs',
        #     c1=0.1,
        #     c2=0.1,
        #     max_iterations=100,
        #     all_possible_transitions=True
        # )
        # crf.fit(X_train_tmp, y_train_tmp)
        #
        # # Performance evaluation.
        # y_pred = crf.predict(X_test)
        # phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
        # print(phrase_count, phrase_correct, out_count, out_correct)

        # Use the best estimator.
        crf = rs.best_estimator_
        y_pred = crf.predict(X_test)
        phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
        print(phrase_count, phrase_correct, out_count, out_correct)
        phrase_acc[num_training-3] = phrase_correct/phrase_count
        out_acc[num_training-3] = out_correct/out_count

    plt.plot(np.arange(3,max_samples+3),phrase_acc,'ro',np.arange(3,max_samples+3),out_acc,'bs')
    plt.xlabel('number of training samples')
    plt.ylabel('testing accuracy')
    plt.legend(['phrase accuracy', 'out-of-phrase accuracy'])
    plt.show()

import os
import pickle
import numpy as np
import editdistance
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import RepeatedKFold
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
def cv_edit_active_learn(args):

    # Read the input args.
    train_idx = args['train_idx']
    test_idx = args['test_idx']
    dataset = args['dataset']
    strings = args['strings']
    max_samples_batch = args['max_samples_batch']
    batch_size = args['batch_size']

    phrase_acc = np.zeros([max_samples_batch])
    out_acc = np.zeros([max_samples_batch])

    # Define training set and testing set.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set from the training pool.
    train_set_current = train_set[:2]
    train_set_new = train_set[2:]
    train_string_current = train_string[:2]
    train_string_new = train_string[2:]

    # tmp_record_1 = 0
    # tmp_record_2 = 1
    # for i in range(len(train_string)):
    #     if train_string[i] == 'SODA4337A_RVAV':
    #         tmp_record_1 = i
    #         break
    # for i in range(len(train_string)):
    #     if train_string[i] == 'SOD__BLD_CURTL':
    #         tmp_record_2 = i
    #         break
    #
    # train_set_current = [train_set[tmp_record_1]]
    # train_set_current.append(train_set[tmp_record_2])
    # train_set_new = train_set
    # train_set_new.remove(train_set[tmp_record_1])
    # train_set_new.remove(train_set[tmp_record_2])
    #
    # train_string_current = [train_string[tmp_record_1]]
    # train_string_current.append(train_string[tmp_record_2])
    # train_string_new = train_string
    # train_string_new.remove(train_string[tmp_record_1])
    # train_string_new.remove(train_string[tmp_record_2])

    # Obtain testing features and labels.
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]

    string_observe = []
    len_test = len(test_string)
    indicator = 0
    for num_training in range(max_samples_batch):

        # # Calculate average distance from new sample candidates to current training set.
        # distance = avr_edit_distance(train_string_current, train_string_new)
        # sort_idx = np.argsort(-distance, kind='mergesort').tolist()
        # # update training strings
        # string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
        # for i in string_to_remove:
        #     train_string_current.append(i)
        #     train_string_new.remove(i)
        # # update training set
        # sample_to_remove = [train_set_new[i] for i in sort_idx[:batch_size]]
        # for i in sample_to_remove:
        #     train_set_current.append(i)
        #     train_set_new.remove(i)
        # # To see what the model learns from 90 samples to 100 samples.
        # if (num_training >= 0) & (num_training <= 150):
        #     string_observe.extend(string_to_remove)


        # # Calculate average distance from new sample to the test set.
        # distance = avr_edit_distance(test_string, train_string_new)
        # sort_idx = np.argsort(distance, kind='mergesort').tolist()
        # # Add new samples from training pool to training set.
        # for i in range(batch_size):
        #     j = sort_idx[num_training*batch_size+i]
        #     train_set_current.append(train_set_new[j])
        #     string_observe.append(train_string_new[j])


        # Calculate the distance to a single sample of test set.
        distance = avr_edit_distance(test_string[indicator], train_string_new)
        sort_idx = np.argsort(distance, kind='mergesort').tolist()
        # update training strings
        string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
        for i in string_to_remove:
            train_string_current.append(i)
            train_string_new.remove(i)
        # update training set
        sample_to_remove = [train_set_new[i] for i in sort_idx[:batch_size]]
        for i in sample_to_remove:
            train_set_current.append(i)
            train_set_new.remove(i)
        # To see what the model learns from 90 samples to 100 samples.
        if (num_training >= 0) & (num_training <= 150):
            string_observe.extend(string_to_remove)
        if indicator < len_test-1:
            indicator += 1
        else:
            indicator = 0


        # Obtain current training features.
        X_train_current = [sent2features(s) for s in train_set_current]
        y_train_current = [sent2labels(s) for s in train_set_current]

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
        phrase_acc[num_training] = phrase_correct / phrase_count
        out_acc[num_training] = out_correct / out_count

    return phrase_acc, out_acc, string_observe

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
    max_samples_batch = 200
    batch_size = 1

    pool = multiprocessing.Pool(os.cpu_count()-1)
    args = []
    # print(os.cpu_count()) # It counts for logical processors instead of physical cores.
    for train_idx, test_idx in kf.split(dataset):
        tmp_args = {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'dataset': dataset,
            'strings': strings,
            'max_samples_batch': max_samples_batch,
            'batch_size': batch_size,
        }
        args.append(tmp_args)
    results = pool.map(cv_edit_active_learn, args)
    # print(len(results))
    # print(len(results[0]))
    phrase_acc = [results[i][0] for i in range(num_fold)]
    out_acc = [results[i][1] for i in range(num_fold)]
    # print(len(phrase_acc))
    # print(len(phrase_acc[0]))

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    out_acc_av = np.sum(out_acc, axis=0)/num_fold
    plt.plot(np.arange(3,max_samples_batch*batch_size+3,batch_size),phrase_acc_av,'r',
             np.arange(3,max_samples_batch*batch_size+3,batch_size),out_acc_av,'b')
    plt.xlabel('number of training samples')
    plt.ylabel('testing accuracy')
    plt.legend(['phrase accuracy', 'out-of-phrase accuracy'])
    plt.show()

    # Save data for future plotting.
    with open("phrase_acc_edit.bin", "wb") as phrase_edit_file:
        pickle.dump(phrase_acc, phrase_edit_file)
    with open("out_acc_edit.bin", "wb") as out_edit_file:
        pickle.dump(out_acc, out_edit_file)

    # Observed strings.
    print([results[i][2] for i in range(num_fold)])

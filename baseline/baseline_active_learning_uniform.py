import os
import pickle
import numpy as np
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import RepeatedKFold
import multiprocessing
import pickle

import utils.utils as utils

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

# Active learning using uniform sampling with cross validation.
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
    #test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set from the training pool.
    train_set_current = train_set[:2]
    train_set_new = train_set[2:]
    train_string_current = train_string[:2]
    train_string_new = train_string[2:]

    # Obtain testing features and labels.
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]

    for num_training in range(max_samples_batch):

        # Uniformly take samples from the training pool.
        len_train_new = len(train_set_current)
        sample_idx = []
        for i in range(batch_size):
            rand_tmp = random.randint(0,len_train_new-1)
            while rand_tmp in sample_idx:
                rand_tmp = random.randint(0,len_train_new-1)
            sample_idx.append(rand_tmp)

        # update training strings
        string_to_remove = [train_string_new[i] for i in sample_idx[:batch_size]]
        for i in string_to_remove:
            train_string_current.append(i)
            train_string_new.remove(i)
        # update training set
        sample_to_remove = [train_set_new[i] for i in sample_idx[:batch_size]]
        for i in sample_to_remove:
            train_set_current.append(i)
            train_set_new.remove(i)

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
        # print(phrase_count, phrase_correct, out_count, out_correct)
        phrase_acc[num_training] = phrase_correct / phrase_count
        out_acc[num_training] = out_correct / out_count

    return phrase_acc, out_acc

# This is the main function.
if __name__ == '__main__':

    with open("../dataset/filtered_dataset.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("../dataset/filtered_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)

    # Randomly select test set and training pool in the way of cross validation.
    num_fold = 8
    kf = RepeatedKFold(n_splits=num_fold, n_repeats=1, random_state=666)

    # Define a loop for plotting figures.
    max_samples_batch = 100
    batch_size = 1

    pool = multiprocessing.Pool(os.cpu_count())
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

    # Use the same parameters for more than once of uniformly random sampling.
    results = pool.map(cv_edit_active_learn, args)
    phrase_acc = np.array([results[i][0] for i in range(num_fold)])
    out_acc = np.array([results[i][1] for i in range(num_fold)])

    # # Run multiple sampling for each fold.
    # number_iter = 4
    # for i in range(number_iter):
    #     results = pool.map(cv_edit_active_learn, args)
    #     phrase_acc = phrase_acc + np.array([results[i][0] for i in range(num_fold)])
    #     out_acc = out_acc + np.array([results[i][1] for i in range(num_fold)])
    # phrase_acc = phrase_acc/(number_iter+1)
    # out_acc = out_acc/(number_iter+1)

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    out_acc_av = np.sum(out_acc, axis=0)/num_fold
    plt.plot(np.arange(3, max_samples_batch*batch_size+3, batch_size), phrase_acc_av, 'r',
             np.arange(3, max_samples_batch*batch_size+3, batch_size), out_acc_av, 'b')
    plt.xlabel('number of training samples')
    plt.ylabel('testing accuracy')
    plt.legend(['phrase accuracy', 'out-of-phrase accuracy'])
    plt.show()

    # Save data for future plotting.
    with open("phrase_acc_uniform.bin", "wb") as phrase_uniform_file:
        pickle.dump(phrase_acc, phrase_uniform_file)
    with open("out_acc_uniform.bin", "wb") as out_uniform_file:
        pickle.dump(out_acc, out_uniform_file)

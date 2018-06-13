import os
import pickle
import numpy as np
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import multiprocessing
from pycrfsuite import Tagger
import editdistance
import math
from scipy import spatial
import sys

import utils.utils as utils

# # Calculate average Edit distance from each element of new_sample_set to current_set.
# def avr_edit_distance(current_set, new_sample_set):
#     len_current = len(current_set)
#     len_new = len(new_sample_set)
#     distance = np.zeros(len_new)
#     for k in range(len_new):
#         for j in range(len_current):
#             distance[k] = distance[k] + editdistance.eval(new_sample_set[k], current_set[j])
#         distance[k] = distance[k]/len_current
#     return distance

# Define feature dictionary.
def word2features(sent, i):
    # obtain some overall information of the point name string
    num_part = 4
    len_string = len(sent)
    mod = len_string % num_part
    part_size = int(math.floor(len_string/num_part))
    # determine which part the current character belongs to
    # larger part will be at the beginning if the whole sequence can't be divided evenly
    size_list = []
    mod_count = 0
    for j in range(num_part):
        if mod_count < mod:
            size_list.append(part_size+1)
            mod_count += 1
        else:
            size_list.append(part_size)
    # for current character
    # part_indicator = []
    # for j in size_list:
    #     #
    word = sent[i][0]
    if word.isdigit():
        itself = 'NUM'
    else:
        itself = word
    features = {
        'word': itself,
    }
    # for previous character
    if i > 0:
        word1 = sent[i-1][0]
        if word1.isdigit():
            itself1 = 'NUM'
        else:
            itself1 = word1
        features.update({
            '-1:word': itself1,
        })
    else:
        features['BOS'] = True
    # for next character
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        if word1.isdigit():
            itself1 = 'NUM'
        else:
            itself1 = word1
        features.update({
            '+1:word': itself1,
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

    phrase_acc = np.zeros([max_samples_batch + 1])
    out_acc = np.zeros([max_samples_batch + 1])
    label_count = np.zeros([max_samples_batch + 1])

    # Define training set and testing set.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set from the training pool.
    initial_size = 2
    train_set_current = train_set[:initial_size]
    train_set_new = train_set[initial_size:]
    train_string_current = train_string[:initial_size]
    train_string_new = train_string[initial_size:]
    for i in range(initial_size):
        label_count[0] += len(train_string[i])

    # Obtain testing features and labels.
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]

    # Train a CRF using the current training set.
    X_train_current = [sent2features(s) for s in train_set_current]
    y_train_current = [sent2labels(s) for s in train_set_current]
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
    phrase_acc[0] = phrase_correct / phrase_count
    out_acc[0] = out_correct / out_count

    # Vectorized and clustered test set.
    total_string = test_string[:]
    total_string.extend(train_string_new)
    vec, _ = utils.string_vectorize(total_string)
    test_vec = vec[:len(test_string)].tolist()
    train_new_vec = vec[len(test_string):].tolist()

    # Pre-calculate similarity.
    # This will be efficient if the number of iterations is large.
    sim_matrix = np.zeros((len(train_new_vec), len(test_vec)))
    for i in range(len(train_new_vec)):
        for j in range(len(test_vec)):
            sim_matrix[i, j] = 1 - spatial.distance.cosine(train_new_vec[i], test_vec[j])

    len_test = len(test_set)

    for num_training in range(max_samples_batch):

        # Calculate the confidence on the testing set using the current CRF.
        prob_list = []
        for i in range(len_test):
            # crf.tagger_.set(X_train_new[i])
            y_sequence = crf.tagger_.tag(X_test[i])
            # print(crf.tagger_.probability(y_sequence))
            # normalized sequence probability
            prob_norm = math.exp(math.log(crf.tagger_.probability(y_sequence)) / len(test_string[i]))
            prob_list.append(prob_norm)

        # Sort the test set based on confidence.
        sort_idx_temp = np.argsort(np.array(prob_list), kind='mergesort').tolist()

        # Calculate the average similarity between the unlabeled samples and the selected test samples.
        # temp_set = [test_string[i] for i in sort_idx_temp[:5]]
        # distance = utils.avr_edit_distance(temp_set, train_string_new, True)
        group_size = 5
        avr_sim = np.sum(sim_matrix[:, sort_idx_temp[:group_size]], axis=1)/group_size
        distance = avr_sim

        # We want to have information weighted by such distance.
        X_train_new = [sent2features(s) for s in train_set_new]
        len_train_new = len(train_set_new)
        prob_list_candidate = []
        for i in range(len_train_new):
            y_sequence = crf.tagger_.tag(X_train_new[i])
            prob_norm = math.exp(math.log(crf.tagger_.probability(y_sequence)) / len(train_string_new[i]))
            prob_list_candidate.append(prob_norm)
        candidate_score = []
        for i in range(len_train_new):
            if distance[i] == 0:
                candidate_score.append(sys.float_info.max)
            else:
                candidate_score.append(prob_list_candidate[i] / distance[i])

        sort_idx = np.argsort(candidate_score, kind='mergesort').tolist()

        # if (num_training>=20)&(num_training<=40):
        #     print([train_string_new[i] for i in sort_idx[:batch_size]])

        # Assume the batch size is 1.
        label_count[num_training + 1] = label_count[num_training] + len(train_set_new[sort_idx[0]])

        # update training set
        # sample_to_remove = [train_set_new[i] for i in sort_idx[:batch_size]]
        # for i in sample_to_remove:
        #     train_set_current.append(i)
        #     train_set_new.remove(i)
        # string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
        # for i in string_to_remove:
        #     train_string_current.append(i)
        #     train_string_new.remove(i)
        idx_to_remove = sort_idx[:batch_size]
        idx_to_remove = np.sort(idx_to_remove, kind='mergesort').tolist()
        for i in range(batch_size):
            sim_matrix = np.delete(sim_matrix, idx_to_remove[-i-1], 0)
            train_set_current.append(train_set_new[idx_to_remove[-i-1]])
            del train_set_new[idx_to_remove[-i-1]]
            train_string_current.append(train_string_new[idx_to_remove[-i-1]])
            del train_string_new[idx_to_remove[-i-1]]

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
        phrase_acc[num_training+1] = phrase_correct / phrase_count
        out_acc[num_training+1] = out_correct / out_count

    return phrase_acc, out_acc, label_count

# This is the main function.
if __name__ == '__main__':

    # with open("../dataset/filtered_dataset.bin", "rb") as my_dataset:
    #     dataset = pickle.load(my_dataset)
    # with open("../dataset/filtered_string.bin", "rb") as my_string:
    #     strings = pickle.load(my_string)
    with open("../dataset/sdh_dataset.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("../dataset/sdh_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)

    # Randomly select test set and training pool in the way of cross validation.
    num_fold = 8
    kf = RepeatedKFold(n_splits=num_fold, n_repeats=1, random_state=666)

    # Define a loop for plotting figures.
    max_samples_batch = 50
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
    results = pool.map(cv_edit_active_learn, args)
    # print(len(results))
    # print(len(results[0]))
    phrase_acc = [results[i][0] for i in range(num_fold)]
    out_acc = [results[i][1] for i in range(num_fold)]
    label_count = [results[i][2] for i in range(num_fold)]
    # print(len(phrase_acc))
    # print(len(phrase_acc[0]))

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    out_acc_av = np.sum(out_acc, axis=0)/num_fold
    plt.plot(np.arange(2, max_samples_batch*batch_size+3, batch_size), phrase_acc_av, 'r',
             np.arange(2, max_samples_batch*batch_size+3, batch_size), out_acc_av, 'b')
    plt.xlabel('number of training samples')
    plt.ylabel('testing accuracy')
    plt.legend(['phrase accuracy', 'out-of-phrase accuracy'])
    plt.show()

    with open("sdh_phrase_acc_confidence_edit.bin", "wb") as phrase_confidence_file:
        pickle.dump(phrase_acc, phrase_confidence_file)
    with open("sdh_out_acc_confidence_edit.bin", "wb") as out_confidence_file:
        pickle.dump(out_acc, out_confidence_file)
    with open("sdh_confidence_edit_num.bin", "wb") as label_count_file:
        pickle.dump(label_count, label_count_file)

import os
import pickle
import numpy as np
import sklearn_crfsuite
import scipy.stats
import math
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import multiprocessing
from collections import Counter
import editdistance
import operator

import utils.utils as utils


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
    part_cumulative = [0]*num_part
    for j in range(num_part):
        if j > 0:
            part_cumulative[j] = part_cumulative[j-1] + size_list[j]
        else:
            part_cumulative[j] = size_list[j] - 1   # indices start from 0
    part_indicator = [0]*num_part
    for j in range(num_part):
        if part_cumulative[j] >= i:
            part_indicator[j] = 1
            break
    word = sent[i][0]
    if word.isdigit():
        itself = 'NUM'
    else:
        itself = word
    features = {
        'word': itself,
        'part0': part_indicator[0] == 1,
        'part1': part_indicator[1] == 1,
        'part2': part_indicator[2] == 1,
        'part3': part_indicator[3] == 1,
    }
    # for previous character
    if i > 0:
        part_indicator = [0] * num_part
        for j in range(num_part):
            if part_cumulative[j] >= i-1:
                part_indicator[j] = 1
                break
        word1 = sent[i-1][0]
        if word1.isdigit():
            itself1 = 'NUM'
        else:
            itself1 = word1
        features.update({
            '-1:word': itself1,
            '-1:part0': part_indicator[0] == 1,
            '-1:part1': part_indicator[1] == 1,
            '-1:part2': part_indicator[2] == 1,
            '-1:part3': part_indicator[3] == 1,
        })
    else:
        features['BOS'] = True
    # for next character
    if i < len(sent)-1:
        part_indicator = [0] * num_part
        for j in range(num_part):
            if part_cumulative[j] >= i + 1:
                part_indicator[j] = 1
                break
        word1 = sent[i+1][0]
        if word1.isdigit():
            itself1 = 'NUM'
        else:
            itself1 = word1
        features.update({
            '+1:word': itself1,
            '+1:part0': part_indicator[0] == 1,
            '+1:part1': part_indicator[1] == 1,
            '+1:part2': part_indicator[2] == 1,
            '+1:part3': part_indicator[3] == 1,
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

    # Initialize arrays to store results.
    phrase_acc = np.zeros([max_samples_batch+1])
    out_acc = np.zeros([max_samples_batch+1])
    label_count = np.zeros([max_samples_batch+1])
    pseudo_acc = np.zeros([max_samples_batch+1])

    # Define training set and testing set and corresponding original strings.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set and the training pool (unlabeled data).
    initial_size = 2
    train_set_current = train_set[:initial_size]
    train_set_new = train_set[initial_size:]
    train_string_current = train_string[:initial_size]
    train_string_new = train_string[initial_size:]
    count = 0
    for i in range(initial_size):
        count += len(train_string[i])

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
    label_count[0] = count
    pseudo_acc[0] = 1   # There is no pseudo-label at the beginning.

    # len_test = len(test_set)
    initial_budget = 100
    if count >= initial_budget:
        print('Error: initial budget is less than initial number of labels.')
    else:
        label_threshold = initial_budget

    for num_training in range(max_samples_batch):

        label_list = crf.tagger_.labels()
        # # Want to look at the confidence (average entropy for each character of each string) on unlabeled data.
        # entropy_list = []
        # for i in train_set_new:
        #     crf.tagger_.set(sent2features(i))
        #     entropy_seq = []
        #     len_ptname = len(i)
        #     for j in range(len_ptname):
        #         marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
        #         entropy_seq.append(scipy.stats.entropy(marginal_prob))
        #     entropy_list.append(entropy_seq)
        #
        # # Select the string with the largest entropy sum.
        # candidate_score = []
        # for i in range(len(entropy_list)):
        #     candidate_score.append(sum(entropy_list[i]))
        # sort_idx = np.argmax(candidate_score)

        # Calculate the confidence on the training pool (train_set_new) using the current CRF.
        X_train_new = [sent2features(s) for s in train_set_new]
        len_train_new = len(train_set_new)
        prob_list = []
        for i in range(len_train_new):
            y_sequence = crf.tagger_.tag(X_train_new[i])
            # normalized sequence probability
            prob_norm = math.exp(math.log(crf.tagger_.probability(y_sequence)) / len(train_string_new[i]))
            prob_list.append(prob_norm)

        # Sort the training pool based on confidence.
        sort_idx = np.argsort(np.array(prob_list), kind='mergesort').tolist()
        sort_idx = sort_idx[0]

        # Exhausted search through all substrings.
        # Search substrings with length 2 to len_ptname.
        y_sequence = crf.tagger_.tag(sent2features(train_set_new[sort_idx]))  # generate pseudo-label firstly
        candidate_entropy_list = []
        len_ptname = len(train_set_new[sort_idx])
        for j in range(len_ptname):
            marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
            candidate_entropy_list.append(scipy.stats.entropy(marginal_prob))
            # sorted_marginal_prob = np.sort(marginal_prob, kind='mergesort').tolist()
            # sorted_marginal_prob.reverse()
            # candidate_entropy_list.append(sorted_marginal_prob[0]-sorted_marginal_prob[1])
        substring_score = {}
        for i in range(len_ptname - 4):
            for j in range(i + 5, len_ptname):  # should be len_ptname+1 if want to include full string
                selected_entropy = sum(candidate_entropy_list[i:j]) / (j - i)
                rest_entropy = (sum(candidate_entropy_list) - sum(candidate_entropy_list[i:j])) / (len_ptname - (j - i))
                substring_score[(i, j)] = selected_entropy - rest_entropy

        # Rank the substrings based on their scores in descending order.
        sorted_substring_score = sorted(substring_score.items(), key=operator.itemgetter(1))
        sorted_substring_score.reverse()
        index_tuple = sorted_substring_score[0][0]
        label_index = []
        for i in range(index_tuple[0], index_tuple[1]):
            label_index.append(i)

        # Apply pseudo-labeling.
        y_sequence_truth = sent2labels(train_set_new[sort_idx])
        pseudo_label_total = 0
        pseudo_label_correct = 0
        for i in label_index:
            count += 1
            if y_sequence[i] == sent2labels(train_set_new[sort_idx])[i]:
                pseudo_label_correct += 1
            y_sequence[i] = sent2labels(train_set_new[sort_idx])[i]
            pseudo_label_total += 1
        label_count[num_training + 1] = count
        if pseudo_label_total != 0:
            pseudo_acc[num_training + 1] = pseudo_label_correct / pseudo_label_total
        else:
            pseudo_acc[num_training + 1] = 1

        # Update training set.
        train_set_current.append(train_set_new[sort_idx])
        train_string_current.append(train_string_new[sort_idx])
        X_train_current.append(sent2features(train_set_new[sort_idx]))
        y_train_current.append(y_sequence)
        del train_set_new[sort_idx]
        del train_string_new[sort_idx]

        # Train the CRF.
        # crf = sklearn_crfsuite.CRF(
        #     algorithm='lbfgs',
        #     c1=0.1,
        #     c2=0.1,
        #     max_iterations=100,
        #     all_possible_transitions=True
        # )
        crf.fit(X_train_current, y_train_current)

        # Use the estimator.
        y_pred = crf.predict(X_test)
        phrase_count, phrase_correct, out_count, out_correct = utils.phrase_acc(y_test, y_pred)
        # print(phrase_count, phrase_correct, out_count, out_correct)
        phrase_acc[num_training+1] = phrase_correct / phrase_count
        out_acc[num_training+1] = out_correct / out_count

    return phrase_acc, out_acc, label_count, pseudo_acc

# This is the main function.
if __name__ == '__main__':

    with open("../dataset/filtered_dataset.bin", "rb") as my_dataset:
        dataset = pickle.load(my_dataset)
    with open("../dataset/filtered_string.bin", "rb") as my_string:
        strings = pickle.load(my_string)
    # with open("../dataset/ibm_dataset.bin", "rb") as my_dataset:
    #     dataset = pickle.load(my_dataset)
    # with open("../dataset/ibm_string.bin", "rb") as my_string:
    #     strings = pickle.load(my_string)

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
    results = pool.map(cv_edit_active_learn, args)
    # print(len(results))
    # print(len(results[0]))
    phrase_acc = [results[i][0] for i in range(num_fold)]
    out_acc = [results[i][1] for i in range(num_fold)]
    # print(len(phrase_acc))
    # print(len(phrase_acc[0]))
    label_count = [results[i][2] for i in range(num_fold)]
    pseudo_acc = [results[i][3] for i in range(num_fold)]

    with open("../baseline/ibm_phrase_acc_confidence_edit.bin", "rb") as phrase_confidence_file_temp:
        phrase_acc_confidence_edit = pickle.load(phrase_confidence_file_temp)
    with open("../baseline/ibm_out_acc_confidence_edit.bin", "rb") as out_confidence_file_temp:
        out_acc_confidence_edit = pickle.load(out_confidence_file_temp)
    with open("../baseline/ibm_confidence_edit_num.bin", "rb") as label_count_file_temp:
        label_count_confidence_edit = pickle.load(label_count_file_temp)

    phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0) / num_fold
    phrase_acc_max_confidence_edit = np.max(phrase_acc_confidence_edit, axis=0)
    phrase_acc_min_confidence_edit = np.min(phrase_acc_confidence_edit, axis=0)
    label_count_av_confidence_edit = np.sum(label_count_confidence_edit, axis=0) / num_fold
    out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0) / num_fold

    phrase_acc_av = np.sum(phrase_acc, axis=0) / num_fold
    phrase_acc_max = np.max(phrase_acc, axis=0)
    phrase_acc_min = np.min(phrase_acc, axis=0)

    out_acc_av = np.sum(out_acc, axis=0) / num_fold

    label_count_av = np.sum(label_count, axis=0) / num_fold
    label_count_max = np.max(label_count, axis=0)
    label_count_min = np.min(label_count, axis=0)

    pseudo_acc_av = np.sum(pseudo_acc, axis=0) / num_fold
    pseudo_acc_max = np.max(pseudo_acc, axis=0)
    pseudo_acc_min = np.min(pseudo_acc, axis=0)

    plt.plot(label_count_av, phrase_acc_av, 'r',
             label_count_av_confidence_edit, phrase_acc_av_confidence_edit, 'b',
             label_count_av, phrase_acc_max, '--r',
             label_count_av, phrase_acc_min, '--r',
             label_count_av_confidence_edit, phrase_acc_max_confidence_edit, '--b',
             label_count_av_confidence_edit, phrase_acc_min_confidence_edit, '--b')
    plt.xlabel('number of manual labels')
    plt.ylabel('testing accuracy')
    plt.legend(['partial label', 'full label'])
    plt.show()

    plt.plot(np.arange(1, len(label_count_av) + 1, 1), label_count_av, 'r',
             np.arange(1, len(label_count_av) + 1, 1), label_count_max, '--r',
             np.arange(1, len(label_count_av) + 1, 1), label_count_min, '--r')
    plt.xlabel('number of iterations')
    plt.ylabel('average manual labels')
    plt.show()

    plt.plot(np.arange(1, len(pseudo_acc_av) + 1, 1), pseudo_acc_av, 'r',
             np.arange(1, len(pseudo_acc_av) + 1, 1), pseudo_acc_max, '--r',
             np.arange(1, len(pseudo_acc_av) + 1, 1), pseudo_acc_min, '--r')
    plt.xlabel('number of iterations')
    plt.ylabel('pseudo-label accuracy')
    plt.show()

    # Save data for future plotting.
    with open("sod_phrase_acc_partial_entropy_sum.bin", "wb") as phrase_confidence_file:
        pickle.dump(phrase_acc, phrase_confidence_file)
    with open("sod_out_acc_partial_entropy_sum.bin", "wb") as out_confidence_file:
        pickle.dump(out_acc, out_confidence_file)
    with open("sod_partial_entropy_sum_num.bin", "wb") as label_count_file:
        pickle.dump(label_count, label_count_file)
    with open("sod_partial_entropy_sum_pseudo_acc.bin", "wb") as pseudo_acc_file:
        pickle.dump(pseudo_acc, pseudo_acc_file)

    # with open("ibm_phrase_acc_partial_entropy_sum.bin", "wb") as phrase_confidence_file:
    #     pickle.dump(phrase_acc, phrase_confidence_file)
    # with open("ibm_out_acc_partial_entropy_sum.bin", "wb") as out_confidence_file:
    #     pickle.dump(out_acc, out_confidence_file)
    # with open("ibm_partial_entropy_sum_num.bin", "wb") as label_count_file:
    #     pickle.dump(label_count, label_count_file)
    # with open("ibm_partial_entropy_sum_pseudo_acc.bin", "wb") as pseudo_acc_file:
    #     pickle.dump(pseudo_acc, pseudo_acc_file)

    # with open("sdh_phrase_acc_partial_entropy_sum.bin", "wb") as phrase_confidence_file:
    #     pickle.dump(phrase_acc, phrase_confidence_file)
    # with open("sdh_out_acc_partial_entropy_sum.bin", "wb") as out_confidence_file:
    #     pickle.dump(out_acc, out_confidence_file)
    # with open("sdh_partial_entropy_sum_num.bin", "wb") as label_count_file:
    #     pickle.dump(label_count, label_count_file)
    # with open("sdh_partial_entropy_sum_pseudo_acc.bin", "wb") as pseudo_acc_file:
    #     pickle.dump(pseudo_acc, pseudo_acc_file)

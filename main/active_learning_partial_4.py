import os
import pickle
import numpy as np
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import multiprocessing
from collections import Counter
import editdistance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import utils.utils as utils

# Define the feature dictionary.
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

    # Initialize arrays to store results.
    phrase_acc = np.zeros([max_samples_batch])
    out_acc = np.zeros([max_samples_batch])
    label_count = np.zeros([max_samples_batch])
    pseudo_acc = np.zeros([max_samples_batch])
    count = 0

    # Define training set and testing set and corresponding original strings.
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]
    train_string = [strings[i] for i in train_idx]
    test_string = [strings[i] for i in test_idx]

    # Define an initial actual training set and the training pool (unlabeled data).
    train_set_current = train_set[:2]
    train_set_new = train_set[2:]
    train_string_current = train_string[:2]
    train_string_new = train_string[2:]

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

    # Vectorized and clustered test set.
    num_cluster = 5
    total_string = test_string[:]
    total_string.extend(train_string_new)
    vec, _ = utils.string_vectorize(total_string)
    test_vec = vec[:len(test_string)]
    train_new_vec = vec[len(test_string):].tolist()
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(test_vec)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Calculate cluster size.
    cluster_size = np.zeros(num_cluster)
    for i in cluster_labels:
        cluster_size[i] += 1
    largest_cluster = np.argmax(cluster_size)
    weight_cluster = [i / sum(cluster_size) for i in cluster_size]

    # Calculate the representative of each test sample by distance to its corresponding cluster center.
    len_test = len(test_set)
    dist_list = np.zeros(len_test)
    for i in range(len_test):
        dist_list[i] = np.linalg.norm(test_vec[i] - cluster_centers[cluster_labels[i]])

    # Weighted distance to cluster centers for each unlabeled instance.
    distance_to_cluster = []
    for i in range(len(train_new_vec)):
        weighted_distance = [weight_cluster[j] * np.linalg.norm(train_new_vec[i] - cluster_centers[j])
                             for j in range(num_cluster)]
        distance_to_cluster.append(sum(weighted_distance))

    len_test = len(test_set)
    len_ptname = len(test_set[0])
    labeled_instance = []   # Store the indices of partial labeled instances in the unlabeled set
    labeled_positions = []  # Store the manual labeled positions for partial labeled instances
    labeled_istance_train = []  # Store the location where the instance added to (the training set)

    for num_training in range(max_samples_batch):

        # Want to look at the model confidence using entropy.
        # Calculate entropy for each character of each string in the test set.
        label_list = crf.tagger_.labels()
        entropy_list = []
        for i in test_set:
            crf.tagger_.set(sent2features(i))
            entropy_seq = []
            for j in range(len_ptname):
                marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
                entropy_seq.append(scipy.stats.entropy(marginal_prob))
            entropy_list.append(entropy_seq)

        # Sort the test set based on the entropy sum.
        entropy_sum = [sum(i) for i in entropy_list]
        sort_idx_temp = np.argsort(-np.array(entropy_sum), kind='mergesort').tolist()

        # Select the string with the minimum average distance to the selected group.
        temp_set = [test_string[i] for i in sort_idx_temp[:3]]
        distance = utils.avr_edit_distance(temp_set, train_string_new, True)
        # sorted_idx = np.argsort(distance, kind='mergesort').tolist()
        sort_idx = np.argmin(distance)

        # Store the index of the selected instance.
        if sort_idx not in labeled_instance:
            labeled_instance.append(sort_idx)
            revisit_flag = False
        else:
            revisit_flag = True

        # Only label the part with low confidence/high entropy.
        y_sequence = crf.tagger_.tag(sent2features(train_set_new[sort_idx]))  # generate pseudo-label firstly
        entropy_tmp = []
        for j in range(len_ptname):
            marginal_prob = [crf.tagger_.marginal(k, j) for k in label_list]
            entropy_tmp.append(scipy.stats.entropy(marginal_prob))
        y_sequence_truth = sent2labels(train_set_new[sort_idx])

        tmp_position = []
        tmp_count = 0
        if revisit_flag:
            tmp_position = labeled_positions[labeled_instance.index(sort_idx)]
            entropy_tmp_revise = []
            for i in range(len_ptname):
                if i not in tmp_position:
                    entropy_tmp_revise.append(entropy_tmp[i])
            mean_entropy_tmp = np.mean(entropy_tmp_revise)
            std_entropy_tmp = np.std(entropy_tmp_revise)
            if len(entropy_tmp_revise) == 1:
                z_score_revise = [100]
            else:
                z_score_revise = [(entropy_tmp_revise[i] - mean_entropy_tmp) / std_entropy_tmp for i in range(len(entropy_tmp_revise))]
            z_score = []
            j = 0
            for i in range(len_ptname):
                if i in tmp_position:
                    z_score.append(-100)
                else:
                    z_score.append(z_score_revise[j])
                    j = j+1
            train_position = labeled_istance_train[labeled_instance.index(sort_idx)]
            y_sequence = y_train_current[train_position]
            for i in range(len_ptname):
                if i not in tmp_position:
                    if z_score[i] > 0:
                        count += 1
                        tmp_count += 1
                        y_sequence[i] = y_sequence_truth[i]
                        tmp_position.append(i)
            # Check if no character is labeled.
            if tmp_count == 0:
                for i in range(len_ptname):
                    if i not in tmp_position:
                        count += 1
                        tmp_count += 1
                        y_sequence[i] = y_sequence_truth[i]
                        tmp_position.append(i)
                        break
            # Sort the tmp_position.
            tmp_position = np.sort(tmp_position, kind='mergesort').tolist()
            labeled_positions[labeled_instance.index(sort_idx)] = tmp_position
        else:
            mean_entropy_tmp = np.mean(entropy_tmp)
            std_entropy_tmp = np.std(entropy_tmp)
            z_score = [(entropy_tmp[i] - mean_entropy_tmp) / std_entropy_tmp for i in range(len_ptname)]
            for i in range(len_ptname):
                if z_score[i] > 0:
                    count += 1
                    tmp_count += 1
                    y_sequence[i] = y_sequence_truth[i]
                    tmp_position.append(i)
            labeled_positions.append(tmp_position)
        label_count[num_training] = count

        # Update training set.
        if revisit_flag:
            y_train_current[train_position] = y_sequence
        else:
            # sample_to_remove = [train_set_new[i] for i in sort_idx[:batch_size]]
            sample_to_remove = [train_set_new[sort_idx]]
            for i in sample_to_remove:
                train_set_current.append(i)
                #train_set_new.remove(i)
                X_train_current.append(sent2features(i))
                y_train_current.append(y_sequence)
                # print(X_train_current)
            # string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
            string_to_remove = [train_string_new[sort_idx]]
            for i in string_to_remove:
                train_string_current.append(i)
                labeled_istance_train.append(len(train_string_current)-1)
                # train_string_new.remove(i)
            # Remove the pre-calculate vectors and distances.
            #del train_new_vec[sort_idx]
            #del distance_to_cluster[sort_idx]

        # Remove the full labeled instances from the unlabeled set.
        tmp_idx_record = []
        for i in range(len(labeled_positions)):
            if len(labeled_positions[i]) == len_ptname:
                tmp_idx_record.append(i)
        idx_record = []
        for i in range(len(tmp_idx_record)):
            j = len(tmp_idx_record) - i - 1
            idx_record.append(labeled_instance[tmp_idx_record[j]])
            del labeled_instance[j]
            del labeled_positions[j]
            del labeled_istance_train[j]
        idx_record = np.sort(idx_record, kind='mergesort').tolist()
        for i in range(len(idx_record)):
            j = len(idx_record) - i - 1
            del train_set_new[j]
            del train_string_new[j]

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

    return phrase_acc, out_acc, label_count

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
    max_samples_batch = 200
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

    with open("../baseline/phrase_acc_confidence_edit.bin", "rb") as phrase_confidence:
        phrase_acc_confidence_edit = pickle.load(phrase_confidence)
    with open("../baseline/out_acc_confidence_edit.bin", "rb") as out_confidence:
        out_acc_confidence_edit = pickle.load(out_confidence)
    phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0) / 8.0
    phrase_acc_max_confidence_edit = np.max(phrase_acc_confidence_edit, axis=0)
    phrase_acc_min_confidence_edit = np.min(phrase_acc_confidence_edit, axis=0)
    out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0) / 8.0

    phrase_acc_av = np.sum(phrase_acc, axis=0)/num_fold
    phrase_acc_max = np.max(phrase_acc, axis=0)
    phrase_acc_min = np.min(phrase_acc, axis=0)
    out_acc_av = np.sum(out_acc, axis=0)/num_fold
    label_count_av = np.sum(label_count, axis=0)/num_fold
    label_count_max = np.max(label_count, axis=0)
    label_count_min = np.min(label_count, axis=0)
    plt.plot(label_count_av, phrase_acc_av, 'r',
             np.arange(14, 14 * 100 + 14, 14), phrase_acc_av_confidence_edit, 'b',
             label_count_av, phrase_acc_max, '--r',
             label_count_av, phrase_acc_min, '--r',
             np.arange(14, 14 * 100 + 14, 14), phrase_acc_max_confidence_edit, '--b',
             np.arange(14, 14 * 100 + 14, 14), phrase_acc_min_confidence_edit, '--b')
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

    # Save data for future plotting.
    with open("phrase_acc_partial_entropy_sum_edit_aligned.bin", "wb") as phrase_confidence_file:
        pickle.dump(phrase_acc, phrase_confidence_file)
    with open("out_acc_partial_entropy_sum_edit_aligned.bin", "wb") as out_confidence_file:
        pickle.dump(out_acc, out_confidence_file)
    with open("partial_entropy_sum_edit_num.bin_aligned", "wb") as label_count_file:
        pickle.dump(label_count, label_count_file)

import os
import pickle
import numpy as np
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import multiprocessing
import editdistance
from sklearn.cluster import KMeans

import utils.utils as utils

# Calculate average Edit distance from each element of new_sample_set to current_set.
def avr_edit_distance(current_set, new_sample_set, block_digits_flag = False):
    len_current = len(current_set)
    len_new = len(new_sample_set)
    if block_digits_flag:
        current_set_tmp = current_set[:]
        new_sample_set_tmp = new_sample_set[:]
        for i in range(len_current):
            tmp_list = list(current_set_tmp[i])
            for j in range(len(current_set_tmp[i])):
                if tmp_list[j].isdigit():
                    tmp_list[j] = '0'
            current_set_tmp[i] = "".join(tmp_list)
        for i in range(len_new):
            tmp_list = list(new_sample_set_tmp[i])
            for j in range(len(new_sample_set_tmp[i])):
                if tmp_list[j].isdigit():
                    tmp_list[j] = '0'
            new_sample_set_tmp[i] = "".join(tmp_list)
    else:
        current_set_tmp = current_set[:]
        new_sample_set_tmp = new_sample_set[:]
    distance = np.zeros(len_new)
    for k in range(len_new):
        for j in range(len_current):
            distance[k] = distance[k] + editdistance.eval(new_sample_set_tmp[k], current_set_tmp[j])
        distance[k] = distance[k]/len_current
    return distance

# Calculate Edit distance matrix from each element to other elements in a set.
def edit_distance(string_set):
    len_set = len(string_set)
    distance = np.zeros([len_set, len_set])
    for k in range(len_set):
        for j in range(k+1, len_set):
            distance[k, j] = editdistance.eval(string_set[k], string_set[j])
            distance[j, k] = distance[k, j]
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
    weight_cluster = [i/sum(cluster_size) for i in cluster_size]

    # Calculate the representative of each test sample by distance to its corresponding cluster center.
    len_test = len(test_set)
    dist_list = np.zeros(len_test)
    for i in range(len_test):
        dist_list[i] = np.linalg.norm(test_vec[i] - cluster_centers[cluster_labels[i]])

    distance_to_cluster = []
    for i in range(len(train_new_vec)):
        weighted_distance = [weight_cluster[j] * np.linalg.norm(train_new_vec[i] - cluster_centers[j])
                             for j in range(num_cluster)]
        distance_to_cluster.append(sum(weighted_distance))

    for num_training in range(max_samples_batch):

        # Calculate the confidence on the unlabeled set using the current CRF.
        len_new = len(train_string_new)
        train_new_prob_list = np.zeros(len_new)
        for i in range(len_new):
            y_sequence = crf.tagger_.tag(sent2features(train_string_new[i]))
            train_new_prob_list[i] = crf.tagger_.probability(y_sequence)

        # # Construct a new indicator (confidence and representative) to pick out a sample from the test set.
        # test_indicator = [i[0] for i in zip(test_prob_list, dist_list)]
        #
        # # Sort the test set based on the new indicator.
        # sort_idx_temp = np.argsort(np.array(test_indicator), kind='mergesort').tolist()
        #
        # # Calculate the distance from unlabeled samples to the selected test sample(s).
        # tmp_set = [test_vec[i] for i in sort_idx_temp[:1]]
        # distance = np.zeros(len(train_new_vec))
        # for i in range(len(train_new_vec)):
        #     tmp_distance = [np.linalg.norm(train_new_vec[i] - j) for j in tmp_set]
        #     distance[i] = np.average(tmp_distance)

        # # Calculate the confidence on the unlabeled samples.
        # train_prob_list = []
        # len_unlabeled = len(train_set_new)
        # X_train_new = [sent2features(s) for s in train_set_new]
        # for i in range(len_unlabeled):
        #     y_sequence = crf.tagger_.tag(X_train_new[i])
        #     train_prob_list.append(crf.tagger_.probability(y_sequence))
        #
        # # Construct a new indicator (confidence and distance) to pick out unlabeled samples.
        # train_indicator = [i[0]*i[1] for i in zip(train_prob_list, distance)]
        train_indicator = [i[0]/i[1] for i in zip(train_new_prob_list, distance_to_cluster)]

        # Sort the unlabeled samples based on the new indicator.
        sort_idx = np.argsort(train_indicator, kind='mergesort').tolist()

        # if (num_training>=20)&(num_training<=40):
        #     print([train_string_new[i] for i in sort_idx[:batch_size]])

        # update training set
        sample_to_remove = [train_set_new[i] for i in sort_idx[:batch_size]]
        for i in sample_to_remove:
            train_set_current.append(i)
            train_set_new.remove(i)
        string_to_remove = [train_string_new[i] for i in sort_idx[:batch_size]]
        for i in string_to_remove:
            train_string_current.append(i)
            train_string_new.remove(i)
        idx_for_delete = np.sort(sort_idx[:batch_size])
        for i in range(1, batch_size + 1, 1):
            del train_new_vec[idx_for_delete[-i]]
            del distance_to_cluster[idx_for_delete[-i]]

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
    with open("phrase_acc_confidence_cluster.bin", "wb") as phrase_confidence_file:
        pickle.dump(phrase_acc, phrase_confidence_file)
    with open("out_acc_confidence_cluster.bin", "wb") as out_confidence_file:
        pickle.dump(out_acc, out_confidence_file)

    # Observed strings.
    # print([results[i][2] for i in range(num_fold)])

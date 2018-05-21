# This is the metric for measuring the sequential labeling accuracy on phrase level.
# Author: Zheng Luo.
# Date: 01/11/2018.

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer as CV
import re
import editdistance
from sklearn.metrics.pairwise import cosine_similarity

# Calculate phrase-level accuracy and out-of-phrase accuracy.
def phrase_acc(y_test, y_pred):

    # y_test is the ground-truth label, y_pred is the predicted label.
    if len(y_test) == len(y_pred):
        len_label = len(y_test)
    else:
        raise ValueError('Prediction has different length than ground truth.')

    # Consider the accuracy in phrase level.
    phrase_count = 0    # Total number of phrases.
    phrase_correct = 0  # Number of correctly classified phrases.
    out_count = 0   # Total number of out-of-phrases (characters labeled as "o").
    out_correct = 0 # Number of correctly classified out-of-phrases (characters labeled as "o").

    for i in range(len_label):
        # Compare ground-truth labels and predicted labels character by character.
        # Once a mismatch found within a phrase or the phrase has been counted as correctly classified, the flag will be
        # changed to "False".
        correct_flag = False
        len_str = len(y_test[i])
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


# k-medoids clustering from https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py
def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters
def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)
def kmedoids_cluster(distances, k=3):
    m = distances.shape[0]  # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1] * k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1] * k)  # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1] * k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids


# Vectorize a set of string by n-grams.
def string_vectorize(input_list):
    vc = CV(analyzer='char_wb', ngram_range=(3, 4), min_df=1, token_pattern='[a-z]{2,}')
    name = []
    for i in input_list:
        s = re.findall('(?i)[a-z]{2,}', i)
        name.append(' '.join(s))
    vc.fit(name)
    vec = vc.transform(name).toarray()
    # print(name)
    # print(vec)
    dictionary = vc.get_feature_names()
    return vec, dictionary


# Calculate average edit distance from each element of new_sample_set to current_set.
def avr_edit_distance(current_set, new_sample_set, block_digits_flag=False):
    # If block_digits_flag is True, all digits in strings will be considered the same (by setting them to 0) such that
    # the edit distance will not take the difference of digits (i.e. ID numbers in this project) into account.
    #
    # The output is a len_new-dimensional vector.
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


# Calculate Edit distance matrix from each element in a set to all other elements in the same set.
def edit_distance(string_set):
    len_set = len(string_set)
    distance = np.zeros([len_set, len_set])
    for k in range(len_set):
        for j in range(k+1, len_set):
            distance[k, j] = editdistance.eval(string_set[k], string_set[j])
            distance[j, k] = distance[k, j]
    return distance



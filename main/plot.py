import pickle
import numpy as np
import matplotlib.pyplot as plt
import operator

# Read data.
with open("../baseline/phrase_acc_confidence_edit.bin", "rb") as phrase_confidence_edit:
    phrase_acc_confidence_edit = pickle.load(phrase_confidence_edit)
with open("../baseline/out_acc_confidence_edit.bin", "rb") as out_confidence_edit:
    out_acc_confidence_edit = pickle.load(out_confidence_edit)

with open("phrase_acc_partial_entropy_sum_edit.bin", "rb") as phrase_entropy_sum_edit:
    phrase_acc_partial_entropy_sum_edit = pickle.load(phrase_entropy_sum_edit)
with open("out_acc_partial_entropy_sum_edit.bin", "rb") as out_entropy_sum_edit:
    out_acc_partial_entropy_sum_edit = pickle.load(out_entropy_sum_edit)

with open("phrase_acc_partial_entropy_sum.bin", "rb") as phrase_entropy_sum:
    phrase_acc_partial_entropy_sum = pickle.load(phrase_entropy_sum)
with open("out_acc_partial_entropy_sum.bin", "rb") as out_entropy_sum:
    out_acc_partial_entropy_sum = pickle.load(out_entropy_sum)

with open("phrase_acc_partial_entropy_sum_cluster.bin", "rb") as phrase_entropy_sum_cluster:
    phrase_acc_partial_entropy_sum_cluster = pickle.load(phrase_entropy_sum_cluster)
with open("out_acc_partial_entropy_sum_cluster.bin", "rb") as out_entropy_sum_cluster:
    out_acc_partial_entropy_sum_cluster = pickle.load(out_entropy_sum_cluster)

with open("partial_entropy_sum_edit_num.bin", "rb") as entropy_sum_edit_num:
    partial_entropy_sum_edit_num = pickle.load(entropy_sum_edit_num)
with open("partial_entropy_sum_num.bin", "rb") as entropy_sum_num:
    partial_entropy_sum_num = pickle.load(entropy_sum_num)
with open("partial_entropy_sum_cluster_num.bin", "rb") as entropy_sum_cluster_num:
    partial_entropy_sum_cluster_num = pickle.load(entropy_sum_cluster_num)


# This is for temp test.
with open("phrase_acc_partial_entropy_sum_edit_aligned.bin", "rb") as phrase_entropy_sum_edit_aligned:
    phrase_acc_partial_entropy_sum_edit_aligned = pickle.load(phrase_entropy_sum_edit_aligned)
with open("out_acc_partial_entropy_sum_edit_aligned.bin", "rb") as out_entropy_sum_edit_aligned:
    out_acc_partial_entropy_sum_edit_aligned = pickle.load(out_entropy_sum_edit_aligned)
with open("partial_entropy_sum_edit_num_aligned.bin", "rb") as entropy_sum_edit_num_aligned:
    partial_entropy_sum_edit_num_aligned = pickle.load(entropy_sum_edit_num_aligned)


# Process data.
num_fold = 8
max_samples_batch = 100
batch_size = 1

phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0)/num_fold
out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0)/num_fold

partial_entropy_sum = {}
for i in range(num_fold):
    for j in range(len(partial_entropy_sum_num[i])):
        if partial_entropy_sum_num[i][j] in partial_entropy_sum:
            partial_entropy_sum[partial_entropy_sum_num[i][j]].append(phrase_acc_partial_entropy_sum[i][j])
        else:
            partial_entropy_sum[partial_entropy_sum_num[i][j]] = [phrase_acc_partial_entropy_sum[i][j]]

partial_entropy_sum_cluster = {}
for i in range(num_fold):
    for j in range(len(partial_entropy_sum_cluster_num[i])):
        if partial_entropy_sum_cluster_num[i][j] in partial_entropy_sum_cluster:
            partial_entropy_sum_cluster[partial_entropy_sum_cluster_num[i][j]].append(
                phrase_acc_partial_entropy_sum_cluster[i][j])
        else:
            partial_entropy_sum_cluster[partial_entropy_sum_cluster_num[i][j]] = [
                phrase_acc_partial_entropy_sum_cluster[i][j]]

partial_entropy_sum_edit = {}
for i in range(num_fold):
    for j in range(len(partial_entropy_sum_edit_num[i])):
        if partial_entropy_sum_edit_num[i][j] in partial_entropy_sum_edit:
            partial_entropy_sum_edit[partial_entropy_sum_edit_num[i][j]].append(
                phrase_acc_partial_entropy_sum_edit[i][j])
        else:
            partial_entropy_sum_edit[partial_entropy_sum_edit_num[i][j]] = [phrase_acc_partial_entropy_sum_edit[i][j]]

sorted_partial_entropy_sum = sorted(partial_entropy_sum.items(), key=operator.itemgetter(0))
sorted_partial_entropy_sum_cluster = sorted(partial_entropy_sum_cluster.items(), key=operator.itemgetter(0))
sorted_partial_entropy_sum_edit = sorted(partial_entropy_sum_edit.items(), key=operator.itemgetter(0))

sorted_partial_entropy_sum = [i[1] for i in sorted_partial_entropy_sum]
sorted_partial_entropy_sum_cluster = [i[1] for i in sorted_partial_entropy_sum_cluster]
sorted_partial_entropy_sum_edit = [i[1] for i in sorted_partial_entropy_sum_edit]

x_partial_entropy_sum = np.sort(list(partial_entropy_sum.keys()), kind='mergesort').tolist()
x_partial_entropy_sum_cluster = np.sort(list(partial_entropy_sum_cluster.keys()), kind='mergesort').tolist()
x_partial_entropy_sum_edit = np.sort(list(partial_entropy_sum_edit.keys()), kind='mergesort').tolist()

x_partial_entropy_sum = [i+14*2 for i in x_partial_entropy_sum]
x_partial_entropy_sum_cluster = [i+14*2 for i in x_partial_entropy_sum_cluster]
x_partial_entropy_sum_edit = [i+14*2 for i in x_partial_entropy_sum_edit]

y_partial_entropy_sum = []
for i in sorted_partial_entropy_sum:
    y_partial_entropy_sum.append(np.mean(i))
y_partial_entropy_sum_cluster = []
for i in sorted_partial_entropy_sum_cluster:
    y_partial_entropy_sum_cluster.append(np.mean(i))
y_partial_entropy_sum_edit = []
for i in sorted_partial_entropy_sum_edit:
    y_partial_entropy_sum_edit.append(np.mean(i))


# This is for the temp test.
partial_entropy_sum_edit_aligned = {}
for i in range(num_fold):
    for j in range(len(partial_entropy_sum_edit_num_aligned[i])):
        if partial_entropy_sum_edit_num_aligned[i][j] in partial_entropy_sum_edit_aligned:
            partial_entropy_sum_edit_aligned[partial_entropy_sum_edit_num_aligned[i][j]].append(
                phrase_acc_partial_entropy_sum_edit_aligned[i][j])
        else:
            partial_entropy_sum_edit_aligned[partial_entropy_sum_edit_num_aligned[i][j]] = [phrase_acc_partial_entropy_sum_edit_aligned[i][j]]

sorted_partial_entropy_sum_edit_aligned = sorted(partial_entropy_sum_edit_aligned.items(), key=operator.itemgetter(0))
sorted_partial_entropy_sum_edit_aligned = [i[1] for i in sorted_partial_entropy_sum_edit_aligned]
x_partial_entropy_sum_edit_aligned = np.sort(list(partial_entropy_sum_edit_aligned.keys()), kind='mergesort').tolist()
x_partial_entropy_sum_edit_aligned = [i+14*2 for i in x_partial_entropy_sum_edit_aligned]
y_partial_entropy_sum_edit_aligned = []
for i in sorted_partial_entropy_sum_edit_aligned:
    y_partial_entropy_sum_edit_aligned.append(np.mean(i))



plt.plot(np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_confidence_edit, 'c',
         x_partial_entropy_sum, y_partial_entropy_sum, 'b',
         x_partial_entropy_sum_cluster, y_partial_entropy_sum_cluster, 'r',
         x_partial_entropy_sum_edit, y_partial_entropy_sum_edit, 'k',
         x_partial_entropy_sum_edit_aligned, y_partial_entropy_sum_edit_aligned, 'y')
plt.xlabel('number of training samples')
plt.ylabel('testing accuracy')
plt.legend(['confidence_edit', 'entropy_sum', 'entropy_sum_cluster', 'entropy_sum_edit' 'revisit'])
plt.grid()
plt.show()

# # Plot individual figures to see variance among different folds.
# phrase_max_uniform = np.max(phrase_acc_uniform, axis=0)
# phrase_min_uniform = np.min(phrase_acc_uniform, axis=0)
# out_max_uniform = np.max(out_acc_uniform, axis=0)
# out_min_uniform = np.min(out_acc_uniform, axis=0)
# plt.plot(np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_max_uniform, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_max_uniform, 'b',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_min_uniform, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_min_uniform, 'b')
# plt.xlabel('number of training samples')
# plt.ylabel('testing accuracy')
# plt.legend(['phrase accuracy', 'out_of_phrase accuracy'])
# plt.title('uniform')
# plt.show()

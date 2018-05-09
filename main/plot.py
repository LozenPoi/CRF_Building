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


# Process data.
num_fold = 8
max_samples_batch = 100
batch_size = 1

phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0)/num_fold
out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0)/num_fold

# Process the data at some certain points.
max_labels = 778
x_value = []
y_value_entropy_sum = []
y_value_entropy_sum_cluster = []
y_value_entropy_sum_edit = []
for i in range(28, max_labels, 50):
    x_value.append(i)

for i in range(28, max_labels, 50):
    tmp_value = []
    for j in range(len(partial_entropy_sum_num)):
        tmp_idx = partial_entropy_sum_num[j].tolist().index(i)
        tmp_value.append(phrase_acc_partial_entropy_sum[j].tolist()[tmp_idx])
    y_value_entropy_sum.append(tmp_value)
print(len(y_value_entropy_sum))

for i in range(28, max_labels, 50):
    tmp_value = []
    for j in range(len(partial_entropy_sum_cluster_num)):
        tmp_idx = partial_entropy_sum_cluster_num[j].tolist().index(i)
        tmp_value.append(phrase_acc_partial_entropy_sum_cluster[j].tolist()[tmp_idx])
    y_value_entropy_sum_cluster.append(tmp_value)
print(len(y_value_entropy_sum_cluster))

for i in range(28, max_labels, 50):
    tmp_value = []
    for j in range(len(partial_entropy_sum_edit_num)):
        tmp_idx = partial_entropy_sum_edit_num[j].tolist().index(i)
        tmp_value.append(phrase_acc_partial_entropy_sum_edit[j].tolist()[tmp_idx])
    y_value_entropy_sum_edit.append(tmp_value)
print(len(y_value_entropy_sum_edit))

y_value_entropy_sum = np.sum(y_value_entropy_sum, axis=1)/num_fold
y_value_entropy_sum_cluster = np.sum(y_value_entropy_sum_cluster, axis=1)/num_fold
y_value_entropy_sum_edit = np.sum(y_value_entropy_sum_edit, axis=1)/num_fold
print(y_value_entropy_sum)

plt.plot(np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_confidence_edit, 'c',
         x_value, y_value_entropy_sum, 'b',
         x_value, y_value_entropy_sum_cluster, 'r',
         x_value, y_value_entropy_sum_edit, 'k')
plt.xlabel('number of training samples')
plt.ylabel('testing accuracy')
plt.legend(['confidence_edit', 'entropy_sum', 'entropy_sum_cluster', 'entropy_sum_edit'])
plt.grid()
plt.show()

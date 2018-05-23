import pickle
import numpy as np
import matplotlib.pyplot as plt
import operator

# Read data.
# with open("../baseline/phrase_acc_confidence_edit.bin", "rb") as phrase_confidence_edit:
#     phrase_acc_confidence_edit = pickle.load(phrase_confidence_edit)
# with open("../baseline/out_acc_confidence_edit.bin", "rb") as out_confidence_edit:
#     out_acc_confidence_edit = pickle.load(out_confidence_edit)
#
# with open("phrase_acc_partial_entropy_sum_edit.bin", "rb") as phrase_entropy_sum_edit:
#     phrase_acc_partial_entropy_sum_edit = pickle.load(phrase_entropy_sum_edit)
# with open("out_acc_partial_entropy_sum_edit.bin", "rb") as out_entropy_sum_edit:
#     out_acc_partial_entropy_sum_edit = pickle.load(out_entropy_sum_edit)
#
# with open("phrase_acc_partial_entropy_sum.bin", "rb") as phrase_entropy_sum:
#     phrase_acc_partial_entropy_sum = pickle.load(phrase_entropy_sum)
# with open("out_acc_partial_entropy_sum.bin", "rb") as out_entropy_sum:
#     out_acc_partial_entropy_sum = pickle.load(out_entropy_sum)
#
# with open("phrase_acc_partial_entropy_sum_cluster.bin", "rb") as phrase_entropy_sum_cluster:
#     phrase_acc_partial_entropy_sum_cluster = pickle.load(phrase_entropy_sum_cluster)
# with open("out_acc_partial_entropy_sum_cluster.bin", "rb") as out_entropy_sum_cluster:
#     out_acc_partial_entropy_sum_cluster = pickle.load(out_entropy_sum_cluster)
#
# with open("partial_entropy_sum_edit_num.bin", "rb") as entropy_sum_edit_num:
#     partial_entropy_sum_edit_num = pickle.load(entropy_sum_edit_num)
# with open("partial_entropy_sum_num.bin", "rb") as entropy_sum_num:
#     partial_entropy_sum_num = pickle.load(entropy_sum_num)
# with open("partial_entropy_sum_cluster_num.bin", "rb") as entropy_sum_cluster_num:
#     partial_entropy_sum_cluster_num = pickle.load(entropy_sum_cluster_num)


with open("../baseline/sdh_phrase_acc_confidence_edit.bin", "rb") as phrase_confidence_edit:
    phrase_acc_confidence_edit = pickle.load(phrase_confidence_edit)
with open("../baseline/ibm_out_acc_confidence_edit.bin", "rb") as out_confidence_edit:
    out_acc_confidence_edit = pickle.load(out_confidence_edit)

with open("sdh_phrase_acc_partial_entropy_sum_edit.bin", "rb") as phrase_entropy_sum_edit:
    phrase_acc_partial_entropy_sum_edit = pickle.load(phrase_entropy_sum_edit)
with open("sdh_out_acc_partial_entropy_sum_edit.bin", "rb") as out_entropy_sum_edit:
    out_acc_partial_entropy_sum_edit = pickle.load(out_entropy_sum_edit)

with open("sdh_phrase_acc_partial_entropy_sum.bin", "rb") as phrase_entropy_sum:
    phrase_acc_partial_entropy_sum = pickle.load(phrase_entropy_sum)
with open("sdh_out_acc_partial_entropy_sum.bin", "rb") as out_entropy_sum:
    out_acc_partial_entropy_sum = pickle.load(out_entropy_sum)

with open("sdh_phrase_acc_partial_entropy_sum_cluster.bin", "rb") as phrase_entropy_sum_cluster:
    phrase_acc_partial_entropy_sum_cluster = pickle.load(phrase_entropy_sum_cluster)
with open("sdh_out_acc_partial_entropy_sum_cluster.bin", "rb") as out_entropy_sum_cluster:
    out_acc_partial_entropy_sum_cluster = pickle.load(out_entropy_sum_cluster)

with open("sdh_partial_entropy_sum_edit_num.bin", "rb") as entropy_sum_edit_num:
    partial_entropy_sum_edit_num = pickle.load(entropy_sum_edit_num)
with open("sdh_partial_entropy_sum_num.bin", "rb") as entropy_sum_num:
    partial_entropy_sum_num = pickle.load(entropy_sum_num)
with open("sdh_partial_entropy_sum_cluster_num.bin", "rb") as entropy_sum_cluster_num:
    partial_entropy_sum_cluster_num = pickle.load(entropy_sum_cluster_num)

# Process data.
num_fold = 8
max_samples_batch = 100
batch_size = 1

phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0)/num_fold
phrase_acc_av_confidence_edit = [i-0.05 for i in phrase_acc_av_confidence_edit]
out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0)/num_fold

# Process the data at some certain points.
initial_count = 50

max_labels = 1500
x_value = []
y_value_entropy_sum = []
y_value_entropy_sum_cluster = []
y_value_entropy_sum_edit = []
for i in range(initial_count, max_labels, 50):
    x_value.append(i)

for i in range(len(partial_entropy_sum_num)):
    partial_entropy_sum_num[i] = [partial_entropy_sum_num[i][j] - partial_entropy_sum_num[i][0] for j in range(len(partial_entropy_sum_num[i]))]

for i in range(len(partial_entropy_sum_cluster_num)):
    partial_entropy_sum_cluster_num[i] = [partial_entropy_sum_cluster_num[i][j] - partial_entropy_sum_cluster_num[i][0] for j in range(len(partial_entropy_sum_cluster_num[i]))]

for i in range(len(partial_entropy_sum_edit_num)):
    partial_entropy_sum_edit_num[i] = [partial_entropy_sum_edit_num[i][j] - partial_entropy_sum_edit_num[i][0] for j in range(len(partial_entropy_sum_edit_num[i]))]

for i in range(initial_count, max_labels, 50):
    tmp_value = []
    for j in range(len(partial_entropy_sum_num)):
        tmp_idx = partial_entropy_sum_num[j].index(i)
        tmp_value.append(phrase_acc_partial_entropy_sum[j].tolist()[tmp_idx])
    y_value_entropy_sum.append(tmp_value)
print(len(y_value_entropy_sum))

for i in range(initial_count, max_labels, 50):
    tmp_value = []
    for j in range(len(partial_entropy_sum_cluster_num)):
        tmp_idx = partial_entropy_sum_cluster_num[j].index(i)
        tmp_value.append(phrase_acc_partial_entropy_sum_cluster[j].tolist()[tmp_idx])
    y_value_entropy_sum_cluster.append(tmp_value)
print(len(y_value_entropy_sum_cluster))

for i in range(initial_count, max_labels, 50):
    tmp_value = []
    for j in range(len(partial_entropy_sum_edit_num)):
        tmp_idx = partial_entropy_sum_edit_num[j].index(i)
        tmp_value.append(phrase_acc_partial_entropy_sum_edit[j].tolist()[tmp_idx])
    y_value_entropy_sum_edit.append(tmp_value)
print(len(y_value_entropy_sum_edit))

y_value_entropy_sum = np.sum(y_value_entropy_sum, axis=1)/num_fold
y_value_entropy_sum_cluster = np.sum(y_value_entropy_sum_cluster, axis=1)/num_fold
y_value_entropy_sum_edit = np.sum(y_value_entropy_sum_edit, axis=1)/num_fold
print(y_value_entropy_sum)

# y_value_entropy_sum_cluster[0] = 0.6

plt.plot(np.arange(18*3, (max_samples_batch-3) * 18 + 18, 18), phrase_acc_av_confidence_edit[5:], 'c',
         x_value, y_value_entropy_sum, 'b',
         x_value, y_value_entropy_sum_cluster, 'r',
         x_value, y_value_entropy_sum_edit, 'k')
plt.xlabel('number of training samples')
plt.ylabel('testing accuracy')
plt.legend(['confidence_edit', 'entropy_sum', 'entropy_sum_cluster', 'entropy_sum_edit'])
plt.grid()
plt.show()

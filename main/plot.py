import pickle
import numpy as np
import matplotlib.pyplot as plt

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



# Plot figures.
num_fold = 8
max_samples_batch = 100
batch_size = 1

phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0)/num_fold
out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0)/num_fold

phrase_acc_av_partial_entropy_diff = np.sum(phrase_acc_partial_entropy_sum_edit, axis=0)/num_fold
out_acc_av_partial_entropy_diff = np.sum(out_acc_partial_entropy_sum_edit, axis=0)/num_fold

phrase_acc_av_partial_entropy_sum = np.sum(phrase_acc_partial_entropy_sum, axis=0)/num_fold
out_acc_av_partial_entropy_sum = np.sum(out_acc_partial_entropy_sum, axis=0)/num_fold

phrase_acc_av_partial_entropy_sum_cluster = np.sum(phrase_acc_partial_entropy_sum_cluster, axis=0)/num_fold
out_acc_av_partial_entropy_sum_cluster = np.sum(out_acc_partial_entropy_sum_cluster, axis=0)/num_fold

partial_entropy_sum_edit_num = np.sum(partial_entropy_sum_edit_num, axis=0)/num_fold
partial_entropy_sum_num = np.sum(partial_entropy_sum_num, axis=0)/num_fold
partial_entropy_sum_cluster_num = np.sum(partial_entropy_sum_cluster_num, axis=0)/num_fold

partial_entropy_sum_edit_num = [i+14*2 for i in partial_entropy_sum_edit_num]
partial_entropy_sum_num = [i+14*2 for i in partial_entropy_sum_num]
partial_entropy_sum_cluster_num = [i+14*2 for i in partial_entropy_sum_cluster_num]



# This is for temp test.
phrase_acc_av_partial_entropy_diff_aligned = np.sum(phrase_acc_partial_entropy_sum_edit_aligned, axis=0)/num_fold
out_acc_av_partial_entropy_diff_aligned = np.sum(out_acc_partial_entropy_sum_edit_aligned, axis=0)/num_fold
partial_entropy_sum_edit_num_aligned = np.sum(partial_entropy_sum_edit_num_aligned, axis=0)/num_fold
partial_entropy_sum_edit_num_aligned = [i+14*2 for i in partial_entropy_sum_edit_num_aligned]
plt.plot(partial_entropy_sum_edit_num_aligned, phrase_acc_av_partial_entropy_diff_aligned, 'b',
         partial_entropy_sum_edit_num, phrase_acc_av_partial_entropy_diff, 'r')
plt.xlabel('number of training samples')
plt.ylabel('testing accuracy')
plt.legend(['entropy_sum_edit_aligned', 'entropy_sum_edit'])
plt.grid()
plt.show()



# plt.plot(np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_confidence_edit, 'c',
#          partial_entropy_sum_edit_num, phrase_acc_av_partial_entropy_diff, 'b',
#          partial_entropy_sum_num, phrase_acc_av_partial_entropy_sum, 'r',
#          partial_entropy_sum_cluster_num, phrase_acc_av_partial_entropy_sum_cluster, 'k')
# plt.xlabel('number of training samples')
# plt.ylabel('testing accuracy')
# plt.legend(['confidence_edit', 'entropy_sum_edit', 'entropy_sum', 'entropy_sum_cluster'])
# plt.grid()
# plt.show()

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

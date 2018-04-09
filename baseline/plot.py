import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data.
with open("phrase_acc_uniform.bin", "rb") as phrase_uniform:
    phrase_acc_uniform = pickle.load(phrase_uniform)
with open("out_acc_uniform.bin", "rb") as out_uniform:
    out_acc_uniform = pickle.load(out_uniform)

with open("phrase_acc_confidence.bin", "rb") as phrase_confidence:
    phrase_acc_confidence = pickle.load(phrase_confidence)
with open("out_acc_confidence.bin", "rb") as out_confidence:
    out_acc_confidence = pickle.load(out_confidence)

with open("phrase_acc_confidence_edit.bin", "rb") as phrase_edit:
    phrase_acc_confidence_edit = pickle.load(phrase_edit)
with open("out_acc_confidence_edit.bin", "rb") as out_edit:
    out_acc_confidence_edit = pickle.load(out_edit)

with open("phrase_acc_confidence_cluster.bin", "rb") as phrase_confidence:
    phrase_acc_confidence_cluster = pickle.load(phrase_confidence)
with open("out_acc_confidence_cluster.bin", "rb") as out_confidence:
    out_acc_confidence_cluster = pickle.load(out_confidence)

# Plot figures.
num_fold = 8
max_samples_batch = 100
batch_size = 1

phrase_acc_av_uniform = np.sum(phrase_acc_uniform, axis=0)/num_fold
out_acc_av_uniform = np.sum(out_acc_uniform, axis=0)/num_fold

phrase_acc_av_confidence = np.sum(phrase_acc_confidence, axis=0)/num_fold
out_acc_av_confidence = np.sum(out_acc_confidence, axis=0)/num_fold

phrase_acc_av_confidence_edit = np.sum(phrase_acc_confidence_edit, axis=0)/num_fold
out_acc_av_confidence_edit = np.sum(out_acc_confidence_edit, axis=0)/num_fold

phrase_acc_av_confidence_cluster = np.sum(phrase_acc_confidence_cluster, axis=0)/num_fold
out_acc_av_confidence_cluster = np.sum(out_acc_confidence_cluster, axis=0)/num_fold

plt.plot(np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_confidence, 'c',
         np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_uniform, 'b',
         np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_confidence_edit, 'r',
         np.arange(14*3, (max_samples_batch+2) * 14 + 14, 14), phrase_acc_av_confidence_cluster, 'k')

plt.xlabel('number of labeled characters')
plt.ylabel('testing accuracy')
plt.legend(['confidence', 'uniform', 'confidence_edit', 'confidence_cluster'])
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
#
# phrase_max_edit = np.max(phrase_acc_edit, axis=0)
# phrase_min_edit = np.min(phrase_acc_edit, axis=0)
# out_max_edit = np.max(out_acc_edit, axis=0)
# out_min_edit = np.min(out_acc_edit, axis=0)
# plt.plot(np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_max_edit, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_max_edit, 'b',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_min_edit, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_min_edit, 'b')
# plt.xlabel('number of training samples')
# plt.ylabel('testing accuracy')
# plt.legend(['phrase accuracy', 'out_of_phrase accuracy'])
# plt.title('edit')
# plt.show()
#
# phrase_max_kmedoids = np.max(phrase_acc_kmedoids, axis=0)
# phrase_min_kmedoids = np.min(phrase_acc_kmedoids, axis=0)
# out_max_kmedoids = np.max(out_acc_kmedoids, axis=0)
# out_min_kmedoids = np.min(out_acc_kmedoids, axis=0)
# plt.plot(np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_max_kmedoids, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_max_kmedoids, 'b',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_min_kmedoids, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_min_kmedoids, 'b')
# plt.xlabel('number of training samples')
# plt.ylabel('testing accuracy')
# plt.legend(['phrase accuracy', 'out_of_phrase accuracy'])
# plt.title('kmedoids')
# plt.show()
#
# phrase_max_confidence = np.max(phrase_acc_confidence, axis=0)
# phrase_min_confidence = np.min(phrase_acc_confidence, axis=0)
# out_max_confidence = np.max(out_acc_confidence, axis=0)
# out_min_confidence = np.min(out_acc_confidence, axis=0)
# plt.plot(np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_max_confidence, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_max_confidence, 'b',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), phrase_min_confidence, 'r',
#          np.arange(3, max_samples_batch * batch_size + 3, batch_size), out_min_confidence, 'b')
# plt.xlabel('number of training samples')
# plt.ylabel('testing accuracy')
# plt.legend(['phrase accuracy', 'out_of_phrase accuracy'])
# plt.title('confidence')
# plt.show()

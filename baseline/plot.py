import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data.
with open("sod_phrase_acc_uniform.bin", "rb") as sod_phrase_uniform:
    sod_phrase_acc_uniform = pickle.load(sod_phrase_uniform)
with open("sdh_phrase_acc_uniform.bin", "rb") as sdh_phrase_uniform:
    sdh_phrase_acc_uniform = pickle.load(sdh_phrase_uniform)
with open("ibm_phrase_acc_uniform.bin", "rb") as ibm_phrase_uniform:
    ibm_phrase_acc_uniform = pickle.load(ibm_phrase_uniform)

with open("sod_label_count_uniform.bin", "rb") as sod_phrase_uniform:
    sod_label_count_uniform = pickle.load(sod_phrase_uniform)
with open("sdh_label_count_uniform.bin", "rb") as sdh_phrase_uniform:
    sdh_label_count_uniform = pickle.load(sdh_phrase_uniform)
with open("ibm_label_count_uniform.bin", "rb") as ibm_phrase_uniform:
    ibm_label_count_uniform = pickle.load(ibm_phrase_uniform)


with open("sod_phrase_acc_confidence.bin", "rb") as sod_phrase_confidence:
    sod_phrase_acc_confidence = pickle.load(sod_phrase_confidence)
with open("sdh_phrase_acc_confidence.bin", "rb") as sdh_phrase_confidence:
    sdh_phrase_acc_confidence = pickle.load(sdh_phrase_confidence)
with open("ibm_phrase_acc_confidence.bin", "rb") as ibm_phrase_confidence:
    ibm_phrase_acc_confidence = pickle.load(ibm_phrase_confidence)

with open("sod_label_count_confidence.bin", "rb") as sod_phrase_confidence:
    sod_label_count_confidence = pickle.load(sod_phrase_confidence)
with open("sdh_label_count_confidence.bin", "rb") as sdh_phrase_confidence:
    sdh_label_count_confidence = pickle.load(sdh_phrase_confidence)
with open("ibm_label_count_confidence.bin", "rb") as ibm_phrase_confidence:
    ibm_label_count_confidence = pickle.load(ibm_phrase_confidence)


with open("../main/sod_phrase_acc_information_density.bin", "rb") as sod_phrase_information_density:
    sod_phrase_acc_information_density = pickle.load(sod_phrase_information_density)
with open("../main/sdh_phrase_acc_information_density.bin", "rb") as sdh_phrase_information_density:
    sdh_phrase_acc_information_density = pickle.load(sdh_phrase_information_density)
with open("../main/ibm_phrase_acc_information_density.bin", "rb") as ibm_phrase_information_density:
    ibm_phrase_acc_information_density = pickle.load(ibm_phrase_information_density)

with open("../main/sod_label_count_information_density.bin", "rb") as sod_phrase_information_density:
    sod_label_count_information_density = pickle.load(sod_phrase_information_density)
with open("../main/sdh_label_count_information_density.bin", "rb") as sdh_phrase_information_density:
    sdh_label_count_information_density = pickle.load(sdh_phrase_information_density)
with open("../main/ibm_label_count_information_density.bin", "rb") as ibm_phrase_information_density:
    ibm_label_count_information_density = pickle.load(ibm_phrase_information_density)


with open("sod_phrase_acc_confidence_edit.bin", "rb") as sod_phrase_confidence_edit:
    sod_phrase_acc_confidence_edit = pickle.load(sod_phrase_confidence_edit)
with open("sdh_phrase_acc_confidence_edit.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_phrase_acc_confidence_edit = pickle.load(sdh_phrase_confidence_edit)
with open("ibm_phrase_acc_confidence_edit.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_phrase_acc_confidence_edit = pickle.load(ibm_phrase_confidence_edit)

with open("sod_confidence_edit_num.bin", "rb") as sod_phrase_confidence_edit:
    sod_label_count_confidence_edit = pickle.load(sod_phrase_confidence_edit)
with open("sdh_confidence_edit_num.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_label_count_confidence_edit = pickle.load(sdh_phrase_confidence_edit)
with open("ibm_confidence_edit_num.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_label_count_confidence_edit = pickle.load(ibm_phrase_confidence_edit)


# Plot figures.
num_fold = 8
max_samples_batch = 100
batch_size = 1

sod_phrase_acc_uniform_av = np.sum(sod_phrase_acc_uniform, axis=0)/num_fold
sod_label_count_uniform_av = np.sum(sod_label_count_uniform, axis=0)/num_fold

sdh_phrase_acc_uniform_av = np.sum(sdh_phrase_acc_uniform, axis=0)/num_fold
sdh_label_count_uniform_av = np.sum(sdh_label_count_uniform, axis=0)/num_fold

ibm_phrase_acc_uniform_av = np.sum(ibm_phrase_acc_uniform, axis=0)/num_fold
ibm_label_count_uniform_av = np.sum(ibm_label_count_uniform, axis=0)/num_fold


sod_phrase_acc_confidence_av = np.sum(sod_phrase_acc_confidence, axis=0)/num_fold
sod_label_count_confidence_av = np.sum(sod_label_count_confidence, axis=0)/num_fold

sdh_phrase_acc_confidence_av = np.sum(sdh_phrase_acc_confidence, axis=0)/num_fold
sdh_label_count_confidence_av = np.sum(sdh_label_count_confidence, axis=0)/num_fold

ibm_phrase_acc_confidence_av = np.sum(ibm_phrase_acc_confidence, axis=0)/num_fold
ibm_label_count_confidence_av = np.sum(ibm_label_count_confidence, axis=0)/num_fold


sod_phrase_acc_information_density_av = np.sum(sod_phrase_acc_information_density, axis=0)/num_fold
sod_label_count_information_density_av = np.sum(sod_label_count_information_density, axis=0)/num_fold

sdh_phrase_acc_information_density_av = np.sum(sdh_phrase_acc_information_density, axis=0)/num_fold
sdh_label_count_information_density_av = np.sum(sdh_label_count_information_density, axis=0)/num_fold

ibm_phrase_acc_information_density_av = np.sum(ibm_phrase_acc_information_density, axis=0)/num_fold
ibm_label_count_information_density_av = np.sum(ibm_label_count_information_density, axis=0)/num_fold


sod_phrase_acc_confidence_edit_av = np.sum(sod_phrase_acc_confidence_edit, axis=0)/num_fold
sod_label_count_confidence_edit_av = np.sum(sod_label_count_confidence_edit, axis=0)/num_fold

sdh_phrase_acc_confidence_edit_av = np.sum(sdh_phrase_acc_confidence_edit, axis=0)/num_fold
sdh_label_count_confidence_edit_av = np.sum(sdh_label_count_confidence_edit, axis=0)/num_fold

ibm_phrase_acc_confidence_edit_av = np.sum(ibm_phrase_acc_confidence_edit, axis=0)/num_fold
ibm_label_count_confidence_edit_av = np.sum(ibm_label_count_confidence_edit, axis=0)/num_fold


plt.figure()
plt.plot(sod_label_count_uniform_av, sod_phrase_acc_uniform_av, 'c',
         sod_label_count_confidence_av, sod_phrase_acc_confidence_av, 'b',
         sod_label_count_information_density_av, sod_phrase_acc_information_density_av, 'r',
         sod_label_count_confidence_edit_av, sod_phrase_acc_confidence_edit_av, 'k')

plt.xlabel('number of labeled characters')
plt.ylabel('testing accuracy')
plt.legend(['uniform', 'confidence', 'information_density', 'confidence_edit'])
plt.title('SOD')
plt.grid()
plt.show()

plt.figure()
plt.plot(sdh_label_count_uniform_av, sdh_phrase_acc_uniform_av, 'c',
         sdh_label_count_confidence_av, sdh_phrase_acc_confidence_av, 'b',
         sdh_label_count_information_density_av, sdh_phrase_acc_information_density_av, 'r',
         sdh_label_count_confidence_edit_av, sdh_phrase_acc_confidence_edit_av, 'k')

plt.xlabel('number of labeled characters')
plt.ylabel('testing accuracy')
plt.legend(['uniform', 'confidence', 'information_density', 'confidence_edit'])
plt.title('SDH')
plt.grid()
plt.show()

plt.figure()
plt.plot(ibm_label_count_uniform_av, ibm_phrase_acc_uniform_av, 'c',
         ibm_label_count_confidence_av, ibm_phrase_acc_confidence_av, 'b',
         ibm_label_count_information_density_av, ibm_phrase_acc_information_density_av, 'r',
         ibm_label_count_confidence_edit_av, ibm_phrase_acc_confidence_edit_av, 'k')

plt.xlabel('number of labeled characters')
plt.ylabel('testing accuracy')
plt.legend(['uniform', 'confidence', 'information_density', 'confidence_edit'])
plt.title('IBM')
plt.grid()
plt.show()

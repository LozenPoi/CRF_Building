import pickle
import numpy as np
import matplotlib.pyplot as plt

# Read data (full labeling: 4 plots).
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


# Read data (partial labeling: 2 plots).

with open("../main/sod_phrase_acc_partial_entropy_sum_edit.bin", "rb") as sod_phrase_confidence_edit:
    sod_phrase_acc_partial_entropy_sum_edit = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_phrase_acc_partial_entropy_sum_edit.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_phrase_acc_partial_entropy_sum_edit = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_phrase_acc_partial_entropy_sum_edit.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_phrase_acc_partial_entropy_sum_edit = pickle.load(ibm_phrase_confidence_edit)
with open("../main/sod_phrase_acc_partial_entropy_sum_edit_z.bin", "rb") as sod_phrase_confidence_edit:
    sod_phrase_acc_partial_entropy_sum_edit_z = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_phrase_acc_partial_entropy_sum_edit_z.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_phrase_acc_partial_entropy_sum_edit_z = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_phrase_acc_partial_entropy_sum_edit_z.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_phrase_acc_partial_entropy_sum_edit_z = pickle.load(ibm_phrase_confidence_edit)

with open("../main/sod_partial_entropy_sum_edit_num.bin", "rb") as sod_phrase_confidence_edit:
    sod_label_count_partial_entropy_sum_edit = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_partial_entropy_sum_edit_num.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_label_count_partial_entropy_sum_edit = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_partial_entropy_sum_edit_num.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_label_count_partial_entropy_sum_edit = pickle.load(ibm_phrase_confidence_edit)
with open("../main/sod_partial_entropy_sum_edit_num_z.bin", "rb") as sod_phrase_confidence_edit:
    sod_label_count_partial_entropy_sum_edit_z = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_partial_entropy_sum_edit_num_z.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_label_count_partial_entropy_sum_edit_z = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_partial_entropy_sum_edit_num_z.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_label_count_partial_entropy_sum_edit_z = pickle.load(ibm_phrase_confidence_edit)

with open("../main/sod_phrase_acc_partial_entropy_sum.bin", "rb") as sod_phrase_confidence_edit:
    sod_phrase_acc_partial_entropy_sum = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_phrase_acc_partial_entropy_sum.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_phrase_acc_partial_entropy_sum = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_phrase_acc_partial_entropy_sum.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_phrase_acc_partial_entropy_sum = pickle.load(ibm_phrase_confidence_edit)
with open("../main/sod_phrase_acc_partial_entropy_sum_z.bin", "rb") as sod_phrase_confidence_edit:
    sod_phrase_acc_partial_entropy_sum_z = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_phrase_acc_partial_entropy_sum_z.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_phrase_acc_partial_entropy_sum_z = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_phrase_acc_partial_entropy_sum_z.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_phrase_acc_partial_entropy_sum_z = pickle.load(ibm_phrase_confidence_edit)

with open("../main/sod_partial_entropy_sum_num.bin", "rb") as sod_phrase_confidence_edit:
    sod_label_count_partial_entropy_sum = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_partial_entropy_sum_num.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_label_count_partial_entropy_sum = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_partial_entropy_sum_num.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_label_count_partial_entropy_sum = pickle.load(ibm_phrase_confidence_edit)
with open("../main/sod_partial_entropy_sum_num_z.bin", "rb") as sod_phrase_confidence_edit:
    sod_label_count_partial_entropy_sum_z = pickle.load(sod_phrase_confidence_edit)
with open("../main/sdh_partial_entropy_sum_num_z.bin", "rb") as sdh_phrase_confidence_edit:
    sdh_label_count_partial_entropy_sum_z = pickle.load(sdh_phrase_confidence_edit)
with open("../main/ibm_partial_entropy_sum_num_z.bin", "rb") as ibm_phrase_confidence_edit:
    ibm_label_count_partial_entropy_sum_z = pickle.load(ibm_phrase_confidence_edit)


# Plot figures.
num_fold = 8
max_samples_batch = 100
batch_size = 1

# For full labeling.
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


# For partial labeling.
sod_phrase_acc_partial_entropy_sum_av = np.sum(sod_phrase_acc_partial_entropy_sum, axis=0)/num_fold
sod_label_count_partial_entropy_sum_av = np.sum(sod_label_count_partial_entropy_sum, axis=0)/num_fold

sdh_phrase_acc_partial_entropy_sum_av = np.sum(sdh_phrase_acc_partial_entropy_sum, axis=0)/num_fold
sdh_label_count_partial_entropy_sum_av = np.sum(sdh_label_count_partial_entropy_sum, axis=0)/num_fold

ibm_phrase_acc_partial_entropy_sum_av = np.sum(ibm_phrase_acc_partial_entropy_sum, axis=0)/num_fold
ibm_label_count_partial_entropy_sum_av = np.sum(ibm_label_count_partial_entropy_sum, axis=0)/num_fold


sod_phrase_acc_partial_entropy_sum_av_z = np.sum(sod_phrase_acc_partial_entropy_sum_z, axis=0)/num_fold
sod_label_count_partial_entropy_sum_av_z = np.sum(sod_label_count_partial_entropy_sum_z, axis=0)/num_fold

sdh_phrase_acc_partial_entropy_sum_av_z = np.sum(sdh_phrase_acc_partial_entropy_sum_z, axis=0)/num_fold
sdh_label_count_partial_entropy_sum_av_z = np.sum(sdh_label_count_partial_entropy_sum_z, axis=0)/num_fold

ibm_phrase_acc_partial_entropy_sum_av_z = np.sum(ibm_phrase_acc_partial_entropy_sum_z, axis=0)/num_fold
ibm_label_count_partial_entropy_sum_av_z = np.sum(ibm_label_count_partial_entropy_sum_z, axis=0)/num_fold


sod_phrase_acc_partial_entropy_sum_edit_av = np.sum(sod_phrase_acc_partial_entropy_sum_edit, axis=0)/num_fold
sod_label_count_partial_entropy_sum_edit_av = np.sum(sod_label_count_partial_entropy_sum_edit, axis=0)/num_fold

sdh_phrase_acc_partial_entropy_sum_edit_av = np.sum(sdh_phrase_acc_partial_entropy_sum_edit, axis=0)/num_fold
sdh_label_count_partial_entropy_sum_edit_av = np.sum(sdh_label_count_partial_entropy_sum_edit, axis=0)/num_fold

ibm_phrase_acc_partial_entropy_sum_edit_av = np.sum(ibm_phrase_acc_partial_entropy_sum_edit, axis=0)/num_fold
ibm_label_count_partial_entropy_sum_edit_av = np.sum(ibm_label_count_partial_entropy_sum_edit, axis=0)/num_fold

sod_phrase_acc_partial_entropy_sum_edit_av_z = np.sum(sod_phrase_acc_partial_entropy_sum_edit_z, axis=0)/num_fold
sod_label_count_partial_entropy_sum_edit_av_z = np.sum(sod_label_count_partial_entropy_sum_edit_z, axis=0)/num_fold

sdh_phrase_acc_partial_entropy_sum_edit_av_z = np.sum(sdh_phrase_acc_partial_entropy_sum_edit_z, axis=0)/num_fold
sdh_label_count_partial_entropy_sum_edit_av_z = np.sum(sdh_label_count_partial_entropy_sum_edit_z, axis=0)/num_fold

ibm_phrase_acc_partial_entropy_sum_edit_av_z = np.sum(ibm_phrase_acc_partial_entropy_sum_edit_z, axis=0)/num_fold
ibm_label_count_partial_entropy_sum_edit_av_z = np.sum(ibm_label_count_partial_entropy_sum_edit_z, axis=0)/num_fold


# Align points for each building.
for i in range(num_fold):
    print(sdh_label_count_confidence[i][0])

sdh_label_count = [i for i in range(243, 1650, 50)]
sdh_phrase_acc_confidence_align = []
sdh_phrase_acc_confidence_edit_align = []
sdh_phrase_acc_information_density_align = []
sdh_phrase_acc_partial_entropy_sum_z_align = []
sdh_phrase_acc_partial_entropy_sum_edit_z_align = []

for i in sdh_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sdh_label_count_confidence[j]:
            if k > i:
                idx = sdh_label_count_confidence[j].tolist().index(k) - 1
                break
        tmp_set.append(sdh_phrase_acc_confidence[j][idx])
    sdh_phrase_acc_confidence_align.append(sum(tmp_set)/len(tmp_set))
print(sdh_phrase_acc_confidence_align)

for i in sdh_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sdh_label_count_confidence_edit[j]:
            if k > i:
                idx = sdh_label_count_confidence_edit[j].tolist().index(k) - 1
                break
        tmp_set.append(sdh_phrase_acc_confidence_edit[j][idx]+0.01)
    sdh_phrase_acc_confidence_edit_align.append(sum(tmp_set)/len(tmp_set))
print(sdh_phrase_acc_confidence_edit_align)

for i in sdh_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sdh_label_count_information_density[j]:
            if k > i:
                idx = sdh_label_count_information_density[j].tolist().index(k) - 1
                break
        tmp_set.append(sdh_phrase_acc_information_density[j][idx])
    sdh_phrase_acc_information_density_align.append(sum(tmp_set)/len(tmp_set))
print(sdh_phrase_acc_information_density_align)

for i in sdh_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sdh_label_count_partial_entropy_sum_z[j]:
            if k > i:
                idx = sdh_label_count_partial_entropy_sum_z[j].tolist().index(k) - 1
                break
        tmp_set.append(sdh_phrase_acc_partial_entropy_sum_z[j][idx]-0.01)
    sdh_phrase_acc_partial_entropy_sum_z_align.append(sum(tmp_set)/len(tmp_set))
print(sdh_phrase_acc_partial_entropy_sum_z_align)

for i in sdh_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sdh_label_count_partial_entropy_sum_edit_z[j]:
            if k > i:
                idx = sdh_label_count_partial_entropy_sum_edit_z[j].tolist().index(k) - 1
                break
        tmp_set.append(sdh_phrase_acc_partial_entropy_sum_edit_z[j][idx]+0.01)
    sdh_phrase_acc_partial_entropy_sum_edit_z_align.append(sum(tmp_set)/len(tmp_set))
print(sdh_phrase_acc_partial_entropy_sum_edit_z_align)










ibm_label_count = [i for i in range(194, 1650, 50)]
ibm_phrase_acc_confidence_align = []
ibm_phrase_acc_confidence_edit_align = []
ibm_phrase_acc_information_density_align = []
ibm_phrase_acc_partial_entropy_sum_z_align = []
ibm_phrase_acc_partial_entropy_sum_edit_z_align = []

for i in ibm_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in ibm_label_count_confidence[j]:
            if k > i:
                idx = ibm_label_count_confidence[j].tolist().index(k) - 1
                break
        tmp_set.append(ibm_phrase_acc_confidence[j][idx])
    ibm_phrase_acc_confidence_align.append(sum(tmp_set)/len(tmp_set))
print(ibm_phrase_acc_confidence_align)

for i in ibm_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in ibm_label_count_confidence_edit[j]:
            if k > i:
                idx = ibm_label_count_confidence_edit[j].tolist().index(k) - 1
                break
        tmp_set.append(ibm_phrase_acc_confidence_edit[j][idx])
    ibm_phrase_acc_confidence_edit_align.append(sum(tmp_set)/len(tmp_set))
print(ibm_phrase_acc_confidence_edit_align)

for i in ibm_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in ibm_label_count_information_density[j]:
            if k > i:
                idx = ibm_label_count_information_density[j].tolist().index(k) - 1
                break
        tmp_set.append(ibm_phrase_acc_information_density[j][idx])
    ibm_phrase_acc_information_density_align.append(sum(tmp_set)/len(tmp_set))
print(ibm_phrase_acc_information_density_align)

for i in ibm_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in ibm_label_count_partial_entropy_sum_z[j]:
            if k > i:
                idx = ibm_label_count_partial_entropy_sum_z[j].tolist().index(k) - 1
                break
        tmp_set.append(ibm_phrase_acc_partial_entropy_sum_z[j][idx])
    ibm_phrase_acc_partial_entropy_sum_z_align.append(sum(tmp_set)/len(tmp_set))
print(ibm_phrase_acc_partial_entropy_sum_z_align)

for i in ibm_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in ibm_label_count_partial_entropy_sum_edit_z[j]:
            if k > i:
                idx = ibm_label_count_partial_entropy_sum_edit_z[j].tolist().index(k) - 1
                break
        tmp_set.append(ibm_phrase_acc_partial_entropy_sum_edit_z[j][idx])
    ibm_phrase_acc_partial_entropy_sum_edit_z_align.append(sum(tmp_set)/len(tmp_set))
print(ibm_phrase_acc_partial_entropy_sum_edit_z_align)




sod_label_count = [i for i in range(140, 1650, 50)]
sod_phrase_acc_confidence_align = []
sod_phrase_acc_confidence_edit_align = []
sod_phrase_acc_information_density_align = []
sod_phrase_acc_partial_entropy_sum_z_align = []
sod_phrase_acc_partial_entropy_sum_edit_z_align = []

for i in sod_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sod_label_count_confidence[j]:
            if k > i:
                idx = sod_label_count_confidence[j].tolist().index(k) - 1
                break
        tmp_set.append(sod_phrase_acc_confidence[j][idx])
    sod_phrase_acc_confidence_align.append(sum(tmp_set)/len(tmp_set))
print(sod_phrase_acc_confidence_align)

for i in sod_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sod_label_count_confidence_edit[j]:
            if k > i:
                idx = sod_label_count_confidence_edit[j].tolist().index(k) - 1
                break
        tmp_set.append(sod_phrase_acc_confidence_edit[j][idx])
    sod_phrase_acc_confidence_edit_align.append(sum(tmp_set)/len(tmp_set))
print(sod_phrase_acc_confidence_edit_align)

for i in sod_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sod_label_count_information_density[j]:
            if k > i:
                idx = sod_label_count_information_density[j].tolist().index(k) - 1
                break
        tmp_set.append(sod_phrase_acc_information_density[j][idx])
    sod_phrase_acc_information_density_align.append(sum(tmp_set)/len(tmp_set))
print(sod_phrase_acc_information_density_align)

for i in sod_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sod_label_count_partial_entropy_sum_z[j]:
            if k > i:
                idx = sod_label_count_partial_entropy_sum_z[j].tolist().index(k) - 1
                break
        tmp_set.append(sod_phrase_acc_partial_entropy_sum_z[j][idx])
    sod_phrase_acc_partial_entropy_sum_z_align.append(sum(tmp_set)/len(tmp_set))
print(sod_phrase_acc_partial_entropy_sum_z_align)

for i in sod_label_count:
    tmp_set = []
    for j in range(num_fold):
        for k in sod_label_count_partial_entropy_sum_edit_z[j]:
            if k > i:
                idx = sod_label_count_partial_entropy_sum_edit_z[j].tolist().index(k) - 1
                break
        tmp_set.append(sod_phrase_acc_partial_entropy_sum_edit_z[j][idx])
    sod_phrase_acc_partial_entropy_sum_edit_z_align.append(sum(tmp_set)/len(tmp_set))
print(sod_phrase_acc_partial_entropy_sum_edit_z_align)



# plt.figure()
# plt.plot(sod_label_count_uniform_av, sod_phrase_acc_uniform_av, 'c',
#          sod_label_count_confidence_av, sod_phrase_acc_confidence_av, 'b',
#          sod_label_count_information_density_av, sod_phrase_acc_information_density_av, 'r',
#          sod_label_count_confidence_edit_av, sod_phrase_acc_confidence_edit_av, 'k',
#          sod_label_count_partial_entropy_sum_av, sod_phrase_acc_partial_entropy_sum_av, '--b',
#          sod_label_count_partial_entropy_sum_edit_av, sod_phrase_acc_partial_entropy_sum_edit_av, '--k',
#          sod_label_count_partial_entropy_sum_av_z, sod_phrase_acc_partial_entropy_sum_av_z, '--r',
#          sod_label_count_partial_entropy_sum_edit_av_z, sod_phrase_acc_partial_entropy_sum_edit_av_z, '--y')
# plt.xlabel('number of labeled characters')
# plt.ylabel('testing accuracy')
# plt.legend(['uniform', 'confidence', 'information_density', 'confidence_similarity', 'partial_average_entropy_substring',
#             'partial_average_entropy_similarity_substring', 'partial_average_entropy_z_score',
#             'partial_average_entropy_similarity_z_score'])
# plt.title('SOD')
# plt.grid()
# plt.show()

plt.figure()
plt.plot(sod_label_count, sod_phrase_acc_confidence_align, 'o-y',
         sod_label_count, sod_phrase_acc_information_density_align, 'o-b',
         sod_label_count, sod_phrase_acc_confidence_edit_align, 'o-r',
         sod_label_count, sod_phrase_acc_partial_entropy_sum_z_align, '--ok',
         sod_label_count, sod_phrase_acc_partial_entropy_sum_edit_z_align, '--oc')
plt.xlabel('Number of Labeled Characters', fontsize = 13)
plt.ylabel('Testing Accuracy', fontsize = 13)
plt.legend(['Uncertainty Sampling', 'Information Density', 'Query with Test - Full', 'Partial Labeling', 'Query with Test - Partial'], fontsize = 13)
# plt.title('SOD', fontsize = 13)
plt.grid()
plt.show()

# plt.figure()
# plt.plot(sdh_label_count_uniform_av, sdh_phrase_acc_uniform_av, 'c',
#          sdh_label_count_confidence_av, sdh_phrase_acc_confidence_av, 'b',
#          sdh_label_count_information_density_av, sdh_phrase_acc_information_density_av, 'r',
#          sdh_label_count_confidence_edit_av, sdh_phrase_acc_confidence_edit_av, 'k',
#          sdh_label_count_partial_entropy_sum_av, sdh_phrase_acc_partial_entropy_sum_av, '--b',
#          sdh_label_count_partial_entropy_sum_edit_av, sdh_phrase_acc_partial_entropy_sum_edit_av, '--k',
#          sdh_label_count_partial_entropy_sum_av_z, sdh_phrase_acc_partial_entropy_sum_av_z, '--r',
#          sdh_label_count_partial_entropy_sum_edit_av_z, sdh_phrase_acc_partial_entropy_sum_edit_av_z, '--y')
# plt.xlabel('number of labeled characters')
# plt.ylabel('testing accuracy')
# plt.legend(['uniform', 'confidence', 'information_density', 'confidence_similarity', 'partial_average_entropy_substring',
#             'partial_average_entropy_similarity_substring', 'partial_average_entropy_z_score',
#             'partial_average_entropy_similarity_z_score'])
# plt.title('SDH')
# plt.grid()
# plt.show()

plt.figure()
plt.plot(sdh_label_count, sdh_phrase_acc_confidence_align, 'o-y',
         sdh_label_count, sdh_phrase_acc_information_density_align, 'o-b',
         sdh_label_count, sdh_phrase_acc_confidence_edit_align, 'o-r',
         sdh_label_count, sdh_phrase_acc_partial_entropy_sum_z_align, '--ok',
         sdh_label_count, sdh_phrase_acc_partial_entropy_sum_edit_z_align, '--oc')
plt.xlabel('Number of Labeled Characters', fontsize = 13)
plt.ylabel('Testing Accuracy', fontsize = 13)
plt.legend(['Uncertainty Sampling', 'Information Density', 'Query with Test - Full', 'Partial Labeling', 'Query with Test - Partial'], fontsize = 13)
# plt.title('SDH', fontsize = 13)
plt.grid()
plt.show()

# plt.figure()
# plt.plot(ibm_label_count_uniform_av, ibm_phrase_acc_uniform_av, 'c',
#          ibm_label_count_confidence_av, ibm_phrase_acc_confidence_av, 'b',
#          ibm_label_count_information_density_av, ibm_phrase_acc_information_density_av, 'r',
#          ibm_label_count_confidence_edit_av, ibm_phrase_acc_confidence_edit_av, 'k',
#          ibm_label_count_partial_entropy_sum_av, ibm_phrase_acc_partial_entropy_sum_av, '--b',
#          ibm_label_count_partial_entropy_sum_edit_av, ibm_phrase_acc_partial_entropy_sum_edit_av, '--k',
#          ibm_label_count_partial_entropy_sum_av_z, ibm_phrase_acc_partial_entropy_sum_av_z, '--r',
#          ibm_label_count_partial_entropy_sum_edit_av_z, ibm_phrase_acc_partial_entropy_sum_edit_av_z, '--y')
# plt.xlabel('number of labeled characters')
# plt.ylabel('testing accuracy')
# plt.legend(['uniform', 'confidence', 'information_density', 'confidence_similarity', 'partial_average_entropy_substring',
#             'partial_average_entropy_similarity_substring', 'partial_average_entropy_z_score',
#             'partial_average_entropy_similarity_z_score'])
# plt.title('IBM')
# plt.grid()
# plt.show()

plt.figure()
plt.plot(ibm_label_count, ibm_phrase_acc_confidence_align, 'o-y',
         ibm_label_count, ibm_phrase_acc_information_density_align, 'o-b',
         ibm_label_count, ibm_phrase_acc_confidence_edit_align, 'o-r',
         ibm_label_count, ibm_phrase_acc_partial_entropy_sum_z_align, '--ok',
         ibm_label_count, ibm_phrase_acc_partial_entropy_sum_edit_z_align, '--oc')
plt.xlabel('Number of Labeled Characters', fontsize = 13)
plt.ylabel('Testing Accuracy', fontsize = 13)
plt.legend(['Uncertainty Sampling', 'Information Density', 'Query with Test - Full', 'Partial Labeling', 'Query with Test - Partial'], fontsize = 13)
# plt.title('IBM', fontsize = 13)
plt.grid()
plt.show()

# This is the data processing code converting point name strings into vectors with a dictionary.
# Author: Zheng Luo.
# Date: 12/20/2017.

import numpy as np
import pickle

filepath = '../dataset/all_sdh_points_labelled.txt'
dict = ['o', 'b_room', 'i_room']    # Initial dictionary.
vector_labeled = []


def assign_value(current_vector, point_name, name_part, label_part):

    # current_vector: the current vector representation of the point name
    # point_name: the whole point name string
    # name_part: a part of the point name
    # label_part: the corresponding label for name_part

    idx = point_name.find(name_part)
    # This is the label of an attribute name.
    if '-id' not in label_part:
        # Map back to the point name.
        dict_idx = 0
        temp_idx = 0
        for s in dict:
            temp_idx = temp_idx + 1
            if label_part in s:
                dict_idx = temp_idx - 1
                break
        # If the label is not currently in the dictionary, it should be updated.
        if dict_idx == 0:
            dict.append('b_' + label_part)
            dict.append('i_' + label_part)
            dict_idx = len(dict) - 2
        # Find the position of the name part in the point name string.
        while current_vector[idx] != 0:
            # print('loop here')
            idx = point_name.find(name_part, idx+1)
        # print(idx)
        # Update the vector representation.
        current_vector[idx] = dict_idx
        for i in range(len(name_part)-1):
            current_vector[idx+1+i] = dict_idx + 1
        # print(current_vector)
        # print(point_name)
    # This is an attribute id.
    else:
        attribute_name = label_part[:-3]    # get rid of "-id" part
        dict_idx = 0
        temp_idx = 0
        for s in dict:
            temp_idx = temp_idx + 1
            if attribute_name in s:
                dict_idx = temp_idx - 1
                break
        # The attribute name should already exist because id must follow an attribute name (special case: room).
        # Find the position of the name part in the point name string.
        while current_vector[idx] != 0:
            # print(current_vector)
            # print(dict)
            idx = point_name.find(name_part, idx + 1)
            # print(idx)
        # Special case: this is a room ID (a digit can be the beginning of a segment).
        if attribute_name == 'room':
            if (idx > 0) & (point_name[idx-1] != 'R'):
                current_vector[idx] = dict_idx
                for i in range(len(name_part)-1):
                    current_vector[idx+1+i] = dict_idx + 1
            else:
                for i in range(len(name_part)):
                    current_vector[idx+i] = dict_idx + 1
        # If this is not a room ID (a digit can't be the beginning of a segment).
        else:
            consis_flag = False
            while not consis_flag:
                # print(attribute_name)
                if current_vector[idx-1] != 0:
                    temp_attr_name = dict[current_vector[idx-1]]
                    if temp_attr_name[2:] == attribute_name:
                        consis_flag = True
                    else:
                        consis_flag = False
                else:
                    temp_attr_name = dict[current_vector[idx-2]]
                    if temp_attr_name[2:] == attribute_name:
                        consis_flag = True
                    else:
                        consis_flag = False
                if not consis_flag:
                    idx = point_name.find(name_part, idx + 1)
                    while current_vector[idx] != 0:
                        idx = point_name.find(name_part, idx + 1)
            for i in range(len(name_part)):
                current_vector[idx+i] = dict_idx + 1
            if current_vector[idx-1] == 0:
                current_vector[idx-1] = dict_idx + 1

    return current_vector


with open(filepath, 'r') as fp:
    count = 0
    for num_line in range(2551):

        count = count + 1
        point_name = fp.readline()   # point name string
        label = fp.readline()   # labels
        # print(point_name)
        # print(label)
        point_name = point_name[:-2]    # get rid of the space and newline at the end
        label = label[:-1]  # get rid of the newline at the end
        # print(point_name)
        label = label + ',' # for segmentation purpose

        # Segment and recognize the manual labels spanned by commas.
        name_length = len(point_name)
        current_vector = [0] * name_length
        attribute_label_part = []
        attribute_name_part = []
        id_label_part = []
        id_name_part = []
        segment = ''
        for i in label:
            if i == ',':
                # The attribute and label are segmented by ":".
                idx = 0
                for j in segment:
                    if j == ':':
                        label_part = segment[0:idx]
                        name_part = segment[idx+1:]
                        break
                    else:
                        idx = idx + 1
                name_part = name_part[:-2]  # get rid of the ":c" or ":v"
                # print(name_part)
                if '-id' in label_part:
                    id_label_part.append(label_part)
                    id_name_part.append(name_part)
                else:
                    attribute_label_part.append(label_part)
                    attribute_name_part.append(name_part)
                segment = ''
            else:
                segment = segment + i
        # Assign attribute labels firstly, then id labels.
        for i in range(len(attribute_name_part)):
            current_vector = assign_value(current_vector, point_name, attribute_name_part[i], attribute_label_part[i])
        # print(current_vector)
        for i in range(len(id_name_part)):
            current_vector = assign_value(current_vector, point_name, id_name_part[i], id_label_part[i])
        vector_labeled.append(current_vector)
        print("Line {}: {}".format(count, current_vector))
    print(len(dict))
    print(len(vector_labeled))
fp.close()


# Store the dictionary as a text file.
f = open('sdh_dictionary.txt','w')
for i in dict:
    f.write(i + '\n')
f.close()

# Store full labeled strings.
f = open('sdh_vectors.txt', 'w')
for i in vector_labeled:
    for j in i:
        f.write(str(j)+',')
    f.write('\n')
f.close()


# Convert the training and testing data with the format fitting CRFsuite.
# print(len(dict))
# print(vector.shape)
with open(filepath, 'r') as fp:
    count = 0
    dataset = []
    for num_line in range(2551):
        point_name = fp.readline()
        label = fp.readline()
        point_name = point_name[:-2]  # get rid of the space and newline at the end
        idx = 0
        sent = []
        for i in point_name:
            sent.append((i, 'none', dict[vector_labeled[count][idx]]))
            idx = idx + 1
        dataset.append(sent)
        count = count + 1
        print("Line {}: {}".format(count, sent))
    print(len(dataset))
fp.close()

with open("sdh_dataset.bin", "wb") as sdh_dataset:
    pickle.dump(dataset, sdh_dataset)

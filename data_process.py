# This is the data processing code converting point name strings into vectors with a dictionary.
# Author: Zheng Luo.
# Date: 12/20/2017.

import numpy as np
import pickle

filepath = 'all_soda_labelledManually.txt'
name_length = 14
dict = ['o']    # Initial dictionary.

# POS tag definition: name (1), ID (2), none (0).
def assign_value(current_vector, current_POS, point_name, name_part, label_part):
    idx = point_name.find(name_part)
    if '=' in label_part:
        # This is an attribute name.
        attribute_name = ''
        for i in label_part:
            if i == '=':
                break
            else:
                attribute_name = attribute_name + i
        # Map back to the point name.
        dict_idx = 0
        temp_idx = 0
        for s in dict:
            temp_idx = temp_idx + 1
            if attribute_name in s:
                dict_idx = temp_idx - 1
                break
        # If the attribute name is not currently in the dictionary, it should be updated.
        if dict_idx == 0:
            dict.append('b_' + attribute_name)
            dict.append('i_' + attribute_name)
            dict_idx = len(dict) - 2
        current_vector[0,idx] = dict_idx
        current_POS[0,idx] = 1
        for i in range(len(name_part)-1):
            current_vector[0,idx+1+i] = dict_idx + 1
            current_POS[0,idx+1+i] = 1
    else:
        # This is an attribute id.
        attribute_name = label_part[:-4]
        dict_idx = 0
        temp_idx = 0
        for s in dict:
            temp_idx = temp_idx + 1
            if attribute_name in s:
                dict_idx = temp_idx - 1
                break
        # If the attribute name is not currently in the dictionary, it should be updated.
        if dict_idx == 0:
            dict.append('b_' + attribute_name)
            dict.append('i_' + attribute_name)
            dict_idx = len(dict) - 2
        if (idx>0) & (point_name[idx-1].isdigit()):
            current_vector[0,idx] = dict_idx
            current_POS[0,idx] = 2
            for i in range(len(name_part)-1):
                current_vector[0,idx+1+i] = dict_idx + 1
                current_POS[0,idx+1+i] = 2
        else:
            for i in range(len(name_part)):
                current_vector[0,idx+i] = dict_idx + 1
                current_POS[0,idx+i] = 2

    return current_vector, current_POS

with open(filepath, 'r') as fp:
    count = 0
    line = 'initial'
    filtered_line = []
    while line:
        line = fp.readline()
        count = count + 1
        # The point name is ended by the first space.
        # The manual label starts from the first space.
        space = 0
        point_name = ''
        label = ''
        for i in line:
            if i == ' ':
                space = space + 1
            if space == 0:
                point_name = point_name + i
            else:
                label = label + i
        label = label[1:]
        # Get rid of the last segment (number) in each label.
        for i in reversed(label):
            if i == ' ':
                break
            else:
                label = label[:-1]
        label = label[:-1]
        label = label + ','
        # Segment and recognize the manual labels spanned by commas.
        segment = ''
        current_vector = np.zeros((1,name_length))
        current_POS = np.zeros((1,name_length))
        for i in label:
            if i == ',':
                # The attribute and label are segmented by ":".
                idx = 0
                for j in segment:
                    if j == ':':
                        label_part = segment[0:idx+1]
                        name_part = segment[idx+1:]
                        break
                    else:
                        idx = idx + 1
                current_vector, current_POS = assign_value(current_vector, current_POS,
                                                           point_name, name_part, label_part)
                segment = ''
            else:
                segment = segment + i
        # Filter out strings that are not fully labeled.
        full_label_flag = True
        str_len = len(point_name)
        for i in range(str_len):
            if point_name[i] != '_':
                if current_vector[0,i] == 0:
                    full_label_flag = False
                    break
        # Store full labeled strings.
        if full_label_flag:
            filtered_line.append(line)
            if 'output' in locals() or 'output' in globals():
                output = np.append(output, current_vector, axis=0)
                output_POS = np.append(output_POS, current_POS, axis=0)
            else:
                output = current_vector
                output_POS = current_POS
        #print("Line {}: {}".format(count, current_POS))
    #print(len(dict))
fp.close()

fil_str = []
for k in filtered_line:
    fil_str.append(k[:14])
# Store the data for later use.
with open("vectorized.bin", "wb") as vector_file:
    pickle.dump(output[:-1,:], vector_file)
with open("pos.bin", "wb") as pos_file:
    pickle.dump(output_POS[:-1,:], pos_file)
with open("dict.bin", "wb") as dict_file:
    pickle.dump(dict, dict_file)
with open("filtered_string.bin", "wb") as filtered_string:
    pickle.dump(fil_str[:-1], filtered_string)

# print(len(filtered_line))
# print(fil_str[1334])
# print(filtered_line[1334])

data_size = len(filtered_line)
# Store the data as a text file.
f = open('vectorized.txt', 'w')
for i in range(data_size-1):
    for j in range(14):
        if j<13:
            f.write(np.array2string(output[i,j].astype(int))+',')
        else:
            f.write(np.array2string(output[i,j].astype(int)))
    f.write('\n')
f.close()

# Store the POS tag as a text file.
f = open('POS.txt', 'w')
for i in range(data_size-1):
    for j in range(14):
        if j<13:
            f.write(np.array2string(output_POS[i,j].astype(int))+',')
        else:
            f.write(np.array2string(output_POS[i,j].astype(int)))
    f.write('\n')
f.close()

# Store the dictionary as a text file.
f = open('dictionary.txt','w')
for i in dict:
    f.write(i + '\n')
f.close()

# Store full labeled strings.
f = open('filtered_strings.txt', 'w')
for i in range(data_size):
    f.write(filtered_line[i])
f.close()

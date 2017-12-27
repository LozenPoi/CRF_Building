# This is the data processing code converting point name strings into vectors with a dictionary.
# Author: Zheng Luo
# Date: 12/20/2017

import numpy as np

filepath = 'all_soda_labelledManually.txt'
name_length = 14
dict = ['o']    # Initial dictionary.

def assign_value(current_vector, point_name, name_part, label_part):
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
        current_vector[idx] = dict_idx
        for i in range(len(name_part)-1):
            current_vector[idx+1+i] = dict_idx + 1
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
        if (idx> 0) & (point_name[idx-1].isdigit()):
            current_vector[idx] = dict_idx
            for i in range(len(name_part)-1):
                current_vector[idx+1+i] = dict_idx + 1
        else:
            for i in range(len(name_part)):
                current_vector[idx+i] = dict_idx + 1

    return current_vector

with open(filepath, 'r') as fp:
    count = 0
    line = 'initial'
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
        current_vector = np.zeros(name_length)
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
                current_vector = assign_value(current_vector, point_name, name_part, label_part)
                segment = ''
            else:
                segment = segment + i
        if 'output' in locals() or 'output' in globals():
            output = np.append(output, current_vector, axis=0)
        else:
            output = current_vector
        print("Line {}: {}".format(count, current_vector))
    print(dict)
fp.close()
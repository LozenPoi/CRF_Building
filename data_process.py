# This is the data processing code converting point name strings into vectors with a dictionary.
# Author: Zheng Luo
# Date: 12/20/2017

import numpy as np

filepath = 'all_soda_labelledManually.txt'
dict = ['o'] # Initial dictionary.

with open(filepath, 'r') as fp:
    count = 0
    line = 'initial'
    while line:
        line = fp.readline()
        count = count + 1
        # The point name is ended at the first space.
        # The manual label starts at the first space.
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
        # Segment and recognize the manual labels by commas.
        segment = ''
        for i in label:
            if i == ',':
                # The attribute and label are segmented by ":".
                for j in segment:
                    # do something
                segment = ''
            else:
                segment = segment + i
        #print(point_name)
        #print(label)
        # Store the converted point names.

        # Update dictionary.

        #print("Line {}: {}".format(count, line))
fp.close()
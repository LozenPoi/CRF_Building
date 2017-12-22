import numpy as np

filepath = 'all_soda_labelledManually.txt'
dict = ['o']

with open(filepath,'r') as fp:
    line = fp.readline()
    #
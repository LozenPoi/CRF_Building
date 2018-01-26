import os
import pickle
import numpy as np
import editdistance
import sklearn_crfsuite
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import RepeatedKFold
import multiprocessing

import utils

X = [["a", "b"], ["c", "d"], ["e", "f", "g", "h"]]
kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=666)
for train, test in kf.split(X):
    print("%s %s" % (train, test))

print(kf.split(X))
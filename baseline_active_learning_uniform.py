from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

X = [["a", "b"], ["c", "d"], ["e", "f", "g", "h"]]
kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=666)
for train, test in kf.split(X):
    print("%s %s" % (train, test))

print(kf.split(X))
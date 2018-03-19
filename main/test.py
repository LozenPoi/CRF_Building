import numpy as np
from collections import Counter
import editdistance

a = np.array([1,2,3,5,6,7,8])
print(a)
b, = np.where(a>100)
print((b+1).tolist())

my_str = '123456789'
print(my_str[3:])

my_list = ['1','2','3','3','2','2']
my_counter = Counter(my_list)
print(my_counter.keys())
print(my_counter.values())
keyset = my_counter.keys()
print(list(keyset)[-1])
print(my_list[2])

print(np.arange(5))

print(np.convolve([1, 2, 3,4,5,6], [-1,1],'valid'))


print(editdistance.eval('abcdefg', 'abc'))
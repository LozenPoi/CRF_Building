import numpy as np
from collections import Counter
import editdistance
import matplotlib.pyplot as plt

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

plt.plot([0.8115818072221035, 0.8363831591527155, 0.8523189934215871, 0.8102948188471643, 0.7768951064041224,
     0.6124043745852509, 0.43720730784452805, 0.473210808417582, 0.8531596576385212, 0.14035901395462655,
     0.7113303324211939, 0.27552373818196585, 0.17820317712819186, 0.20092865770892088], 'o')
plt.show()

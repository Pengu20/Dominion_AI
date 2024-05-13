




import numpy as np
import pickle
import os


list_val = [1,2,3,4,5,6,7,8,9]

list_val2 = []

list_val2.append(list_val)
list_val2.append(list_val)
list_val2.append(list_val)
list_val2.append(list_val)

print(np.asarray(list_val2))
print(np.mean(np.asarray(list_val2), axis=0))
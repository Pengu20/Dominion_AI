



import numpy as np
list_NN_input = np.array([1,2,3,45,6])

# Padding the value to 9000
input_padded = np.zeros((1, 9000))
input_padded[1, :len(list_NN_input)] = list_NN_input

NN_inputs[i,:] = input_padded[:,0]
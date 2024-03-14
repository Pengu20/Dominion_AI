import numpy as np


element = 2*np.arange(4).reshape((2, 2))


test_elements = [1, 2, 4, 8,7]


print(np.random.choice(test_elements, 0, replace=False))

def test():
    print("Hello")

    return 7,6


a,b,c,d = test()
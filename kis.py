
from itertools import combinations
import numpy as np

cards_in_hand_before = np.array([0, 3, 0, 3,8])

cards_in_hand_after =  np.array([3, 0, 3,8])

cards_in_hand_after = np.append(cards_in_hand_after, -1)

print(cards_in_hand_after)
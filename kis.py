
from itertools import combinations

list_comb = []
for i in range(1,4+1):
    for com in set(list(combinations([1,2], 0))):
        list_comb.append(com)


print(list_comb)
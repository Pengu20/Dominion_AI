
from cards_base_ed2 import kingdom_cards_ed2_base
from standard_cards import standard_set 

import numpy as np


class deck_generator:
    def __init__(self) -> None:
        self.kingdom_set = []
        self.standard_set = []

        # insert cards into the deck
        self.insert_set(kingdom_cards_ed2_base)
        self.insert_standard_set(standard_set=standard_set)




    def insert_set(self, set):
        self.kingdom_set.append(set)

    def insert_standard_set(self, standard_set):
        self.standard_set.append(standard_set)

    def print_sets(self):
        print("-------------- KINGDOM CARD SETS --------------")
        for sets in self.kingdom_set:
            for card in sets:
                print(card[0])

        print("-------------- Standard set --------------")

        for sets in self.standard_set:
            for card in sets:
                print(card[0])

    def set2numpy(self, input_set):
        res_set = []
        for set in input_set:
            for cards in set:
                res_set.append(cards)

        return np.array(res_set)
    

    def get_n_rand_kingdom_cards(self, n):

        numpy_kingdom_set = self.set2numpy(self.kingdom_set)

        # Get unique index
        res_idx = np.random.choice(range(numpy_kingdom_set.shape[0]), n, replace=False)

        res_set = numpy_kingdom_set[res_idx]

        return res_set

    def get_rand_kingdom_set(self):
        return self.get_n_rand_kingdom_cards(10)

    def get_standard_set(self):
        return self.set2numpy(self.standard_set)
    
    def get_card_set(self):
        kingdom_cards = self.get_rand_kingdom_set()
        standard_set = self.get_standard_set()

        card_list = []

        for card in standard_set:
            card_list.append(card)

        for card in kingdom_cards:
            card_list.append(card)

  
        card_list = np.array(card_list)

        return card_list

deck = deck_generator()

deck.get_n_rand_kingdom_cards(2)




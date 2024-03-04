import numpy as np




class random_player():
    def __init__(self) -> None:
        pass


    def choose_action(self, list_of_actions):
        return np.random.choice(list_of_actions)
    
    def choose_buy(self, list_of_buy_options):
        return np.random.choice(list_of_buy_options)
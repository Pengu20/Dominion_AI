import numpy as np

''' This is all the sizes used for the Q-table
"Player_won": bins: 1 - size: 3 # Amount of players +1

    # ----- SUPPLY RELATED -----
"dominion_cards": bins: 10 - size: 32    # 32 is the amount of unique cards in the game
"supply_amount": bins: 10 - size: 30


    # ----- WHAT THE ACTION MEANS -----
"Unique_actions": bins: 1 - size: 17 # There is 17 unique actions in the game
"Unique_actions_parameter": bins: 1 - size: 32 # The largest bin value is the amount of unique cards in the game


    # ----- MAIN PLAYER -----
"cards_in_hand": bins: 50 - size: 32, # Please do not get over 50 cards in your hand 
"cards_in_deck": bins: 1 - size: 80, # I will actually die if i get more than 80 cards in my deck
"known_cards_top_deck": bins: 20 - size: 32,
"cards_in_discard":  bins: 80 - size: 32,
"owned_cards":  bins: 80 - size: 32,
"played_cards": bins: 20 - size: 32, # "played" cards are cards that are in the current hand
"actions": bins: 1 - size: 10,
"buys": bins: 1 - size: 10,
"value": bins: 1 - size: 30,
"Victory_points": bins: 1 - size: 150,


    # ----- ADVERSARY PLAYER -----
"adv_cards_in_hand": bins: 1 - size: 15,
"adv_cards_in_deck": bins: 1 - size: 80,
"adv_cards_in_discard": bins: 1 - size: 80,
"adv_owned_cards": bins: 80 - size: 32,
"Victory_points": bins: 1 - size: 150,
'''


class Dominion_reward():
    ''' [summary]
        This class is used to determine the reward based on the state given in the dominion game
    '''

    def __init__(self) -> None:
        pass
        
    def get_reward_from_state(self, game_state):
        ''' [summary]
            This function is used to get the reward from the game state
            Args:
                game_state (struct): [description] This game state is a hand tailored struct that defines all the 
                                                   informations about the game
            Returns:
                reward (int): [description] The reward that the player gets from the game state
        '''

        reward = -1
        Victory_reward = 0 #100 if won -100 if lost
        Victory_points_reward = 2 # 1 per victory point

        Province_difference_reward = 0 # 3 per province difference

        Cards_played_reward = 0 # 3 if you played more than 3 cards
        few_coppers_reward = 0 # 3 if you have less than 3 coppers

        no_copper_reward = 0 # 3 if you have no coppers
        no_estates_reward = 0 # 3 if you have no estates

        gold_reward = 0 # get 1 point for each 3 value of player

        curses_owned = 0 # -1 point per curse

        Dead_action_cards = 0 # -1 point per action card in hand, if you have no actions left

    


        # ---------------- Reward based on game end ----------------
        if   (game_state["main_Player_won"] == 1):
            Victory_reward = 200
        elif (game_state["main_Player_won"] == 1):
            Victory_reward = -200



        # ---------------- Reward based on province difference of players ----------------
        province_main = 0
        province_adv = 0

        for card in game_state["owned_cards"]:
            if card == 5:
                province_main += 1

        for card in game_state["adv_owned_cards"]:
            if card == 5:
                province_adv += 1

        Province_difference_reward = (province_main - province_adv) * 3



        # ---------------- Points for having more victory points than the other players ----------------
        Victory_points_diff = game_state["Victory_points"] - game_state["adv_Victory_points"]
        Victory_points_reward = 2 * np.sign(Victory_points_diff)
        


        # ---------------- reward for playing many cards ----------------
        if len(game_state["played_cards"]) > 3:
            Cards_played_reward = 3



        # ---------------- reward for having no coppers ----------------
        coppers = 0
        for card in game_state["owned_cards"]:
            if card == 0:
                coppers += 1

        if coppers < 3:
            few_coppers_reward = 3
        
        if coppers == 0:
            no_copper_reward = 3
        

        # ---------------- reward for having no estates ----------------
        estates = 0
        for card in game_state["owned_cards"]:
            if card == 3:
                estates += 1

        if estates == 0:
            no_estates_reward = 3


        # ---------------- gold reward ----------------

        gold_reward = int(game_state["value"]/3)



        # ---------------- curse punishment ----------------
        curses = 0
        for card in game_state["owned_cards"]:
            if card == 6:
                curses += 1

        curses_owned = -1 * curses


        # ---------------- dead action cards punishment ----------------
        for card in game_state["cards_in_hand"]:
            if game_state["actions"] == 0:
                if card >= 6 and card != 13:
                    Dead_action_cards += 1
        
        Dead_action_cards = -1 * Dead_action_cards


        pass

    def struct_generator(self):
        self.reward = state = {

            # ----- SUPPLY RELATED -----
        "dominion_cards": self.deck.get_card_set(),
        "supply_amount": np.append(self.standard_supply, np.ones(10) * 10), # 10 kingdom cards with 10 supply each


            # ----- WHAT THE ACTION MEANS -----
        "Unique_actions": None, # This is the unique actions that the player can do. This is based on the cards in the players hand
        "Unique_actions_parameter": 0, # This is the parameter that the unique action needs to be executed (Often not needed default is zero)


            # ----- MAIN PLAYER -----
        "main_Player_won": -1,
        "cards_in_hand": np.array([]),
        "cards_in_deck": 0,
        "known_cards_top_deck": np.array([]),
        "cards_in_discard": np.array([]),
        "owned_cards": np.array([]),
        "played_cards": np.array([]), # "played" cards are cards that are in the current hand
        "actions": 0,
        "buys": 0,
        "value": 0,
        "Victory_points": 0,



            # ----- ADVERSARY PLAYER -----
        "adv_Player_won": -1,
        "adv_cards_in_hand": 0,
        "adv_cards_in_deck": 0,
        "adv_cards_in_discard": 0,
        "adv_owned_cards": np.array([]),
        "Victory_points": 0,
        }






class random_player():
    def __init__(self) -> None:
        self.rf = reward_func((2,10,10,17,), 10)
        pass


    def choose_action(self, list_of_actions, game_state):
        return np.random.choice(list_of_actions)
    
    
    def choose_buy(self, list_of_buy_options, game_state):
        return np.random.choice(list_of_buy_options)
    


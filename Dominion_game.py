
from Deck_generator import deck_generator
from Player_AI import random_player
import numpy as np
from cards_base_ed2 import kingdom_cards_ed2_base
from standard_cards import standard_set
from card_effects import card_effects
import state_manipulator as sm

deck = deck_generator()

class Dominion:
    def __init__(self) -> None:

        self.deck = deck_generator()


        # Standard supply [Copper, Silver, Gold, Estate, Duchy, Province, Curse]
        self.standard_supply = np.array([30, 30, 30, 30, 30, 8, 30])

        self.game_state = self.__initialize_game_state()
        
        self.Treasure_card_index = [0,1,2] #Indexes of the treasure cards in the standard set



    def __initialize_game_state(self):
        # This datatype is the object that holds all information regarding the game
        state = {
            # ----- GAME END -----
        "Player_won": -1,


            # ----- SUPPLY RELATED -----
        "dominion_cards": self.deck.get_card_set(),
        "supply_amount": np.append(self.standard_supply, np.ones(10) * 10), # 10 kingdom cards with 10 supply each

            # ----- GAME PHASE RELATED -----
        "game_phase": 0, # 0 indicates action phase, 1 indicates buy phase

            # ----- CARD EFFECT RELATED -----

        "Unique_actions": "None", # This is the unique actions that the player can do. This is based on the cards in the players hand


            # ----- MAIN PLAYER -----
        "cards_in_hand": np.array([]),
        "cards_in_deck": 0,
        "cards_in_discard": np.array([]),
        "owned_cards": np.array([]),
        "played_cards": np.array([]), # "played" cards are cards that are in the current hand
        "actions": 0,
        "buys": 0,
        "value": 0,
        "Victory_points": 0,



            # ----- ADVERSARY PLAYER -----
        "adv_cards_in_hand": 0,
        "adv_cards_in_deck": 0,
        "adv_cards_in_discard": 0,
        "adv_owned_cards": np.array([]),
        "Victory_points": 0,
        }
        return state



    def __startup_player_state(self):
        player_state_start = {
            "cards_in_hand": np.array([]),
            "cards_in_deck": 10,
            "cards_in_discard": np.array([]),
            "owned_cards": np.array([0, 0, 0, 0, 0, 0, 0, 3, 3, 3]), # 7 coppers and 3 estates
            "played_cards": np.array([]), # "played" cards are cards that are in the current hand
            "actions": 0,
            "buys": 0,
            "value": 0,
            "Victory_points": 0,
        }

        return player_state_start


    def play_loop_AI(self, player1, player2):
        ''' [Summary]
        This function is the main loop of the game. It will keep running until the game is over.

        ARGS:
            player1 player: This type object must be capable of choosing which cards to play and when.
            player2 player: This is also a player type 
        '''

        players_amount = 2

        player1_state = self.__startup_player_state()
        player2_state = self.__startup_player_state()

        players = [player1_state, player2_state]
        players_input = [player1, player2]

        # randomize starting player
        main_player = np.random.choice([0, 1])

        # both players draw 5 cards in the start of the game
        for player in range(players_amount):
            players[player] = sm.draw_n_cards_from_deck(players[player], 5)


        while self.game_state["Player_won"] == -1:
            # --------- ACTION PHASE ---------
            
            
            # Get all possible actions
            actions = self.__get_actions(players[main_player])
  
            # Choose action
            self.game_state = sm.merge_game_player_state(self.game_state, players[main_player], players[main_player*(-1) + 1])
            action = players_input[main_player].choose_action(actions, self.game_state)

            print("Testing chapel")
            print("------------------- BEFORE -------------------")
            print("cards in hand: ", players[main_player]["cards_in_hand"])
            print("cards in discard: ", players[main_player]["cards_in_discard"])
            print("cards in deck: ", players[main_player]["cards_in_deck"])
            print("owned cards: ", players[main_player]["owned_cards"], "-> size ->", len(players[main_player]["owned_cards"]))
            print("action values: ", players[main_player]["actions"])
            print("player value: ", players[main_player]["value"])

            main = int(main_player)
            advesary = int(main_player*(-1) + 1)
            card_val = 10
            sm.get_card2hand(players[main], card_val)
            card_effects().play_card(10, self.game_state, players[main], players_input[main],  players[advesary], players_input[advesary])
            
            print("------------------- AFTER -------------------")
            print("cards in hand: ", players[main_player]["cards_in_hand"])
            print("cards in discard: ", players[main_player]["cards_in_discard"])
            print("cards in deck: ", players[main_player]["cards_in_deck"])
            print("owned cards: ", players[main_player]["owned_cards"], "-> size ->", len(players[main_player]["owned_cards"]))
            print("action values: ", players[main_player]["actions"])
            print("player value: ", players[main_player]["value"])


            while action != -1:
                action = player1.choose_action(actions, self.game_state)



            # --------- BUY PHASE ---------
            players[main_player]["buys"] += 1
            players[main_player]["value"] += self.__get_player_treasure_value(players[main_player], self.game_state, players[main_player])


            buy_actions = self.__get_buys(players[main_player], self.game_state)

            # Choose a buy
            action = player1.choose_action(buy_actions, self.game_state)


            while action != -1:
                players[main_player], self.game_state = self.__buy_card_from_supply(player_state=players[main_player], game_state=self.game_state, card_idx=action)
                buy_actions = self.__get_buys(players[main_player], self.game_state)

                action = player1.choose_action(buy_actions, self.game_state)




            # Reset and draw new hand
            main_player += 1
            if main_player >= players_amount:
                main_player = 0

            self.game_state["Player_won"] = 1




        return 0 # return index of who wins.

        
        

    def __get_actions(self, player_state):
        '''[Summary]
        
        This function will return the actions that the player can do. 
        The actions are based on the index of the cards in the players hand.
        -1 means end turn
        '''

        if player_state["actions"] == 0:
            return np.array([-1])
        
        actions = player_state["cards_in_hand"]
        actions = np.append(-1, actions) # add the ability to terminate the action phase

        return actions
    

    def __get_buys(self, player_state, game_state):
        '''[Summary]
        This function will return the buys the player can do.
        '''


        # If there is enough cards in supply, and the player has enough currency, then add the card to the list of buyable cards
            
        buyable_cards = np.array([])

        for i in range(len(game_state["dominion_cards"])):
            if game_state["supply_amount"][i] > 0 and game_state["dominion_cards"][i][2].astype(int) <= player_state["value"]:
                buyable_cards = np.append(buyable_cards, game_state["dominion_cards"][i][1].astype(int))



        # Add the ability to terminate the buying phase (-1)
        buyable_cards = np.append(buyable_cards, -1)

        return buyable_cards



    def __buy_card_from_supply(self, player_state, game_state, card_idx):
        # This function lets the player put a card in the discard pile. based on the input card which is a card ID.


        card = self.get_card_from_index(card_idx=card_idx)
        # Put card into discard pile.
        player_state["cards_in_discard"] = np.append(player_state["cards_in_discard"], card[1].astype(int))


        # remove one instance from supply.
        set_index = self.card_idx_2_set_idx(card_idx) # Get the index in the set of card
        game_state["supply_amount"][set_index] -= 1


        # remove the value of the card from the players value
        player_state["value"] -= card[2].astype(int)


        # Remove one buy power from player
        player_state["buys"] -= 1

        

        return player_state, game_state

    def __get_player_treasure_value(self, player_state, game_state, player_input):
        '''[Summary]
        This function will return the value of the players hand.

        ARGS:
            player_state [dict]: This is the player state object
        
        RETURNS:
            value [int]: The value of the players hand
            card [numpy.list]: All the indexes of the treasure cards sorted from highest value to lowest
        '''


        cards_in_hand = player_state["cards_in_hand"]

        treasure_cards = np.argwhere(np.isin(cards_in_hand, self.Treasure_card_index)).flatten()


        for index in treasure_cards:
            game_state, player_state = card_effects().play_card(cards_in_hand[index].astype(int), game_state, player_state, player_input, card2played_cards=False)


        player_value = player_state["value"]

        return player_value


    
    def get_card_from_index(self, card_idx):
        # this function gets the card from the supply pile. 
        card_idx = int(card_idx)

        for card in self.game_state["dominion_cards"]:
            if int(card[1]) == card_idx:
                return card

    def card_idx_2_set_idx(self, card_idx):
        # This function will return the index of the card in the dominion_cards game state, based on the card index

        card_idx = int(card_idx)

        for i in range(len(self.game_state["dominion_cards"])):
            if int(self.game_state["dominion_cards"][i][1]) == card_idx:
                return i


        return -1 #  Returns -1 if the card is not found in the dominion_cards game state






    


    

    def __insert_card_in_discard(self, player_state, card):
        player_state["cards_in_discard"] = np.append(int(card[1]), player_state["cards_in_discard"])
        return player_state
    
    def __insert_card_in_deck(self, player_state, card):
        player_state["cards_in_deck"] += 1
        return player_state


    
    def __remove_card_from_discard(self, player_state, card):
        player_state["cards_in_discard"] = np.delete(player_state["cards_in_discard"], np.where(player_state["cards_in_discard"] == card))
        return player_state




Dominion_game = Dominion()

Dominion_game.play_loop_AI(random_player(), random_player())

from Deck_generator import deck_generator
from Player_AI import random_player
import numpy as np
from cards_base_ed2 import kingdom_cards_ed2_base
from standard_cards import standard_set
from card_effects import card_effects

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

        # randomize starting player
        main_player = np.random.choice([0, 1])

        # both players draw 5 cards in the start of the game
        for player in range(players_amount):
            players[player] = self.__draw_n_cards_from_deck(players[player], 5)


        while self.game_state["Player_won"] == -1:
            # --------- ACTION PHASE ---------
            
            # Get all possible actions
            actions = self.__get_actions(players[main_player])
  
            # Choose action
            action = player1.choose_action(actions, self.game_state)


            while action != -1:
                action = player1.choose_action(actions, self.game_state)





            # --------- BUY PHASE ---------
            buy_actions = self.__get_buys(players[main_player], self.game_state)

            # Choose a buy
            action = player1.choose_action(buy_actions, self.game_state)

            while action != -1:
                players[main_player] = self.__buy_card_from_supply(player_state=players[main_player], game_state=self.game_state, card_idx=action)
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

        cards_in_hand = player_state["cards_in_hand"]

        treasure_cards = np.argwhere(np.isin(cards_in_hand, self.Treasure_card_index)).flatten()

        # Get the value from all treasure cards in hand
        for index in treasure_cards:
            game_state, player_state = card_effects().play_card(cards_in_hand[index].astype(int), game_state, player_state)


        player_value = player_state["value"]

        # If there is enough cards in supply, and the player has enough currency, then add the card to the list of buyable cards
            
        buyable_cards = np.array([])

        for i in range(len(game_state["dominion_cards"])):
            if game_state["supply_amount"][i] > 0 and game_state["dominion_cards"][i][2].astype(int) <= player_value:
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




    
    def get_card_from_index(self, card_idx):
        # this function gets the card from the supply pile. 
        card_idx = int(card_idx)

        for card in self.game_state["dominion_cards"]:
            if int(card[1]) == card_idx:
                return card





    def __draw_n_cards_from_deck(self, player_state, n):
        deck = self.__get_cards_in_deck(player_state)
        draws = np.random.choice(deck, n, replace=False)
        
        player_state["cards_in_hand"] = np.append(player_state["cards_in_hand"], draws)
        return player_state


    def __get_cards_in_deck(self, player_state):
        ''' [Summary]
        Based on the cards in the discard pile, and cards in the hand and all the known cards.
        This function will return the cards in the deck.

        ARGS:
            player_state [dict]: This is the player state object
        '''

        hand   = player_state["cards_in_hand"]
        discard_pile = player_state["cards_in_discard"]

        hand_discard = np.concatenate((hand, discard_pile), axis=0)


        all_owned_cards = player_state["owned_cards"]
        for cards in hand_discard:
            all_owned_cards = np.delete(all_owned_cards, np.where(all_owned_cards == cards)[0][0])
        
        deck = all_owned_cards
        return deck
    

    def __insert_card_in_hand(self, player_state, card):
        
        player_state["cards_in_hand"] = np.append(int(card[1]), player_state["cards_in_hand"])
        return player_state
    

    def __insert_card_in_discard(self, player_state, card):
        player_state["cards_in_discard"] = np.append(int(card[1]), player_state["cards_in_discard"])
        return player_state
    
    def __insert_card_in_deck(self, player_state, card):
        player_state["cards_in_deck"] += 1
        return player_state

    def __remove_card_from_hand(self, player_state, card):
        player_state["cards_in_hand"] = np.delete(player_state["cards_in_hand"], np.where(player_state["cards_in_hand"] == card))
        return player_state
    
    def __remove_card_from_discard(self, player_state, card):
        player_state["cards_in_discard"] = np.delete(player_state["cards_in_discard"], np.where(player_state["cards_in_discard"] == card))
        return player_state




Dominion_game = Dominion()

Dominion_game.play_loop_AI(random_player(), random_player())
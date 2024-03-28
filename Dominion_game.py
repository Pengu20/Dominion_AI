
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

        self.victory_points_index = [3,4,5,13,6]

        self.card_effects = card_effects()




    def __initialize_game_state(self):
        # This datatype is the object that holds all information regarding the game
        # The game state 
        state = {

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
        "adv_Victory_points": 0,
        }
        return state



    def __startup_player_state(self):
        player_state_start = {
            "cards_in_hand": np.array([]),
            "cards_in_deck": 10,
            "known_cards_top_deck": np.array([]),
            "cards_in_discard": np.array([]),
            "owned_cards": np.array([0, 0, 0, 0, 0, 0, 0, 3, 3, 3]), # 7 coppers and 3 estates
            "played_cards": np.array([]), # "played" cards are cards that are in the current hand
            "actions": 0,
            "buys": 0,
            "value": 0,
            "Victory_points": 0,
        }

        return player_state_start




    def play_loop_AI(self, player1, player2, verbose=True):
        ''' [Summary]
        This function is the main loop of the game. It will keep running until the game is over.

        ARGS:
            player1 player: This type object must be capable of choosing which cards to play and when.
            player2 player: This is also a player type 
            verbose [int]: this variable will determine if the game history is written to a file or not.
        '''

        if verbose:
            game_history_file = open("game_history.txt", "w")
            game_history_file.write(" --------- CARDS IN THIS GAME --------- \n")
        else:
            game_history_file = None


        dominion_cards = self.game_state["dominion_cards"]


        if verbose:
            game_history_file.write(f"Game cards: {dominion_cards}")
            game_history_file.write("\n"*4)


        players_amount = 2 # hard constant for this game setup. Cannot play more or less than two players 


        player1_state = self.__startup_player_state()
        player2_state = self.__startup_player_state()

        # randomize starting player
        main_player = np.random.choice([0, 1])


        # both players draw 5 cards in the start of the game
        players = [player1_state, player2_state]
        players_input = [player1, player2]

        for player in range(players_amount):
            players[player] = sm.draw_n_cards_from_deck(players[player], 5)
        
        turns = 0
        turns_all_players = 0


        while self.game_state["Player_won"] == -1:
            
            if verbose:
                game_history_file.write("\n"*3)

                
            if turns_all_players % 2 == 0:
                turns += 1
                # print("         -------------------- Turn: ", turns, " -------------------- ")

                if verbose:
                    game_history_file.write(f"         -------------------- Turn: {turns} -------------------- \n")


            cards_in_hand = players[main_player]["cards_in_hand"]


            if verbose:
                game_history_file.write(f"Player: {main_player} \n")
                game_history_file.write(f"cards in hand: {cards_in_hand} \n")



            main = int(main_player)
            advesary = int(main_player*(-1) + 1)



            # --------- ACTION PHASE ---------
            if verbose:
                game_history_file.write(f"----------- ACTION PHASE ----------- \n")

            action_turns = self.__action_phase(players, players_input, main, advesary, game_history_file, verbose=verbose)



            # --------- BUY PHASE ---------
            if verbose:
                game_history_file.write("\n"*2)
                game_history_file.write(f"----------- BUY PHASE ----------- \n")
            buy_turns = self.__buy_phase(players, players_input, main, advesary, game_history_file, verbose=verbose)




            if action_turns == 0 and buy_turns == 0 and verbose:
                game_history_file.write(f" --------- NOTHING HAPPENED --------- \n")
                # self.__Debug_state(players, main_player, players_input, game_history_file)




            # ---------- HIDDEN PHASE: check if game should terminate ---------
            if self.__game_is_over() or turns >= 1000:
                self.game_state = self.__Update_victory_points(self.game_state, players[main_player])
                main_player_victory_points = players[main_player]["Victory_points"]

                self.game_state = self.__Update_victory_points(self.game_state, players[advesary])
                advesary_victory_points = players[advesary]["Victory_points"]

                if main_player_victory_points > advesary_victory_points:
                    self.game_state["Player_won"] = main_player
                    if verbose:
                        game_history_file.write(f"Player {main_player} won the game! \n")
                    break
                else:
                    self.game_state["Player_won"] = advesary
                    if verbose:
                        game_history_file.write(f"Player {advesary} won the game! \n")
                    break





            # Reset and draw new hand
            self.game_state = sm.put_player_state_adv_state(self.game_state, players[main_player])  
            players[main_player]["value"] = 0
            players[main_player]["actions"] = 0
            players[main_player]["buys"] = 0

            # flush all cards in hand and discard
            self.game_state = sm.played_cards_2_discard_pile(self.game_state, players[main_player])
            self.game_state = sm.discard_hand(self.game_state, players[main_player])

            players[main_player] = sm.get_player_state_from_game_state(self.game_state)
            players[main_player] = sm.draw_n_cards_from_deck(players[main_player], 5)


            turns_all_players += 1


            main_player += 1
            if main_player >= players_amount:
                main_player = 0
            
            


        return 0 # return index of who wins.
    
    def __game_is_over(self):
        '''[Summary]
        This function will check if the game is over. 
        If the game is over, it will return the index of the player that won.

        Conditions for game is over: All provinces are bought, or three different piles are empty.
        '''

        game_is_over = False

        if self.game_state["supply_amount"][5] == 0:
            game_is_over = True
        
        empty_piles = 0
        for pile in self.game_state["supply_amount"]:
            if pile == 0:
                empty_piles += 1
        
        if empty_piles >= 3:
            game_is_over = True


        return game_is_over

    def __action_phase(self, players, players_input, main, adversary, game_history_file, verbose=True):
        ''' [Summary]
        This function handles the action phase of the game. 
        It will keep running until the player decides to end the action phase.

        ARGS:
            players [list]: This is a list of the players state
            players_input [list]: This is a list of the players input
            main [int]: This is the index of the main player
            adversary [int]: This is the index of the adversary player
            game_history_file [file]: This is the file that the game history is written to
        
        RETURNS:
            action_turns [int]: The amount of action turns the player took
        '''
        action_turns = 0

        players[main]["actions"] = 1

        # Get all possible actions
        actions = self.__get_actions(players[main])
        
        if verbose:
            game_history_file.write(f"action possibilites: {actions} \n")


        # Choose action
        self.game_state = sm.merge_game_player_state(self.game_state, players[main], players[adversary])
        self.game_state["Unique_actions"] = "take_action"
        play_action = int(players_input[main].choose_action(actions, self.game_state))




        # DEBUG option: Play specific card
        debug_card = 32
        sm.get_card2hand(players[main], debug_card)
        play_action = debug_card
    



        card_idx = sm.card_idx_2_set_idx(play_action, self.game_state)
        if card_idx != -1:
            card_obj = self.game_state["dominion_cards"][card_idx]
        else:
            card_obj = "None"


        if verbose:
            game_history_file.write(f"Chosen action: {play_action} : {card_obj}\n")

        
        
        while play_action != -1:
            action_turns += 1
            players[main]["actions"] -= 1

            self.card_effects.play_card(play_action, self.game_state, players[main], players_input[main],  players[adversary], players_input[adversary])
            
            if verbose:
                self.__Debug_state(players, main, players_input, game_history_file)


            actions = self.__get_actions(players[main])
            self.game_state["Unique_actions"] = "buy_card"
            play_action = int(players_input[main].choose_action(actions, self.game_state))


            card_idx = sm.card_idx_2_set_idx(play_action, self.game_state)
            if card_idx != -1:
                card_obj = self.game_state["dominion_cards"][card_idx]
            else:
                card_obj = "None"

            if verbose:
                game_history_file.write(f"action possibilites: {actions} \n")
                game_history_file.write(f"Chosen action: {play_action} : {card_obj}\n")
            



        players[main]["actions"] = 0


        return action_turns

    def __buy_phase(self, players, players_input, main, adversary, game_history_file, verbose):
        '''
        [Summary]
        This function handles the buy phase of the game.
        
        ARGS:
            players [list]: This is a list of the players state
            players_input [list]: This is a list of the players input
            main [int]: This is the index of the main player
            adversary [int]: This is the index of the adversary player
            game_history_file [file]: This is the file that the game history is written to

        RETURNS:
            buy_actions_amount [int]: The amount of buy actions the player took
        '''
        buy_actions_amount = 0


        self.game_state = sm.merge_game_player_state(self.game_state, players[main], players[adversary])
        self.game_state["Unique_actions"] = "buy"
        players[main]["buys"] = 1


        self.game_state = self.__update_player_treasure_value(players[main], self.game_state, players_input[main])


        list_of_actions_buy = self.__get_buys(players[main], self.game_state)

        if verbose: 
            game_history_file.write(f"buy possibilites: {list_of_actions_buy} \n")



        # Choose a buy
        buy_action = players_input[main].choose_action(list_of_actions_buy, self.game_state)
        card_idx = sm.card_idx_2_set_idx(buy_action, self.game_state)
        if card_idx != -1:
            card_obj = self.game_state["dominion_cards"][card_idx]
        else:
            card_obj = "None"

        if verbose: 
            game_history_file.write(f"Chosen action: {buy_action} : {card_obj}\n")



        while buy_action != -1:
            buy_actions_amount += 1
            players[main]["buys"] -= 1



            players[main], self.game_state = self.__buy_card_from_supply(player_state=players[main], game_state=self.game_state, card_idx=buy_action)
            self.game_state = sm.merge_game_player_state(self.game_state, players[main], players[adversary])
            
            if verbose:
                self.__Debug_state(players, main, players_input, game_history_file)
            

            
            list_of_actions_buy = self.__get_buys(players[main], self.game_state)
            buy_action = players_input[main].choose_action(list_of_actions_buy, self.game_state)
            card_idx = sm.card_idx_2_set_idx(buy_action, self.game_state)
            if card_idx != -1:
                card_obj = self.game_state["dominion_cards"][card_idx]
            else:
                card_obj = "None"
            
            if verbose: 
                game_history_file.write(f"buy possibilites: {list_of_actions_buy} \n")
                game_history_file.write(f"Chosen action: {buy_action} : {card_obj}\n")
        


        players[main]["buys"] = 0

        return buy_actions_amount


    def __Update_victory_points(self, game_state, player_state):
        '''[Summary]
        This function will update the victory points state of the player

        ARGS:
            game_state [dict]: This is the game state object
            player_state [dict]: This is the player state object
        
        RETURNS:
            victory_points [int]: The amount of victory points the player has
        '''

        player_state["Victory_points"] = 0
        all_cards = player_state["owned_cards"]

        

        for card in all_cards:
            if card in self.victory_points_index:
                self.card_effects.play_card(card, game_state, player_state, card2played_cards=False)
        

        return game_state


    def __Debug_state(self, players, main_player, players_input, game_history_file, gain_card = False):
            self.game_state = self.__Update_victory_points(self.game_state, players[main_player])
            game_state_temp = self.__update_player_treasure_value(players[main_player], self.game_state, players_input[main_player])

            players[main_player] = sm.get_player_state_from_game_state(game_state_temp)


            game_history_file.write("\n"*2)
            game_history_file.write(f" --- GAME STATE ---\n")

            cards_in_hand = players[main_player]["cards_in_hand"]
            game_history_file.write(f"cards in hand: {cards_in_hand} \n")

            cards_in_discard = players[main_player]["cards_in_discard"]
            game_history_file.write(f"cards in discard: {cards_in_discard} \n")

            cards_in_deck = players[main_player]["cards_in_deck"]
            game_history_file.write(f"cards in deck: {cards_in_deck} \n")

            known_top_deck_cards = self.game_state["known_cards_top_deck"]
            game_history_file.write(f"card top of deck: {known_top_deck_cards} \n")

            played_cards = players[main_player]["played_cards"]
            game_history_file.write(f"played cards: {played_cards} \n")

            owned_cards = players[main_player]["owned_cards"]
            length_owned_cards = len(players[main_player]["owned_cards"])
            game_history_file.write(f"owned cards: {owned_cards} -> size -> {length_owned_cards} \n")

            actions = players[main_player]["actions"]
            game_history_file.write(f"action values: {actions} \n")

            buys = players[main_player]["buys"]
            game_history_file.write(f"buys: {buys} \n")

            value = players[main_player]["value"]
            game_history_file.write(f"player value: {value} \n")

            supply_amount = self.game_state["supply_amount"]
            game_history_file.write(f"card supply: {supply_amount} \n")

            cards_in_hand_adv = players[main_player*(-1) + 1]["cards_in_hand"]
            game_history_file.write(f"adversary cards in hand: {cards_in_hand_adv} \n")

            adv_discard = players[main_player*(-1) + 1]["cards_in_discard"]
            game_history_file.write(f"adversary cards in discard: {adv_discard} \n")

            adv_owned = players[main_player*(-1) + 1]["owned_cards"]
            length_owned_cards = len(players[main_player*(-1) + 1]["owned_cards"])
            game_history_file.write(f"adversary cards in deck: {adv_owned} -> size -> {length_owned_cards} \n")


            victory_points = players[main_player]["Victory_points"]
            game_history_file.write(f"player victory points: {victory_points} \n")


            game_history_file.write("\n"*2)



    def __get_actions(self, player_state):
        '''[Summary]
        
        This function will return the actions that the player can do. 
        The actions are based on the index of the cards in the players hand.
        -1 means end turn
        '''

        if player_state["actions"] == 0:
            return np.array([-1])
        
        actions = player_state["cards_in_hand"]


        # Remove all cards that are not actions, treasures and victory cards
        actions = np.array([card for card in actions if not(card in self.Treasure_card_index)])
        actions = np.array([card for card in actions if not(card in self.victory_points_index)])

        actions = np.append(-1, actions) # add the ability to terminate the action phase

        return actions
    

    def __get_buys(self, player_state, game_state):
        '''[Summary]
        This function will return the buys the player can do.
        '''

        if player_state["buys"] == 0:
            return np.array([-1])


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

        # put the card to owned cards
        player_state["owned_cards"] = np.append(player_state["owned_cards"], card[1].astype(int))

        # remove one instance from supply.
        set_index = sm.card_idx_2_set_idx(card_idx, self.game_state) # Get the index in the set of card
        game_state["supply_amount"][set_index] -= 1


        # remove the value of the card from the players value
        player_state["value"] -= card[2].astype(int)




        return player_state, game_state




    def __update_player_treasure_value(self, player_state, game_state, player_input):
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

        silver_played = False
        for index in treasure_cards:
            game_state, player_state, adv_state = self.card_effects.play_card(cards_in_hand[index].astype(int), game_state, player_state, player_input, card2played_cards=False)
            
            if cards_in_hand[index].astype(int) == 1:
                silver_played = True


        # Special attention to the merchant who also counts as +1 value if there has been a silver.
        if silver_played:
            played_cards = player_state["played_cards"]

            merchants_in_played_cards = [1 for card in played_cards if card == 27]
            player_state["value"] += len(merchants_in_played_cards)



        game_state = sm.put_player_state_adv_state(game_state, player_state)


        return game_state



    
    def get_card_from_index(self, card_idx):
        # this function gets the card from the supply pile. 
        card_idx = int(card_idx)

        for card in self.game_state["dominion_cards"]:
            if int(card[1]) == card_idx:
                return card



    

    def __insert_card_in_discard(self, player_state, card):
        player_state["cards_in_discard"] = np.append(int(card[1]), player_state["cards_in_discard"])
        return player_state
    
    def __insert_card_in_deck(self, player_state, card):
        player_state["cards_in_deck"] += 1
        return player_state


    
    def __remove_card_from_discard(self, player_state, card):
        player_state["cards_in_discard"] = np.delete(player_state["cards_in_discard"], np.where(player_state["cards_in_discard"] == card))
        return player_state







i = 0
while True:
    Dominion_game = Dominion()
    Dominion_game.play_loop_AI(random_player(), random_player(), verbose=False)
    i += 1











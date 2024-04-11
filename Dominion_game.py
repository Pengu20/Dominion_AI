
from Deck_generator import deck_generator


from Player_AI import random_player
from Player_AI import Deep_SARSA
from Player_AI import greedy_NN
from Player_AI import Deep_Q_learning
from Player_AI import Deep_expected_sarsa



import numpy as np
from cards_base_ed2 import kingdom_cards_ed2_base
from standard_cards import standard_set
from card_effects import card_effects
import state_manipulator as sm

import copy

import pickle

deck = deck_generator()


def make_card_set(kingdom_indexes):
    '''
    This function takes the indexes of all the kingdom cards that is desired to be in game, and generated a card set.
    The standard cards remain the same always.
    '''
    kingdom_indexes = [i - 7 for i in kingdom_indexes]

    numpy_kingdom = np.array(kingdom_cards_ed2_base)
    kingdom_cards = numpy_kingdom[kingdom_indexes]

    deck = deck_generator()
    card_set = deck.get_card_set(kingdom_cards=kingdom_cards)

    return card_set

class Dominion:

    def __init__(self) -> None:

        self.deck = deck_generator()

        # Standard supply [Copper, Silver, Gold, Estate, Duchy, Province, Curse]
        self.standard_supply = np.array([30, 30, 30, 30, 30, 8, 30])

        self.card_set = self.deck.get_card_set()

        self.game_state = self.initialize_game_state()
        
        self.Treasure_card_index = [0,1,2] #Indexes of the treasure cards in the standard set

        self.victory_points_index = [3,4,5,13,6]

        self.card_effects = card_effects()

        self.player0_bought_cards = np.array([0]*17)
        self.player1_bought_cards = np.array([0]*17)

        self.testplayer_province_boosted = False

    def get_card_set(self):
        return self.card_set

    def set_players(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def set_player2train(self, player1):
        self.player1 = player1

    def set_player2test(self, player2):
        self.player2 = player2


    def initialize_game_state(self):
        # This datatype is the object that holds all information regarding the game
        # The game state 
        state = {

            # ----- SUPPLY RELATED -----
        "dominion_cards": self.card_set,
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


    def play_loop_AI(self,game_name, player_0_is_NN, player_1_is_NN, verbose=True):
        ''' [Summary]
        This function is the main loop of the game. It will keep running until the game is over.

        ARGS:
            game_name [string]: This is the name of the game. This is used to save the game history to a file.
            player_0_is_NN [bool]: This is a boolean that determines if player 0 is a neural network or not.
            player_1_is_NN [bool]: This is a boolean that determines if player 1 is a neural network or not.
            verbose [int]: this variable will determine if the game history is written to a file or not.

            player1 gets provinces inserted into deck at random times after 20 turns
        '''


        self.player0_bought_cards = np.array([0]*17)
        self.player1_bought_cards = np.array([0]*17)

        self.game_state = copy.deepcopy(self.initialize_game_state())

        if verbose:
            game_history_file = open(f"game_history/{game_name}.txt", "w")
            game_history_file.write(" --------- CARDS IN THIS GAME --------- \n")
        else:
            game_history_file = None


        dominion_cards = self.game_state["dominion_cards"]


        if verbose:
            game_history_file.write(f"Game cards: {dominion_cards}")
            game_history_file.write("\n"*4)


        players_amount = 2 # hard constant for this game setup. Cannot play more or less than two players 




        player1_state = copy.deepcopy(self.__startup_player_state())
        player2_state = copy.deepcopy(self.__startup_player_state())

        # randomize starting player
        main_player = np.random.choice([0, 1])


        # both players draw 5 cards in the start of the game
        players = [player1_state, player2_state]
        players_input = [self.player1, self.player2]
        player_is_NN = [player_0_is_NN, player_1_is_NN]

        for player in range(players_amount):
            players[player] = sm.draw_n_cards_from_deck(players[player], 5)
        
        turns = 0
        turns_all_players = 0

        game_ongoing = True
        while game_ongoing:
            
            
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
            NN_player = player_is_NN[main_player]


            # Give the player a time limit.
            if self.testplayer_province_boosted and main_player == 1:
                if turns >= 25 :
                    province_chance = 0.4

                    if turns >= 30:
                        province_chance = 0.5

                    if turns >= 40:
                        province_chance = 0.6


                    if np.random.rand() < province_chance:
                        players[main_player], self.game_state = self.__buy_card_from_supply(player_state=players[main_player], game_state=self.game_state, card_idx=5)
                        self.game_state = sm.merge_game_player_state(self.game_state, players[main_player])


           
            players[main_player]["victory_points"] = self.__Update_victory_points(self.game_state, players[main_player])
            players[advesary]["victory_points"] = self.__Update_victory_points(self.game_state, players[advesary])
           



            # --------- ACTION PHASE ---------
            if verbose:
                game_history_file.write(f"----------- ACTION PHASE ----------- \n")

            action_turns = self.__action_phase(players, players_input, NN_player, main, advesary, game_history_file, verbose=verbose)



            # --------- BUY PHASE ---------
            if verbose:
                game_history_file.write("\n"*2)
                game_history_file.write(f"----------- BUY PHASE ----------- \n")
            buy_turns = self.__buy_phase(players, players_input, NN_player, main, advesary, game_history_file, verbose=verbose)

            self.game_state["Unique_actions"] = None


            if action_turns == 0 and buy_turns == 0 and verbose:
                game_history_file.write(f" --------- NOTHING HAPPENED --------- \n")
                # self.__Debug_state(players, main_player, players_input, game_history_file)




            # ---------- HIDDEN PHASE: check if game should terminate ---------
            if self.__game_is_over() or turns >= 100:
                game_ongoing = False


                game_state_player0 = sm.merge_game_player_state(copy.deepcopy(self.game_state), players[main_player], players[advesary])
                self.game_state = self.__Update_victory_points(game_state_player0, players[main_player])
                main_player_victory_points = players[main_player]["Victory_points"]


                game_state_player1 = sm.merge_game_player_state(copy.deepcopy(self.game_state), players[advesary], players[main_player])
                self.game_state = self.__Update_victory_points(game_state_player1, players[advesary])
                advesary_victory_points = players[advesary]["Victory_points"]

                

                if main_player_victory_points > advesary_victory_points:
                    player_won = main_player
                    main_player_won = True
                    adv_player_won = False
                    if verbose:
                        game_history_file.write(f"Player {main_player} won the game! \n")
                elif main_player_victory_points < advesary_victory_points:
                    player_won = advesary
                    main_player_won = False
                    adv_player_won = True
                    if verbose:
                        game_history_file.write(f"Player {advesary} won the game! \n")
                else:
                    player_won = 0.5
                    main_player_won = False
                    adv_player_won = False
                    if verbose:
                        game_history_file.write(f"Game is draw!\n")

                if verbose:
                    game_history_file.write(f"\n\n\n")
                    game_history_file.write(f"Player 0 bought cards:\n")

                    for idx in range(len(self.player0_bought_cards)):
                        card = self.game_state["dominion_cards"][idx]
                        game_history_file.write(f"{card[0]}: {self.player0_bought_cards[idx]} \n")
                        if idx == 6:
                            game_history_file.write("\n")

                    self.player0_bought_cards = np.array([0]*17)


                game_state_player0["main_Player_won"] = main_player_won
                game_state_player0["adv_Player_won"] = adv_player_won
                players_input[main].notify_game_end(game_state_player0)
      

                game_state_player1["main_Player_won"] = adv_player_won
                game_state_player1["adv_Player_won"] = main_player_won
                players_input[advesary].notify_game_end(game_state_player1)





            # flush all cards in hand and discard
            self.game_state = sm.played_cards_2_discard_pile(self.game_state, players[main_player])
            self.game_state = sm.discard_hand(self.game_state, players[main_player])


            players[main_player] = sm.get_player_state_from_game_state(self.game_state)
            players[main_player] = sm.draw_n_cards_from_deck(players[main_player], 5)


            # Reset and draw new hand
            self.game_state = sm.put_player_state_adv_state(self.game_state, players[main_player])  
            players[main_player]["value"] = 0
            players[main_player]["actions"] = 0
            players[main_player]["buys"] = 0




            turns_all_players += 1


            main_player += 1
            if main_player >= players_amount:
                main_player = 0
            
            


        return player_won # return index of who wins.
    
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

    def __action_phase(self, players, players_input,NN_player, main, adversary, game_history_file, verbose=True):
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
            if main == 0: # We only want to log the cards bought by player 0 (Sarsa trained player)
                NN_return = players_input[main].NN_get_expected_return(self.game_state, actions)
                game_history_file.write(f"expected returns: {NN_return}\n")


        # Choose action
        self.game_state = sm.merge_game_player_state(self.game_state, players[main], players[adversary])
        self.game_state["Unique_actions"] = "take_action"
        play_action = int(players_input[main].choose_action(actions, copy.deepcopy(self.game_state)))





        # DEBUG option: Play specific card
        # debug_card = 32
        # sm.get_card2hand(players[main], debug_card)
        # play_action = debug_card
    



        card_idx = sm.card_idx_2_set_idx(play_action, self.game_state)
        if card_idx != -1:
            card_obj = self.game_state["dominion_cards"][card_idx]
        else:
            card_obj = "None"


        if verbose:
            game_history_file.write(f"Chosen action: {play_action} : {card_obj}\n")


        if verbose:
            self.__Debug_state(players, self.game_state, main, players_input, game_history_file, player_is_NN=NN_player)


        
        while play_action != -1:
            action_turns += 1
            players[main]["actions"] -= 1

            self.card_effects.play_card(play_action, self.game_state, players[main], players_input[main],  players[adversary], players_input[adversary])
            players[main] = self.__Update_victory_points(self.game_state, players[main])
            

            actions = self.__get_actions(players[main])
            self.game_state["Unique_actions"] = "take_action"
            play_action = int(players_input[main].choose_action(actions, copy.deepcopy(self.game_state)))


            card_idx = sm.card_idx_2_set_idx(play_action, self.game_state)
            if card_idx != -1:
                card_obj = self.game_state["dominion_cards"][card_idx]
            else:
                card_obj = "None"

            if verbose:
                game_history_file.write(f"action possibilites: {actions} \n")

                if main == 0: # We only want to log the cards bought by player 0 (Sarsa trained player)
                    NN_return = players_input[main].NN_get_expected_return(self.game_state, actions)
                    game_history_file.write(f"expected returns: {NN_return}\n")

                game_history_file.write(f"Chosen action: {play_action} : {card_obj}\n")
                self.__Debug_state(players, self.game_state, main, players_input, game_history_file, player_is_NN=NN_player)
        


        players[main]["actions"] = 0


        return action_turns

    def __buy_phase(self, players, players_input,NN_player, main, adversary, game_history_file, verbose):
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

        players[main] = self.__update_player_treasure_value(players[main], self.game_state, players_input[main])
        self.game_state = sm.merge_game_player_state(self.game_state, players[main], players[adversary])
        self.game_state["Unique_actions"] = "buy"
        players[main]["buys"] = 1




        list_of_actions_buy = self.__get_buys(players[main], self.game_state)

        if verbose: 
            game_history_file.write(f"buy possibilites: {list_of_actions_buy} \n")

            if main == 0: # We only want to log the cards bought by player 0 (Sarsa trained player)
                NN_return = players_input[main].NN_get_expected_return(self.game_state, list_of_actions_buy)
                game_history_file.write(f"expected returns: {NN_return}\n")


        # Choose a buy
        buy_action = players_input[main].choose_action(list_of_actions_buy, copy.deepcopy(self.game_state))


            

        card_idx = sm.card_idx_2_set_idx(buy_action, self.game_state)
        if card_idx != -1:
            card_obj = self.game_state["dominion_cards"][card_idx]
        else:
            card_obj = "None"

        if verbose: 
            game_history_file.write(f"Chosen buy action: {buy_action} : {card_obj}\n")
            self.__Debug_state(players, self.game_state, main, players_input, game_history_file, player_is_NN=NN_player)     


        while buy_action != -1:
            buy_actions_amount += 1
            players[main]["buys"] -= 1

            players[main], self.game_state = self.__buy_card_from_supply(player_state=players[main], game_state=self.game_state, card_idx=buy_action)
            self.game_state = sm.merge_game_player_state(self.game_state, players[main])

            if main == 0: # We only want to log the cards bought by player 0 (Sarsa trained player)
                card_set_index = sm.card_idx_2_set_idx(buy_action, self.game_state)
                self.player0_bought_cards[card_set_index] += 1
            


            players[main] = self.__Update_victory_points(self.game_state, players[main])
            
            
            self.game_state = sm.merge_game_player_state(self.game_state, players[main], players[adversary])
            

            
            list_of_actions_buy = self.__get_buys(players[main], self.game_state)
            self.game_state["Unique_actions"] = "buy"
            buy_action = players_input[main].choose_action(list_of_actions_buy, copy.deepcopy(self.game_state))
            card_idx = sm.card_idx_2_set_idx(buy_action, self.game_state)
            if card_idx != -1:
                card_obj = self.game_state["dominion_cards"][card_idx]
            else:
                card_obj = "None"
            
            if verbose: 
                game_history_file.write(f"buy possibilites: {list_of_actions_buy} \n")
                if main == 0: # We only want to log the cards bought by player 0 (Sarsa trained player)
                    NN_return = players_input[main].NN_get_expected_return(self.game_state, list_of_actions_buy)
                    game_history_file.write(f"expected returns: {NN_return}\n")
                    
                game_history_file.write(f"Chosen buy action: {buy_action} : {card_obj}\n")
                self.__Debug_state(players, self.game_state, main, players_input, game_history_file, player_is_NN=NN_player)


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


    def __Debug_state(self, players, game_state, main_player, players_input, game_history_file, gain_card = False, player_is_NN = False):


            game_state_temp = copy.deepcopy(game_state)
            self.game_state = self.__Update_victory_points(game_state_temp, players[main_player])
  

            player_state = sm.get_player_state_from_game_state(game_state_temp)

            advesary = main_player*(-1) + 1


            game_history_file.write("\n"*2)
            game_history_file.write(f" --- GAME STATE ---\n")

            unique_actions_state = game_state["Unique_actions"]
            game_history_file.write(f"Action that is taken: {unique_actions_state} \n")


            cards_in_hand = player_state["cards_in_hand"]
            game_history_file.write(f"cards in hand: {cards_in_hand} \n")

            cards_in_discard = player_state["cards_in_discard"]
            game_history_file.write(f"cards in discard: {cards_in_discard} \n")

            cards_in_deck = player_state["cards_in_deck"]
            game_history_file.write(f"cards in deck: {cards_in_deck} \n")

            known_top_deck_cards = self.game_state["known_cards_top_deck"]
            game_history_file.write(f"card top of deck: {known_top_deck_cards} \n")

            played_cards = player_state["played_cards"]
            game_history_file.write(f"played cards: {played_cards} \n")

            owned_cards = player_state["owned_cards"]
            length_owned_cards = len(player_state["owned_cards"])
            game_history_file.write(f"owned cards: {owned_cards} -> size -> {length_owned_cards} \n")

            actions = player_state["actions"]
            game_history_file.write(f"action values: {actions} \n")

            buys = player_state["buys"]
            game_history_file.write(f"buys: {buys} \n")

            value = player_state["value"]
            game_history_file.write(f"player value: {value} \n")

            supply_amount = self.game_state["supply_amount"]
            game_history_file.write(f"card supply: {supply_amount} \n")

            cards_in_hand_adv = players[advesary]["cards_in_hand"]
            game_history_file.write(f"adversary cards in hand: {cards_in_hand_adv} \n")

            adv_discard = players[advesary]["cards_in_discard"]
            game_history_file.write(f"adversary cards in discard: {adv_discard} \n")

            adv_owned = players[advesary]["owned_cards"]
            length_owned_cards = len(players[advesary]["owned_cards"])
            game_history_file.write(f"adversary owned cards: {adv_owned} -> size -> {length_owned_cards} \n")


            adv_victory_points = game_state_temp["adv_Victory_points"]
            game_history_file.write(f"adversary victory points: {adv_victory_points}\n")


            victory_points = player_state["Victory_points"]
            game_history_file.write(f"player victory points: {victory_points} \n")

            if player_is_NN:
                game_history_file.write("\n"*1)
                reward = players_input[main_player].latest_reward
                game_history_file.write(f"Reward from previous game state: \n{reward} \n")
                game_history_file.write(f"sum of rewards: {np.sum(reward)} \n\n")

                latest_action = players_input[main_player].latest_action
                action_type = players_input[main_player].latest_action_type
                expected_reward_update = players_input[main_player].latest_updated_expected_return
                expected_reward_desired = players_input[main_player].latest_desired_expected_return

                game_history_file.write(f"action type: {action_type} - action {latest_action}\n")
                game_history_file.write(f"Learning step: {expected_reward_update}\n")
                game_history_file.write(f"desired expected reward: {expected_reward_desired}\n\n")




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



Dominion_game = Dominion()
Dominion_game.card_set = make_card_set([16, 11, 8, 25, 29, 14, 23, 10, 22, 15])

player_random1 = random_player(player_name="Ogus_bogus_man")

# Sarsa_player = Deep_SARSA(player_name="Deep_sarsa")
# sarsa_player2 = Deep_SARSA(player_name="Deep_sarsa_2")

Q_learning_player = Deep_Q_learning(player_name="Deep_Q_learning")

# DES_ai = Deep_expected_sarsa(player_name="Deep_expected_sarsa")


# Deep sarsa 2 is trained to get provinces after 20 turns

greedy_test_player = greedy_NN(player_name="Greedy_NN")
# greedy_test_player.load_NN_from_file("NN_models/Deep_sarsa_2_model.keras")
Dominion_game.set_players(Q_learning_player, greedy_test_player) # Training the first player, testing with the second player


trained_player_wins_in_row = 0
test_player_wins_in_row = 0

win_streak_limit = 15

for i in range(100000):
    print(f"Game: {i}")


    if trained_player_wins_in_row >= win_streak_limit:
        # All learned parameters from the trained player, is passed to the test player
        greedy_test_player.model.set_weights(Q_learning_player.model.get_weights())
        Dominion_game.set_player2test(greedy_test_player)
        print("Training player wins in a row: ", trained_player_wins_in_row)
        print("Giving weights of trained player to test player.")
        trained_player_wins_in_row = 0
        test_player_wins_in_row = 0


    elif test_player_wins_in_row >= win_streak_limit:
        # All learned parameters from the test player, is passed to the trained player
        Q_learning_player.model.set_weights(greedy_test_player.model.get_weights())
        Dominion_game.set_player2train(Q_learning_player)
        print("test player wins in a row: ", test_player_wins_in_row)
        print("Giving weights of test player to training player.")
        trained_player_wins_in_row = 0
        test_player_wins_in_row = 0



    Dominion_game.testplayer_province_boosted = True
    Dominion_game.player1.greedy_mode = False
    # Dominion_game.set_player2test(Sarsa_player)
    index_player_won = Dominion_game.play_loop_AI(f"trainer_game_{i}",player_0_is_NN=True, player_1_is_NN=False, verbose=True)

    if index_player_won == 0:
        print("Trained player won!")
        trained_player_wins_in_row += 1
        test_player_wins_in_row = 0

    elif index_player_won == 1:
        print("Test player won!")
        trained_player_wins_in_row = 0
        test_player_wins_in_row += 1

    else:
        print("Draw!")


    Dominion_game.testplayer_province_boosted = False
    Dominion_game.player1.greedy_mode = True
    index_player_won = Dominion_game.play_loop_AI(f"test_game_{i}",player_0_is_NN=True, player_1_is_NN=False, verbose=True)


    if index_player_won == 0:
        print("Trained player won!")
    elif index_player_won == 1:
        print("Test player won!")
    else:
        print("Draw!")

    print("\n")







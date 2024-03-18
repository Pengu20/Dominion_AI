
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



    def __initialize_game_state(self):
        # This datatype is the object that holds all information regarding the game
        state = {
            # ----- GAME END -----
        "Player_won": -1,


            # ----- SUPPLY RELATED -----
        "dominion_cards": self.deck.get_card_set(),
        "supply_amount": np.append(self.standard_supply, np.ones(10) * 10), # 10 kingdom cards with 10 supply each


            # ----- WHAT THE ACTION MEANS -----
        "Unique_actions": None, # This is the unique actions that the player can do. This is based on the cards in the players hand


            # ----- MAIN PLAYER -----
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


    def play_loop_AI(self, player1, player2):
        ''' [Summary]
        This function is the main loop of the game. It will keep running until the game is over.

        ARGS:
            player1 player: This type object must be capable of choosing which cards to play and when.
            player2 player: This is also a player type 
        '''


        game_history_file = open("game_history.txt", "w")

        game_history_file.write(" --------- CARDS IN THIS GAME --------- \n")
        dominion_cards = self.game_state["dominion_cards"]
        game_history_file.write(f"Game cards: {dominion_cards}")
        game_history_file.write("\n"*4)



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
        
        player_turns = 0


        turns_all_players = 0

        while self.game_state["Player_won"] == -1:
            
            # --------- ACTION PHASE ---------
            game_history_file.write("\n"*3)
            game_history_file.write(f" ---------- Turn: {player_turns} ---------- \n")
            if turns_all_players % 2 == 0:
                print(" ---------- Turn: ", player_turns, " ---------- ")


            game_history_file.write(f"Player: {main_player} \n")
            cards_in_hand = players[main_player]["cards_in_hand"]
            game_history_file.write(f"cards in hand: {cards_in_hand} \n")



            game_history_file.write(f"----------- ACTION PHASE ----------- \n")


            main = int(main_player)
            advesary = int(main_player*(-1) + 1)


            # Get all possible actions
            actions = self.__get_actions(players[main_player])
            game_history_file.write(f"action possibilites: {actions} \n")
  

            # Choose action
            self.game_state = sm.merge_game_player_state(self.game_state, players[main_player], players[main_player*(-1) + 1])
            self.game_state["Unique_actions"] = "action"
            action = players_input[main_player].choose_action(actions, self.game_state)
            card_idx = sm.card_idx_2_set_idx(action, self.game_state)
            card_obj = self.game_state["dominion_cards"][card_idx]
            
            game_history_file.write(f"Chosen action: {action} : is the same as card {card_obj}\n")

            
            while action != -1:
                idx = sm.card_idx_2_set_idx(action, self.game_state)
                index = sm.card_idx_2_set_idx(self.game_state["dominion_cards"][idx], self.game_state)

                Dominion_cards = self.game_state["dominion_cards"][index]
                game_history_file.write(f"Played card: {Dominion_cards} \n")


                card_effects().play_card(action, self.game_state, players[main], players_input[main],  players[advesary], players_input[advesary])
                self.__Debug_state(players, main_player, players_input, game_history_file)


                action = player1.choose_action(actions, self.game_state)
                game_history_file.write(f"Chosen action: {action} \n")





            # --------- BUY PHASE ---------
            game_history_file.write("\n"*4)
            game_history_file.write(f"----------- BUY PHASE ----------- \n")



            self.game_state = sm.merge_game_player_state(self.game_state, players[main_player], players[main_player*(-1) + 1])



            self.game_state["Unique_actions"] = "buy"
            players[main_player]["buys"] += 1
            self.game_state = self.__update_player_treasure_value(players[main_player], self.game_state, players[main_player])


            buy_actions = self.__get_buys(players[main_player], self.game_state)
            game_history_file.write(f"buy possibilites: {buy_actions} \n")



            # Choose a buy
            action = player1.choose_action(buy_actions, self.game_state)
            game_history_file.write(f"Chosen action: {action} \n")


            while action != -1:
                players[main_player], self.game_state = self.__buy_card_from_supply(player_state=players[main_player], game_state=self.game_state, card_idx=action)
                self.__Debug_state(players, main_player, players_input, game_history_file)
                

                
                buy_actions = self.__get_buys(players[main_player], self.game_state)
                action = player1.choose_action(buy_actions, self.game_state)
                game_history_file.write(f"Chosen action: {action} \n")



            # ---------- HIDDEN PHASE: Update victory points check if won ---------
            self.game_state = self.__Update_victory_points(self.game_state, players[main_player])

            
            if players[main_player]["Victory_points"] >= 4:
                self.game_state["Player_won"] = main_player
                game_history_file.write(f"Player {main_player} won the game! \n")
                break




            # Reset and draw new hand
            self.game_state = sm.put_player_state_adv_state(self.game_state, players[main_player])    
            turns_all_players += 1

            if turns_all_players % 2 == 0:
                player_turns += 1

            main_player += 1
            if main_player >= players_amount:
                main_player = 0
                
            


        return 0 # return index of who wins.
    



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
                card_effects().play_card(card, game_state, player_state, card2played_cards=False)
                
        return game_state


    def __Debug_state(self, players, main_player, players_input, game_history_file, gain_card = False):
            self.game_state = self.__Update_victory_points(self.game_state, players[main_player])

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




    def __debug_buy_action(self, card_bought, players, main_player, players_input, game_history_file):
            game_history_file.write("----------- CARD bought -----------")
            index = sm.card_idx_2_set_idx(card_bought, self.game_state)

            Dominion_cards = self.game_state["dominion_cards"][index]
            game_history_file.write(f"Card: {Dominion_cards}")


            game_history_file.write(f"------------------- BEFORE -------------------")

            cards_in_discard = players[main_player]["cards_in_discard"]
            game_history_file.write(f"cards in discard: {cards_in_discard} \n")

            owned_cards = players[main_player]["owned_cards"]
            length_owned_cards = len(players[main_player]["owned_cards"])
            game_history_file.write(f"owned cards: {owned_cards} -> size -> {length_owned_cards}")
            
            buys = players[main_player]["buys"]
            game_history_file.write(f"buys: {buys} \n")

            value = players[main_player]["value"]
            game_history_file.write(f"player value: {value} \n")


            supply_amount = self.game_state["supply_amount"]
            game_history_file.write(f"card supply: {supply_amount} \n")
       



            main = int(main_player)
            advesary = int(main_player*(-1) + 1)
            card_effects().play_card(card_bought, self.game_state, players[main], players_input[main],  players[advesary], players_input[advesary])





            game_history_file.write(f"------------------- AFTER -------------------")
            cards_in_discard = players[main_player]["cards_in_discard"]
            game_history_file.write(f"cards in discard: {cards_in_discard} \n")


            owned_cards = players[main_player]["owned_cards"]
            length_owned_cards = len(players[main_player]["owned_cards"])
            game_history_file.write(f"owned cards: {owned_cards} -> size -> {length_owned_cards}")
            

            buys = players[main_player]["buys"]
            game_history_file.write(f"buys: {buys} \n")


            value = players[main_player]["value"]
            game_history_file.write(f"player value: {value} \n")


            supply_amount = self.game_state["supply_amount"]
            game_history_file.write(f"card supply: {supply_amount} \n")



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


        # Remove one buy power from player
        player_state["buys"] -= 1



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

        player_state["value"] = 0
        cards_in_hand = player_state["cards_in_hand"]

        treasure_cards = np.argwhere(np.isin(cards_in_hand, self.Treasure_card_index)).flatten()

        silver_played = False
        for index in treasure_cards:
            game_state, player_state, adv_state = card_effects().play_card(cards_in_hand[index].astype(int), game_state, player_state, player_input, card2played_cards=False)
            
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




Dominion_game = Dominion()

Dominion_game.play_loop_AI(random_player(), random_player())
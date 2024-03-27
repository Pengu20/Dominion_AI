'''[SUMMARY]
This python file holds a list of all the unique actions that is available in the game of dominion,
The functions have been moved to this file for easier access for game simulation, while the game is being played by and AI.
'''


from standard_cards import standard_set
from cards_base_ed2 import kingdom_cards_ed2_base
from itertools import combinations
import numpy as np
import state_manipulator as sm


class unique_action:

    def __init__(self) -> None:
        self.unique_action_dict = self.add_unique_actions_to_dict()
        self.generate_card_set()


    def __get_non_action_cards(self):
        '''
        Returns the index of the non action cards
        '''
        non_action_cards = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse", "Gardens"]
        non_action_index = [card[1] for card in self.card_list if card[0] in non_action_cards]
        return non_action_index


    def __get_treasures(self):
        '''
        Returns the index of the treasure cards
        '''
        treasures = ["Copper", "Silver", "Gold"]
        treasures_index = [card[1] for card in self.card_list if card[0] in treasures]

        return treasures_index
    

    def generate_card_set(self):
        self.card_list = []

        for card in standard_set:
            self.card_list.append(card)

        for card in kingdom_cards_ed2_base:
            self.card_list.append(card)


        return self.card_list




    def add_unique_actions_to_dict(self):
        card_effect_dict = {
            "discard_and_draw": self.discard_and_draw,
            "trash_cards_4": self.trash_4_cards,
            "gain_card_4": self.gain_card_4,
            "adv_discard_down_to_3_cards": self.discard_down_to_3_cards,
            "trash_cards_n_from_hand": self.trash_n_cards_from_hand,
            "gain_card_n": self.gain_card_n,
            "play_card_twice": self.play_card_twice,
            "LIBRARY_skip_action_card": self.lib_skip_action_card,
            "upgrade_treasure": self.upgrade_treasure,
            "discard_pile2deck": self.discard_pile2deck,
            "discard_cards_equal_empty_piles": self.discard_cards_equal_empty_piles,
            "discard_trash_keep_in_deck": self.discard_trash_keep_in_deck,
            "order_cards_2": self.order_cards_2,
            "supply_2_hand_5": self.supply_2_hand_5,
            "put_card_on_deck": self.put_card_on_deck,

            # ----- KINGDOM CARDS base 2. Edition -----
        }

        return card_effect_dict


    def do_unique_action(self, unique_action, game_state, player_state, player_input, adv_state, adv_input):

        game_state["Unique_actions"] = str(unique_action)
        action = str(unique_action)
        self.unique_action_dict[action](game_state, player_state, player_input, adv_state, adv_input)


        game_state["Unique_actions"] = None

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    

    def discard_and_draw(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Discard n cards from hand and draw n cards from deck
        """
        cards_in_hand = player_state["cards_in_hand"]

        # We need every combination of cards to discard. Then draw that many.

        all_combinations = []
        for i in range(1, len(cards_in_hand)+1):
            for combination in set(list(combinations(cards_in_hand, i))):
                all_combinations.append(combination)

        
        actions_list = np.arange(len(all_combinations))
        actions_list = np.append(actions_list, -1) # Append the ability to do nothing


        action = player_input.choose_action(actions_list, game_state)
    
        if action == -1:
            return game_state, player_state, adv_state
        else:

            # Draw cards
            player_state = sm.draw_n_cards_from_deck(player_state, len(all_combinations[action])) # Draw equal to amount of actions that is not terminate action


            # Put cards into discard pile
            for card in all_combinations[action]:
                player_state = sm.put_card_from_hand_to_discard(player_state, card)

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    

    def trash_4_cards(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Trash up to 4 cards from hand
        """
        cards_in_hand = player_state["cards_in_hand"]
        # We need every combination of cards to discard. Then draw that many.

        all_combinations = []
        for i in range(1, 4+1):
            for combination in set(list(combinations(cards_in_hand, i))):
                all_combinations.append(combination)
        
        
        actions_list = np.arange(len(all_combinations))
        actions_list = np.append(actions_list, -1) # Append the ability to do nothing


        action = player_input.choose_action(actions_list, game_state)

        if action == -1:
            return game_state, player_state, adv_state
        else:

            # trash cards
            for card in all_combinations[action]:
                player_state = sm.trash_card(player_state, card)
 

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def gain_card_4(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Gain a card costing up to 4
        """
        card_set = game_state["dominion_cards"]

        Available_cards = []
        for card in card_set:
            set_index = sm.card_idx_2_set_idx(int(card[1]), game_state=game_state)
            if int(card[2]) <= 4 and int(game_state["supply_amount"][set_index]) > 0:
                Available_cards.append(card)
        
        actions_list = np.arange(len(Available_cards))
        actions_list = np.append(actions_list, -1) # Add the ability to terminate

        action = player_input.choose_action(actions_list, game_state)
        if action == -1:
            return game_state, player_state, adv_state
        else:
            game_state = sm.supply2discard(game_state, player_state, int(Available_cards[action][1]))
            player_state = sm.get_player_state_from_game_state(game_state)
        
 
        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state



    def discard_down_to_3_cards(self, game_state, player_state, player_input, adv_state, adv_input):
        # Adversary player discards cards down to 3 cards
        # We need every combination of cards to discard. Then draw that many.

        all_combinations = []
        # If adversary has 3 or less cards, then they do not need to discard.
        if adv_state["cards_in_hand"].shape[0] <= 3:
            return game_state, player_state, adv_state
        

        for combination in set(list(combinations(adv_state["cards_in_hand"], adv_state["cards_in_hand"].shape[0]-3))):
            all_combinations.append(combination)


        actions_list = np.arange(len(all_combinations))
        game_state["Unique_actions"] = "discard_down_to_3_cards"

        action = adv_input.choose_action(actions_list, game_state)

    
        # Put cards into discard pile
        for card in all_combinations[action]:
            adv_state = sm.put_card_from_hand_to_discard(adv_state, card)

 
        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    


    def trash_n_cards_from_hand(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Trash up to n cards from hand
        """
        game_state["adv_cards_in_hand"] = 3


        cards_in_hand = player_state["cards_in_hand"]
        # We need every combination of cards to discard. Then draw that many.


        trash_cards_amount = game_state["Unique_actions_parameter"]

        all_combinations = []
        for i in range(1, trash_cards_amount+1):
            for combination in set(list(combinations(cards_in_hand, i))):
                all_combinations.append(combination)
        
        
        actions_list = np.arange(len(all_combinations))
        if len(actions_list) == 0:
            sm.merge_game_player_state(game_state, player_state)
            return game_state, player_state, adv_state
        
        
        action = player_input.choose_action(actions_list, game_state)

        # trash cards
        for card in all_combinations[action]:
            player_state = sm.trash_card(player_state, card)


        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    



    def gain_card_n(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Gain a card costing up to n based on unique action parameter
        """
        card_set = game_state["dominion_cards"]
        card_cost_up_to = game_state["Unique_actions_parameter"]

        Available_cards = []
        for card in card_set:
            set_index = sm.card_idx_2_set_idx(int(card[1]), game_state=game_state)
            if int(card[2]) <= card_cost_up_to and int(game_state["supply_amount"][set_index]) > 0:
                Available_cards.append(card)

        
        actions_list = np.arange(len(Available_cards))
        if len(actions_list) == 0:
            sm.merge_game_player_state(game_state, player_state)
            return game_state, player_state, adv_state


        action = player_input.choose_action(actions_list, game_state)

        game_state = sm.supply2discard(game_state, player_state, int(Available_cards[action][1]))
        player_state = sm.get_player_state_from_game_state(game_state)
    
 
        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def play_card_twice(self, game_state, player_state, player_input, adv_state, adv_input):
        # Choose an action card from your hand
        

        cards_in_hand = player_state["cards_in_hand"]
        actions_list = []
        list_non_action_cards = self.__get_non_action_cards()

        for card in cards_in_hand:
                if card not in list_non_action_cards:
                    actions_list.append(card)

        

        if len(actions_list) == 0:
            game_state["Unique_actions_parameter"] = -1
            return game_state, player_state, adv_state


        action = player_input.choose_action(actions_list, game_state)
        choosen_card = int(action)


        # Play the card twice

        game_state["Unique_actions_parameter"] = choosen_card


    
        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    

    def lib_skip_action_card(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Skip action card if drawn. is known to only be accessed as part of library card.
        """


        draw_card = int(player_state["cards_in_hand"][-1])

        game_state["Unique_actions"] = "LIBRARY: skip_action_card"


        actions_list = np.array([0, 1])

        action = player_input.choose_action(actions_list, game_state)


        if action == 0:
            player_state = sm.hand_2_played_cards(player_state, draw_card)


        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    


    def upgrade_treasure(self, game_state, player_state, player_input, adv_state, adv_input):

        game_state["Unique_actions"] = "upgrade treasure"

        # Get all treasures in hand
        cards_in_hand = player_state["cards_in_hand"]

        treasures_index = self.__get_treasures()
        

        actions = [treasures for treasures in cards_in_hand if treasures in treasures_index]


        # Choose which treasure to upgrade
        if len(actions) != 0:

            action = player_input.choose_action(actions, game_state)

            trash_card = int(action)

            player_state = sm.trash_card(player_state, trash_card)

            # Gain a treasure card costing up to 3 more
            # If treasure card is gold, then do nothing.
            # If there is no more in supply, then do not get card
            if trash_card != 2 and game_state["supply_amount"][trash_card + 1] > 0:
                    sm.get_card2hand(player_state, trash_card + 1)
                    


        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state




    def discard_pile2deck(self, game_state, player_state, player_input, adv_state, adv_input):

        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 1

        # Look through discard pile
        game_state["Unique_actions"] = "discard_pile2deck"
        actions = player_state["cards_in_discard"]
        actions = np.append(actions, -1) # Add the ability to terminate

        action = int(player_input.choose_action(actions, game_state))

        if action != -1:
            player_state = sm.discard_to_deck(game_state, player_state, action)



        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state



    def discard_cards_equal_empty_piles(self, game_state, player_state, player_input, adv_state, adv_input):
        """
        Discard n cards from hand
        """


        empty_piles = len(np.where(game_state["supply_amount"] == 0)[0])


        if empty_piles > 0:
            game_state["Unique_actions"] = "discard_n_cards"
            all_combinations = []
            actions_list = np.arange(len(player_state["cards_in_hand"]))

            # Can maximally discard as many cards as cards in hand
            for combination in set(list(combinations(player_state["cards_in_hand"], min(empty_piles, len(player_state["cards_in_hand"]))))):
                all_combinations.append(combination)
            

            actions_list = np.arange(len(all_combinations))

            action = player_input.choose_action(actions_list, game_state)

            # Put cards into discard pile
            for card in all_combinations[action]:
                player_state = sm.put_card_from_hand_to_discard(player_state, card)



        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state



    def discard_trash_keep_in_deck(self, game_state, player_state, player_input, adv_state, adv_input):


        # Discard card: 0
        # trash card = 1
        # keep in deck: 2
        card_list_action = [0,1,2]

        card_drawn = game_state["Unique_actions_parameter"]
            
        game_state["Unique_actions"] = "discard_trash_keep_in_deck"
        action = player_input.choose_action(card_list_action, game_state)

        if action == 0:
            print("Card discard")
            player_state = sm.put_card_from_hand_to_discard(player_state, card_drawn)
        elif action == 1:
            print("Card trashed")
            player_state = sm.trash_card(player_state, card_drawn)
        else:
            print("Card kept in deck")
            player_state = sm.hand2deck(game_state, player_state, card_drawn)

        game_state["Unique_actions_parameter"] = action
                
        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state  
    

    def order_cards_2(self, game_state, player_state, player_input, adv_state, adv_input):

        # The two options to put back in deck [0,1] or [1,0]

        order_action = [0, 1]
        game_state = sm.merge_game_player_state(game_state, player_state)
        action = player_input.choose_action(order_action, game_state)

        if action == 0: # Flip the order of the two first cards in the top deck
            card1 = player_state["known_cards_top_deck"][-1]
            card2 = player_state["known_cards_top_deck"][-2]

            player_state["known_cards_top_deck"] = np.delete(player_state["known_cards_top_deck"], -1)
            player_state["known_cards_top_deck"] = np.delete(player_state["known_cards_top_deck"], -1)

            player_state["known_cards_top_deck"] = np.append(player_state["known_cards_top_deck"], card1)
            player_state["known_cards_top_deck"] = np.append(player_state["known_cards_top_deck"], card2)

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state  
    


    def supply_2_hand_5(self, game_state, player_state, player_input, adv_state, adv_input):

    
        card_set = game_state["dominion_cards"]

        Available_cards = []
        for card in card_set:
            set_index = sm.card_idx_2_set_idx(int(card[1]), game_state=game_state)
            if int(card[2]) <= 5 and int(game_state["supply_amount"][set_index]) > 0:
                Available_cards.append(card[1])
        
        

        chosen_card = player_input.choose_action(Available_cards, game_state)

        # Gain card from supply to hand, and remove from supply
        player_state = sm.get_card2hand(player_state, int(chosen_card))
        set_index = sm.card_idx_2_set_idx(int(chosen_card), game_state=game_state)
        game_state["supply_amount"][set_index] = int(game_state["supply_amount"][set_index]) - 1

    
    
    def put_card_on_deck(self, game_state, player_state, player_input, adv_state, adv_input):
        card_on_deck = player_state["cards_in_hand"]

        chosen_card = player_input.choose_action(card_on_deck, game_state)

        player_state = sm.hand2deck(game_state, player_state, int(chosen_card))






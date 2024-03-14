
"""
It is assumed that all card effects only change the state of the game

"""


from standard_cards import standard_set
from cards_base_ed2 import kingdom_cards_ed2_base
from itertools import combinations
import numpy as np
import state_manipulator as sm

class card_effects():
    def __init__(self) -> None:
        self.card_effect_dict = self.Add_cards_to_function_dict()
        self.generate_card_set()



    def generate_card_set(self):
        self.card_list = []

        for card in standard_set:
            self.card_list.append(card)

        for card in kingdom_cards_ed2_base:
            self.card_list.append(card)


        return self.card_list



    def Add_cards_to_function_dict(self):

        card_effect_dict = {
            "Copper": self.copper,
            "Silver": self.silver,
            "Gold": self.gold,
            "Estate": self.estate,
            "Duchy": self.duchy,
            "Province": self.province,
            "Curse": self.curse,

            
            "Cellar": self.cellar,
            "Chapel": self.chapel,
            "Moat": self.moat,
            "Village": self.village,
            "Workshop": self.workshop,
            # ----- KINGDOM CARDS base 2. Edition -----
        }





        '''


                

                
                "Bureaucrat": self.bureaucrat,
                "Gardens": self.gardens,
                "Militia": self.militia,
                "Moneylender": self.moneylender,
                "Remodel": self.remodel,
                "Smithy": self.smithy,
                "Throne room": self.throne_room,
                "Council room": self.council_room,
                "Festival": self.festival,
                "Laboratory": self.laboratory,
                "Library": self.library,
                "Market": self.market,
                "Mine": self.mine,
                "Witch": self.witch,
                "Harbinger": self.harbinger,
                "Merchant": self.merchant,
                "Vassal": self.vassal,
                "Poacher": self.poacher,
                "Bandit": self.bandit,
                "Sentry": self.sentry,
                "Artisan": self.artisan
        '''


        return card_effect_dict
    


    def play_card(self, card_idx, game_state, player_state, player_input, adv_state=None, adv_input=None, card2played_cards=True):
        
        ''' [Summary]
        This function will play a card from the players hand. 
        It will then apply the card effect to the game state and the player state.

        
        '''
        if card2played_cards:
            player_state = sm.hand_2_played_cards(player_state, card_idx)

        game_state, player_state = self.card_effect_dict[self.card_list[card_idx][0]](game_state, player_state, player_input, adv_state=None, adv_input=None)
        
        if card2played_cards:
            player_state = sm.played_cards_2_discard_pile(player_state)

        return game_state, player_state 
    



    def copper(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["value"] += 1
        return game_state, player_state
    
    def silver(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["value"] += 2
        return game_state, player_state
    
    def gold(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["value"] += 3
        return game_state, player_state
    
    def estate(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += 1
        return game_state, player_state
    
    def duchy(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += 3
        return game_state, player_state
    
    def province(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += 6
        return game_state, player_state
    
    def curse(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += -1
        return game_state, player_state
    

    def cellar(self, game_state, player_state, player_input, adv_state, adv_input):
        game_state["Unique_actions"] = "discard_and_draw"
        cards_in_hand = player_state["cards_in_hand"]

        # We need every combination of cards to discard. Then draw that many.

        all_combinations = []
        for i in range(1, len(cards_in_hand)+1):
            for combination in set(list(combinations(cards_in_hand, i))):
                all_combinations.append(combination)

        all_combinations.append(-1) # Append the ability to do nothing
        actions_list = np.arange(len(all_combinations))

        action = player_input.choose_action(actions_list, game_state)
    
        if action == -1:
            return game_state, player_state
        else:

            # Draw cards
            player_state = sm.draw_n_cards_from_deck(player_state, len(all_combinations[action]))


            # Put cards into discard pile
            for card in all_combinations[action]:
                player_state = sm.put_card_from_hand_to_discard(player_state, card)

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state


    def chapel(self, game_state, player_state, player_input, adv_state, adv_input):
        game_state["Unique_actions"] = "trash_n_cards"
        cards_in_hand = player_state["cards_in_hand"]

        # We need every combination of cards to discard. Then draw that many.

        all_combinations = []
        for i in range(1, 4+1):
            for combination in set(list(combinations(cards_in_hand, i))):
                all_combinations.append(combination)
        
        all_combinations.append(-1) # Append the ability to do nothing
        actions_list = np.arange(len(all_combinations))


        action = player_input.choose_action(actions_list, game_state)
        print("action choosen: ", all_combinations[action])


        if action == -1:
            return game_state, player_state
        else:

            # trash cards
            for card in all_combinations[action]:
                player_state = sm.trash_card(player_state, card)
 

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state
    

    def moat(self, game_state, player_state, player_input, adv_state, adv_input):
        # Draw 2 cards
        player_state = sm.draw_n_cards_from_deck(player_state, 2)
        return game_state, player_state
    

    def village(self, game_state, player_state, player_input, adv_state, adv_input):

        # Draw a card and get 2 actions
        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 2

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state
    


    def workshop(self, game_state, player_state, player_input, adv_state, adv_input):
        '''
        Gain a card costing up to 4
        '''

        game_state["Unique_actions"] = "gain_card"
        card_set = game_state["dominion_cards"]

        Available_cards = []
        for card in card_set:
            if card[2] <= 4 and game_state["supply_amount"][card[1]] > 0:
                Available_cards.append(card)
        
        actions_list = np.arange(len(Available_cards))
        actions_list = np.append(actions_list, -1) # Add the ability to terminate

        action = player_input.choose_action(actions_list, game_state)
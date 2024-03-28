
"""
It is assumed that all card effects only change the state of the game

"""


from standard_cards import standard_set
from cards_base_ed2 import kingdom_cards_ed2_base
from itertools import combinations
import numpy as np
import state_manipulator as sm
from Unique_actions import unique_action as ua

class card_effects():
    def __init__(self) -> None:
        self.card_effect_dict = self.Add_cards_to_function_dict()
        self.generate_card_set()
        self.ua = ua() # Unique action


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
            # ----- KINGDOM CARDS base 2. Edition -----
        }


        return card_effect_dict
    


    def play_card(self, card_idx, game_state, player_state, player_input=None, adv_state=None, adv_input=None, card2played_cards=True):
        
        ''' [Summary]
        This function will play a card from the players hand. 
        It will then apply the card effect to the game state and the player state.

        
        '''
        if card2played_cards:
            player_state = sm.hand_2_played_cards(player_state, card_idx)

        game_state, player_state, adv_state = self.card_effect_dict[self.card_list[card_idx][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)
        game_state["Unique_actions"] = None


        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    



    def copper(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["value"] += 1
        return game_state, player_state, adv_state
    
    def silver(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["value"] += 2
        return game_state, player_state, adv_state
    
    def gold(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["value"] += 3
        return game_state, player_state, adv_state
    
    def estate(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += 1
        return game_state, player_state, adv_state
    
    def duchy(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += 3
        return game_state, player_state, adv_state
    
    def province(self, game_state, player_state, player_input, adv_state, adv_input):
        player_state["Victory_points"] += 6
        return game_state, player_state, adv_state
    
    def curse(self, game_state, player_state, player_input, adv_state, adv_input):

        player_state["Victory_points"] += -1
        return game_state, player_state, adv_state
    

    def cellar(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 7
        +1 action. Discard any number of cards. Draw that many
        '''
        player_state["actions"] += 1
        game_state, player_state, adv_state = self.ua.do_unique_action("discard_and_draw", game_state, player_state, player_input, adv_state, adv_input)


        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def chapel(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 8
        Trash up to 4 cards from your hand
        '''

        game_state["Unique_actions"] = "trash_cards_n_from_hand"
        game_state["Unique_actions_parameter"] = 4

        self.ua.do_unique_action("trash_cards_n_from_hand", game_state, player_state, player_input, adv_state, adv_input)
 
        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    

    def moat(self, game_state, player_state, player_input, adv_state, adv_input):
        # Draw 2 cards
        player_state = sm.draw_n_cards_from_deck(player_state, 2)
        return game_state, player_state, adv_state
    

    def village(self, game_state, player_state, player_input, adv_state, adv_input):

        # Draw a card and get 2 actions
        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 2

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    


    def workshop(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 11
        Gain a card costing up to 4
        '''

        game_state["Unique_actions"] = "gain_card_n"
        game_state["Unique_actions_parameter"] = 4

        self.ua.do_unique_action("gain_card_n", game_state, player_state, player_input, adv_state, adv_input)
 

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    



    def bureaucrat(self, game_state, player_state, player_input, adv_state, adv_input):
        '''
        Gain a silver card and put it on top of your deck
        Each other player reveals a victory card and puts it on top of their deck
        '''
        # Get silver
        game_state = sm.supply2deck(game_state, player_state, 1)
        game_state = sm.merge_game_player_state(game_state, player_state)



        return game_state, player_state, adv_state



    def gardens(self, game_state, player_state, player_input, adv_state, adv_input):
        '''
        Worth 1 victory point for every 10 cards in your deck
        '''

        player_state["Victory_points"] += int(len(player_state["owned_cards"])/10)
        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    

    def militia(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 14
        +2 coins
        Each other player discards down to 3 cards in hand
        '''
        player_state["value"] += 2


        self.ua.do_unique_action("adv_discard_down_to_3_cards", game_state, player_state, player_input, adv_state, adv_input)
 
 
        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
        
    def moneylender(self, game_state, player_state, player_input, adv_state, adv_input):
        '''
        Trash a copper from your hand. If you do, +3 coins
        '''

        if "Copper" in player_state["cards_in_hand"]:
            player_state = sm.trash_card(player_state, "Copper")
            player_state["value"] += 3
        
        return game_state, player_state, adv_state


    def remodel(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 16
        Trash a card from your hand. Gain a card costing up to 2 more than the trashed card
        '''

        cards_in_hand_before = player_state["cards_in_hand"]


        if len(cards_in_hand_before) == 0:
            game_state = sm.merge_game_player_state(game_state, player_state)
            return game_state, player_state, adv_state


        # Trash a card
        game_state["Unique_actions"] = "trash_cards_n_from_hand"
        game_state["Unique_actions_parameter"] = 1

        self.ua.do_unique_action("trash_cards_n_from_hand", game_state, player_state, player_input, adv_state, adv_input)


        cards_in_hand_after = player_state["cards_in_hand"]


        # Find the card index that was trashed
        for card in cards_in_hand_after:
            if card in cards_in_hand_before:
                #delete card and search on
                cards_in_hand_before = np.delete(cards_in_hand_before, np.where(cards_in_hand_before == card)[0][0])

        trashed_card = cards_in_hand_before[0]


        game_state["Unique_actions"] = "gain_card_n"
        # Gain card costing up to two more
        game_state["Unique_actions_parameter"] = int(self.card_list[int(trashed_card)][2]) + 2

        self.ua.do_unique_action("gain_card_n", game_state, player_state, player_input, adv_state, adv_input)



        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def smithy(self, game_state, player_state, player_input, adv_state, adv_input):
        '''
        Draw 3 cards
        '''
        player_state = sm.draw_n_cards_from_deck(player_state, 3)       
        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def throne_room(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 18 - DOES NOT FOLLOW THE MARKOV DECISION PROCESS.
        Choose an action card from your hand. Play it twice
        '''

        game_state["Unique_actions"] = "play_card_twice"
        self.ua.do_unique_action("play_card_twice", game_state, player_state, player_input, adv_state, adv_input)

        choosen_card = game_state["Unique_actions_parameter"]

        if choosen_card == -1:
            game_state = sm.merge_game_player_state(game_state, player_state)
            return game_state, player_state, adv_state
        

        player_state = sm.hand_2_played_cards(player_state, choosen_card)

        game_state, player_state, adv_state = self.card_effect_dict[self.card_list[choosen_card][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)
        game_state, player_state, adv_state = self.card_effect_dict[self.card_list[choosen_card][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)

    
        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def council_room(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 19
        Draw 4 cards. +1 buy
        Each other player draws a card
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 4)
        player_state["buys"] += 1
        game_state["adv_cards_in_hand"] += 1
        adv_state = sm.draw_n_cards_from_deck(adv_state, 1)

        return game_state, player_state, adv_state
    

    def festival(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 20
        +2 actions, +1 buy, +2 coins
        '''

        player_state["actions"] += 2
        player_state["buys"] += 1
        player_state["value"] += 2
        return game_state, player_state, adv_state


    def laboratory(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 21
        Draw 2 cards, +1 action
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 2)
        player_state["actions"] += 1
        return game_state, player_state, adv_state



    def library(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 22
        Draw until you have 7 cards in hand.
        optional to skip any action cards drawn 

        '''

        while len(player_state["cards_in_hand"]) < 7:
            # Draw card
            player_state = sm.draw_n_cards_from_deck(player_state, 1)

            # Stop if there is no more cards
            if player_state["cards_in_deck"] == 0:
                break
            

            # If card is an action card then it can be choosen to skip it
            draw_card = int(player_state["cards_in_hand"][-1])


            if draw_card in self.__get_non_action_cards():
                continue
            else:
                game_state["Unique_actions"] = "LIBRARY_skip_action_card"
                self.ua.do_unique_action("LIBRARY_skip_action_card", game_state, player_state, player_input, adv_state, adv_input)



        return game_state, player_state, adv_state


    def market(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 23
        +1 card, +1 action, +1 buy, +1 coin
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 1
        player_state["buys"] += 1
        player_state["value"] += 1
        return game_state, player_state, adv_state


    def mine(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 24
        Trash a treasure card from your hand. 
        Gain a treasure card to hand costing up to 3 more than the trashed card
        '''

        game_state["Unique_actions"] = "upgrade_treasure"
        self.ua.do_unique_action("upgrade_treasure", game_state, player_state, player_input, adv_state, adv_input)


        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state

    def witch(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 25
        +2 cards. Each other player gains a curse
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 2)
        game_state = sm.supply2discard(game_state, adv_state, 6)


        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    



    def harbinger(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 26
        +1 card, +1 action. Look through your discard pile. Put a card from it onto your deck
        '''



        game_state["Unique_actions"] = "discard_pile2deck"
        self.ua.do_unique_action("discard_pile2deck", game_state, player_state, player_input, adv_state, adv_input)


        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    

    def merchant(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 27
        +1 card, +1 action. The first time you play a silver this turn, +1 coin
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 1

        # Treasure feature is handled in the dominion game class (i know its cursed im sorry)

        sm.merge_game_player_state(game_state, player_state)

        return game_state, player_state, adv_state



    def vassal(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 28
        +2 coins. Discard the top card of your deck. If it is an action card, you may play it
        '''

        # Get 2 value
        player_state["value"] += 2

        # Draw a card
        len_hand = len(player_state["cards_in_hand"])
        
        player_state = sm.draw_n_cards_from_deck(player_state, 1)


        if len_hand < len(player_state["cards_in_hand"]):
            top_deck = int(player_state["cards_in_hand"][-1])

            # If top deck is an action card, then play it, else put it in played pile.
            if top_deck in self.__get_non_action_cards():
                player_state = sm.hand_2_played_cards(player_state, top_deck)
            else:
                player_state = sm.hand_2_played_cards(player_state, top_deck)
                game_state, player_state, adv_state = self.card_effect_dict[self.card_list[top_deck][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)


        sm.merge_game_player_state(game_state, player_state)

        return game_state, player_state, adv_state
    

    def poacher(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 29
        +1 card, +1 action, +1 coin. Discard a card per empty supply pile
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 1
        player_state["value"] += 1


        game_state["Unique_actions"] = "discard_cards_equal_empty_piles"
        self.ua.do_unique_action("discard_cards_equal_empty_piles", game_state, player_state, player_input, adv_state, adv_input)


        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state





    def bandit(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 30
        Gain a gold. Each other player reveals the top 2 cards of their deck, 
        trashes a revealed treasure other than copper, and discards the rest
        '''


        game_state = sm.supply2discard(game_state, player_state, 2)


        # Adversary player reveals top 2 cards of their deck
        # Then trashes a revealed treasure other than copper, and discards the rest
        length_adv_hand = len(adv_state["cards_in_hand"])
        adv_state = sm.draw_n_cards_from_deck(adv_state, 2)
        length_adv_hand_after = len(adv_state["cards_in_hand"])
        # If any of the two revealed cards are treasures (other than copper), then trash them

        drawn_cards = length_adv_hand_after - length_adv_hand


        for i in range(-1,-1 -drawn_cards, -1):
            adv_card = int(adv_state["cards_in_hand"][i])

            if adv_card in self.__get_treasures() and adv_card != 0:
                adv_state = sm.trash_card(adv_state, adv_card)
            else:
                adv_state = sm.put_card_from_hand_to_discard(adv_state, adv_card)

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
    


    def sentry(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 31
        +1 card, +1 action. Look at the top 2 cards of your deck. Trash and/or discard any number of them. 
        Put the rest back on top in any order
        '''
        player_state["actions"] += 1
        player_state = sm.draw_n_cards_from_deck(player_state, 1)




        # Look at the top 2 cards of your deck
        original_hand_size = len(player_state["cards_in_hand"])
        player_state = sm.draw_n_cards_from_deck(player_state, 2)
        hand_size_after = len(player_state["cards_in_hand"])
        amount_cards_drawn = hand_size_after - original_hand_size


        # Might not be possible. If so, then only draw as many cards as possible.    
        
        cards_drawn = []

        for i in range(-1, -amount_cards_drawn-1, -1):
            cards_drawn.append(int(player_state["cards_in_hand"][i]))
        

        # Discard card: 0
        # trash card = 1
        # keep in deck: 2


        # Choose what to do with the two cards
        cards_back_in_deck = 0
        for i in range(amount_cards_drawn):
            game_state["Unique_actions_parameter"] = cards_drawn[i]
            self.ua.do_unique_action("discard_trash_keep_in_deck", game_state, player_state, player_input, adv_state, adv_input)

            if game_state["Unique_actions_parameter"] == 2:
                cards_back_in_deck += 1


        # if there is more than 2 cards that must go back in the deck,
        # then choose the order of the cards
                
        if cards_back_in_deck > 1:
            self.ua.do_unique_action("order_cards_2", game_state, player_state, player_input, adv_state, adv_input)



        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state  



    def artisan(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 32
        Gain a card to your hand costing up to 5. 
        Put a card from your hand onto your deck
        '''

        # Gain a card to your hand costing up to 5
        # game_state["Unique_actions"] = "supply_2_hand_5"


        self.ua.do_unique_action("supply_2_hand_5", game_state, player_state, player_input, adv_state, adv_input)

        # Put a card from your hand onto your deck

        self.ua.do_unique_action("put_card_on_deck", game_state, player_state, player_input, adv_state, adv_input)



        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state
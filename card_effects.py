
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
            # ----- KINGDOM CARDS base 2. Edition -----
        }


        '''
                
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

        game_state, player_state, adv_state = self.card_effect_dict[self.card_list[card_idx][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)
        game_state["Unique_actions"] = None



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
            return game_state, player_state, adv_state
        else:

            # Draw cards
            player_state = sm.draw_n_cards_from_deck(player_state, len(all_combinations[action]) - 1 ) # Draw equal to amount of actions that is not terminate action


            # Put cards into discard pile
            for card in all_combinations[action]:
                player_state = sm.put_card_from_hand_to_discard(player_state, card)

        sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


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


        if action == -1:
            return game_state, player_state, adv_state
        else:

            # trash cards
            for card in all_combinations[action]:
                player_state = sm.trash_card(player_state, card)
 

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
        '''
        Gain a card costing up to 4
        '''

        game_state["Unique_actions"] = "gain_card"
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
        '''
        +2 coins
        Each other player discards down to 3 cards in hand
        '''
        player_state["value"] += 2
        game_state["adv_cards_in_hand"] -= 2


        # Adversary player discards cards down to 3 cards
                # We need every combination of cards to discard. Then draw that many.
        
        all_combinations = []

        for combination in set(list(combinations(adv_state["cards_in_hand"], adv_state["cards_in_hand"].shape[0]-3))):
            all_combinations.append(combination)


        actions_list = np.arange(len(all_combinations))
        game_state["Unique_actions"] = "discard_n_cards"

        action = adv_input.choose_action(actions_list, game_state)

    
        # Put cards into discard pile
        for card in all_combinations[action]:
            adv_state = sm.put_card_from_hand_to_discard(adv_state, card)
        print(adv_state["cards_in_hand"])



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
        '''
        Trash a card from your hand. Gain a card costing up to 2 more than the trashed card
        '''

        # Trash a card
        game_state["Unique_actions"] = "trash_card"

        cards_in_hand = player_state["cards_in_hand"]
        
        actions_list = np.arange(len(cards_in_hand))

        action = player_input.choose_action(actions_list, game_state)
        
        # Trash card and gain card costing up to 2 more
        trash_card = int(player_state["cards_in_hand"][action])
        val = int(self.card_list[trash_card][2]) + 2
        print("trashed card: ", self.card_list[trash_card])



        player_state = sm.trash_card(player_state, player_state["cards_in_hand"][action])

        card_set = game_state["dominion_cards"]

        Available_cards = []
        for card in card_set:
            set_index = sm.card_idx_2_set_idx(int(card[1]), game_state=game_state)
            if int(card[2]) <= val and int(game_state["supply_amount"][set_index]) > 0:
                Available_cards.append(card)
        
        if len(Available_cards) == 0:
            return game_state, player_state, adv_state
        

        # Choose to get a given card
        actions_list = np.arange(len(Available_cards))
        action = player_input.choose_action(actions_list, game_state)
        print("gained card: ", Available_cards[action])
        game_state = sm.supply2discard(game_state, player_state, int(Available_cards[action][1]))
        player_state = sm.get_player_state_from_game_state(game_state)

        return game_state, player_state, adv_state


    def smithy(self, game_state, player_state, player_input, adv_state, adv_input):
        '''
        Draw 3 cards
        '''
        player_state = sm.draw_n_cards_from_deck(player_state, 3)       
        game_state = sm.merge_game_player_state(game_state, player_state)
        return game_state, player_state, adv_state


    def throne_room(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 18
        Choose an action card from your hand. Play it twice
        '''
        game_state["Unique_actions"] = "play_card_twice"
        # Choose an action card from your hand
        cards_in_hand = player_state["cards_in_hand"]
        actions_list = []
        list_non_action_cards = self.__get_non_action_cards()

        for card in cards_in_hand:
                if card not in list_non_action_cards:
                    actions_list.append(card)

        action = player_input.choose_action(actions_list, game_state)
        choosen_card = int(action)

        print("Choosen action: ", self.card_list[int(action)])
        # Play the card twice

        player_state = sm.hand_2_played_cards(player_state, choosen_card)


        game_state, player_state, adv_state = self.card_effect_dict[self.card_list[choosen_card][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)
        game_state, player_state, adv_state = self.card_effect_dict[self.card_list[choosen_card][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)

        player_state = sm.played_cards_2_discard_pile(player_state)


        game_state["Unique_actions"] = None
    
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
                actions_list = np.array([0, 1])
                game_state["Unique_actions"] = "skip_action_card"

                action = player_input.choose_action(actions_list, game_state)

            
                if action == 0:
                    print("action Skipped CARD: ", self.card_list[draw_card])
                    player_state = sm.hand_2_played_cards(player_state, draw_card)
                else:
                    print("action ADDED CARD: ", self.card_list[draw_card])
                    continue



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
        # Get all treasures in hand
        cards_in_hand = player_state["cards_in_hand"]

        treasures_index = self.__get_treasures()
        

        actions = [treasures for treasures in cards_in_hand if treasures in treasures_index]


        # Choose which treasure to upgrade
        if len(actions) != 0:

            game_state["Unique_actions"] = "upgrade treasure"

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

    def witch(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 25
        +2 cards. Each other player gains a curse
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 2)
        game_state = sm.supply2discard(game_state, adv_state, 6)
        return game_state, player_state, adv_state
    



    def harbinger(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 26
        +1 card, +1 action. Look through your discard pile. Put a card from it onto your deck
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 1

        # Look through discard pile
        game_state["Unique_actions"] = "look_through_discard_pile"
        actions = player_state["cards_in_discard"]
        actions = np.append(actions, -1) # Add the ability to terminate

        action = int(player_input.choose_action(actions, game_state))
        if action == -1:
            print("NO CARD ADDED TO DECK")
        else:
            print("action: ", self.card_list[action])


        if action != -1:
            player_state = sm.discard_to_deck(game_state, player_state, action)

        return game_state, player_state, adv_state
    

    def merchant(self, game_state, player_state, player_input, adv_state, adv_input):
        ''' 27
        +1 card, +1 action. The first time you play a silver this turn, +1 coin
        '''

        player_state = sm.draw_n_cards_from_deck(player_state, 1)
        player_state["actions"] += 1

        if 1 in player_state["played_cards"]:
            player_state["value"] += 1

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
                print("Cards in hand: ", player_state["cards_in_hand"])
                print("Top deck card: ", self.card_list[top_deck])
                player_state = sm.hand_2_played_cards(player_state, top_deck)
                game_state, player_state, adv_state = self.card_effect_dict[self.card_list[top_deck][0]](game_state, player_state, player_input, adv_state=adv_state, adv_input=adv_input)



        return game_state, player_state, adv_state
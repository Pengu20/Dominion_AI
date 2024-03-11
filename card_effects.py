
"""
It is assumed that all card effects only change the state of the game

"""


from standard_cards import standard_set
from cards_base_ed2 import kingdom_cards_ed2_base
from itertools import combinations

class card_effects():
    def __init__(self) -> None:
        self.card_effect_dict = self.Add_cards_to_function_dict()


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


            # ----- KINGDOM CARDS base 2. Edition -----
        }





        '''


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
        '''


        return card_effect_dict
    


    def play_card(self, card_idx, game_state, player_state, player_input):
        return self.card_effect_dict[standard_set[card_idx][0]](game_state, player_state, player_input)
    
    def copper(self, game_state, player_state, player_input):
        player_state["value"] += 1
        return game_state, player_state
    
    def silver(self, game_state, player_state, player_input):
        player_state["value"] += 2
        return game_state, player_state
    
    def gold(self, game_state, player_state, player_input):
        player_state["value"] += 3
        return game_state, player_state
    
    def estate(self, game_state, player_state, player_input):
        player_state["Victory_points"] += 1
        return game_state, player_state
    
    def duchy(self, game_state, player_state, player_input):
        player_state["Victory_points"] += 3
        return game_state, player_state
    
    def province(self, game_state, player_state, player_input):
        player_state["Victory_points"] += 6
        return game_state, player_state
    
    def curse(self, game_state, player_state, player_input):
        player_state["Victory_points"] += -1
        return game_state, player_state
    

    def cellar(self, game_state, player_state, player_input):
        game_state["Unique_actions"] = "discard_and_draw"
        cards_in_hand = player_state["cards_in_hand"]

        # We need every combination of cards to discard. Then draw that many.
        combinations(cards_in_hand, len(cards_in_hand))


        return game_state, player_state
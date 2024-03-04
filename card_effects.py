
"""
It is assumed that all card effects only change the state of the game

"""


from standard_cards import standard_set
from cards_base_ed2 import kingdom_cards_ed2_base

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
            

            # ----- KINGDOM CARDS base 2. Edition -----
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
        }

        return card_effect_dict
    


    def play_card(self, card_idx, game_state, player_state):
        return self.card_effect_dict[standard_set[card_idx]](game_state, player_state)
    
    def copper(self, game_state, player_state):
        player_state["value"] += 1
        return game_state
    
    def silver(self, game_state, player_state):
        player_state["value"] += 2
        return game_state
    
    def gold(self, game_state, player_state):
        player_state["value"] += 3
        return game_state
    
    def estate(self, game_state, player_state):
        player_state["Victory_points"] += 1
        return game_state
    
    def duchy(self, game_state, player_state):
        player_state["Victory_points"] += 3
        return game_state
    
    def province(self, game_state, player_state):
        player_state["Victory_points"] += 6
        return game_state
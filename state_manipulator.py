import numpy as np

'''
This is a list of usefull functions for manipulating the game state and the player state. 
Both the python files card_effects.py and Dominion_game.py uses these functions.
'''


def merge_game_player_state(game_state, player_state, adversary_state=None): 
    game_state["cards_in_hand"] = player_state["cards_in_hand"]
    game_state["cards_in_deck"] = player_state["cards_in_deck"]
    game_state["cards_in_discard"] = player_state["cards_in_discard"]
    game_state["owned_cards"] = player_state["owned_cards"]
    game_state["played_cards"] = player_state["played_cards"]
    game_state["actions"] = player_state["actions"]
    game_state["buys"] = player_state["buys"]
    game_state["value"] = player_state["value"]
    game_state["Victory_points"] = player_state["Victory_points"]



    if adversary_state != None:
        game_state["adv_cards_in_hand"] = adversary_state["cards_in_hand"]
        game_state["adv_cards_in_deck"] = adversary_state["cards_in_deck"]
        game_state["adv_cards_in_discard"] = adversary_state["cards_in_discard"]
        game_state["adv_owned_cards"] = adversary_state["owned_cards"]
        game_state["adv_Victory_points"] = adversary_state["Victory_points"]
    
    return game_state


def draw_n_cards_from_deck(player_state, n):
    # Shuffle deck if necessary
    if player_state["cards_in_deck"] - n < 0:
        cards_in_discard = len(player_state["cards_in_discard"])
        player_state["cards_in_deck"] += cards_in_discard
        player_state["cards_in_discard"] == 0


    deck = get_cards_in_deck(player_state)
    draws = np.random.choice(deck, min(n, player_state["cards_in_deck"]), replace=False) # Can only draw as many cards as there is in the deck
    player_state["cards_in_deck"] -= len(draws)

    player_state["cards_in_hand"] = np.append(player_state["cards_in_hand"], draws)
    return player_state


def get_cards_in_deck(player_state):
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


def put_card_from_hand_to_discard(player_state, card):
    ''' [Summary]
    This function will put a card from the hand to the discard pile.

    ARGS:
        player_state [dict]: This is the player state object
        card [int]: This is the card that will be put into the discard pile
    '''
    player_state["cards_in_hand"] = np.delete(player_state["cards_in_hand"], np.where(player_state["cards_in_hand"] == card)[0][0])
    player_state["cards_in_discard"] = np.append(player_state["cards_in_discard"], card)
    return player_state


def trash_card(player_state, card):
    ''' [Summary]
    This function will trash a card from the hand.

    ARGS:
        player_state [dict]: This is the player state object
        card [int]: This is the card that will be trashed
    '''
    player_state["cards_in_hand"] = np.delete(player_state["cards_in_hand"], np.where(player_state["cards_in_hand"] == card)[0][0])
    player_state["owned_cards"] = np.delete(player_state["owned_cards"], np.where(player_state["owned_cards"] == card)[0][0])
    return player_state

def hand_2_played_cards(player_state, card):
    ''' [Summary]
    This function will move a card from the hand to the played cards.

    ARGS:
        player_state [dict]: This is the player state object
        card [int]: This is the card that will be moved to the played cards
    '''

    player_state["cards_in_hand"] = np.delete(player_state["cards_in_hand"], np.where(player_state["cards_in_hand"] == card)[0][0])
    player_state["played_cards"] = np.append(player_state["played_cards"], card)

    return player_state


def played_cards_2_discard_pile(player_state):
    ''' [Summary]
    This function will move all the played cards to the discard pile.

    ARGS:
        player_state [dict]: This is the player state object
    '''
    player_state["cards_in_discard"] = np.append(player_state["cards_in_discard"], player_state["played_cards"])
    player_state["played_cards"] = np.array([])
    return player_state


def get_card2hand(player_state, card):
    ''' [Summary]
    This function will move a card from the played cards to the hand.

    ARGS:
        player_state [dict]: This is the player state object
        card [int]: This is the card that will be moved to the hand
    '''
    player_state["cards_in_hand"] = np.append(player_state["cards_in_hand"], card)
    player_state["owned_cards"] = np.append(player_state["owned_cards"], card)
    return player_state

def get_card2discard(player_state, card):
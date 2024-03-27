import numpy as np

'''
This is a list of usefull functions for manipulating the game state and the player state. 
Both the python files card_effects.py and Dominion_game.py uses these functions.
'''


def merge_game_player_state(game_state, player_state, adversary_state=None): 
    game_state["cards_in_hand"] = player_state["cards_in_hand"]
    game_state["cards_in_deck"] = player_state["cards_in_deck"]
    game_state["known_cards_top_deck"] = player_state["known_cards_top_deck"]
    game_state["cards_in_discard"] = player_state["cards_in_discard"]
    game_state["owned_cards"] = player_state["owned_cards"]
    game_state["played_cards"] = player_state["played_cards"]
    game_state["actions"] = player_state["actions"]
    game_state["buys"] = player_state["buys"]
    game_state["value"] = player_state["value"]
    game_state["Victory_points"] = player_state["Victory_points"]



    if adversary_state != None:
        game_state["adv_cards_in_hand"] = len(adversary_state["cards_in_hand"])
        game_state["adv_cards_in_deck"] = adversary_state["cards_in_deck"]
        game_state["adv_cards_in_discard"] = len(adversary_state["cards_in_discard"])
        game_state["adv_owned_cards"] = adversary_state["owned_cards"]
        game_state["adv_Victory_points"] = adversary_state["Victory_points"]
    
    return game_state


def put_player_state_adv_state(game_state, player_state):
    game_state["adv_cards_in_hand"] = len(player_state["cards_in_hand"])
    game_state["adv_cards_in_deck"] = player_state["cards_in_deck"]
    game_state["adv_cards_in_discard"] = len(player_state["cards_in_discard"])
    game_state["adv_owned_cards"] = player_state["owned_cards"]
    game_state["adv_Victory_points"] = player_state["Victory_points"]

    return game_state



def get_player_state_from_game_state(game_state):
    player_state = {
        "cards_in_hand": game_state["cards_in_hand"],
        "cards_in_deck": game_state["cards_in_deck"],
        "known_cards_top_deck": game_state["known_cards_top_deck"],
        "cards_in_discard": game_state["cards_in_discard"],
        "owned_cards": game_state["owned_cards"],
        "played_cards": game_state["played_cards"],
        "actions": game_state["actions"],
        "buys": game_state["buys"],
        "value": game_state["value"],
        "Victory_points": game_state["Victory_points"]
    }
    return player_state


def draw_n_cards_from_deck(player_state, n):
    # Shuffle deck if necessary
    
    if int(player_state["cards_in_deck"]) - n < 0:
        cards_in_discard = len(player_state["cards_in_discard"])
        player_state["cards_in_deck"] += cards_in_discard
        player_state["cards_in_discard"] = np.array([])


    deck = get_cards_in_deck(player_state)
    if len(deck) == 0:
        return player_state
    
    cards_drawn = 0
    top_deck_draws = []



    # If cards is put on top of deck, then draw those cards first
    while len(player_state["known_cards_top_deck"]) > 0:
        card = player_state["known_cards_top_deck"][-1]


        top_deck_draws.append(card)

        # Remove card from top deck
        player_state["known_cards_top_deck"] = np.delete(player_state["known_cards_top_deck"], -1)

        # remove from deck also
        deck = np.delete(deck, np.where(deck == card)[0][0])

        cards_drawn += 1
        if cards_drawn == n:
            break

    if len(deck) == 0:
        return player_state
        
    # Can only draw as many cards as there is in the deck
    draws = np.random.choice(deck, min(n - cards_drawn, len(deck)), replace=False) 


    for card in top_deck_draws:
        draws = np.append(draws, card)

    for card in draws:
        player_state["cards_in_hand"] = np.append(player_state["cards_in_hand"], card.astype(int))

    player_state["cards_in_deck"] -= len(draws)


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
    played_card = player_state["played_cards"]


    cards_not_in_deck = np.concatenate((hand, discard_pile, played_card), axis=0)

    all_owned_cards = player_state["owned_cards"]
    for cards in cards_not_in_deck:
        if np.where(all_owned_cards == cards)[0].size > 0:
            all_owned_cards = np.delete(all_owned_cards, np.where(all_owned_cards == cards)[0][0])
    
    deck = all_owned_cards.astype(int)
    
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


def played_cards_2_discard_pile(game_state, player_state):
    ''' [Summary]
    This function will move all the played cards to the discard pile.

    ARGS:
        player_state [dict]: This is the player state object
    '''
    player_state["cards_in_discard"] = np.append(player_state["cards_in_discard"], player_state["played_cards"])
    player_state["played_cards"] = np.array([])


    game_state = merge_game_player_state(game_state, player_state)

    return game_state


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



def supply2discard(game_state, player_state, card):
    ''' [Summary]
    This function will move a card from the supply pile to the discard pile.
    '''
    # Add card to discard pile
    player_state["cards_in_discard"] = np.append(player_state["cards_in_discard"], card)

    # Add card to owned cards
    player_state["owned_cards"] = np.append(player_state["owned_cards"], card)

    # Remove a card from the supply pile
    set_index = card_idx_2_set_idx(card, game_state)
    game_state["supply_amount"][set_index] = str(int(game_state["supply_amount"][set_index]) - 1)

    game_state = merge_game_player_state(game_state, player_state)

    return game_state


def card_idx_2_set_idx(card_idx, game_state):
    # This function will return the index of the card in the dominion_cards game state, based on the card index
    if card_idx == -1:
        return -1
    

    card_idx = int(card_idx)

    for i in range(len(game_state["dominion_cards"])):
        if int(game_state["dominion_cards"][i][1]) == card_idx:
            return i

    return -1 #  Returns -1 if the card is not found in the dominion_cards game state

def supply2deck(game_state, player_state, card):
    ''' [Summary]
    This function will move a card from the supply pile to the top of the deck
    '''
    # Add card to deck pile
    player_state["cards_in_deck"] = int(player_state["cards_in_deck"]) + 1

    # Add card to owned cards
    player_state["owned_cards"] = np.append(player_state["owned_cards"], card)

    # Remove a card from the supply pile
    card_set_idx = card_idx_2_set_idx(card, game_state)
    game_state["supply_amount"][card_set_idx] = str(int(game_state["supply_amount"][card_set_idx]) - 1)


    player_state["known_cards_top_deck"] = np.append(player_state["known_cards_top_deck"], card)

    game_state = merge_game_player_state(game_state, player_state)


    return game_state



def discard_to_deck(game_state, player_state, card):
    ''' [Summary]
    This function will move card from the discard pile to the deck.
    '''
    ''' [Summary]
    This function will move a card from the supply pile to the top of the deck
    '''
    
    # Add card to deck pile and add to top of deck cards
    player_state["cards_in_deck"] = int(player_state["cards_in_deck"]) + 1
    player_state["known_cards_top_deck"] = np.append(player_state["known_cards_top_deck"], card)

    # Remove card from discard pile
    player_state["cards_in_discard"] = np.delete(player_state["cards_in_discard"], np.where(player_state["cards_in_discard"] == card)[0][0])


    game_state = merge_game_player_state(game_state, player_state)

    return player_state


def hand2deck(game_state, player_state, card):
    ''' [Summary]
    This function will move a card from the hand to the top of the deck.
    '''
    # Add card to deck pile and add to top of deck cards
    player_state["cards_in_deck"] = int(player_state["cards_in_deck"]) + 1
    player_state["known_cards_top_deck"] = np.append(player_state["known_cards_top_deck"], card)

    # Remove card from hand
    player_state["cards_in_hand"] = np.delete(player_state["cards_in_hand"], np.where(player_state["cards_in_hand"] == card)[0][0])

    game_state = merge_game_player_state(game_state, player_state)

    return player_state


def discard_hand(game_state, player_state):
    ''' [Summary]
    This function will move all the cards from the hand to the discard pile.
    '''

    player_state["cards_in_discard"] = np.append(player_state["cards_in_discard"], player_state["cards_in_hand"])
    player_state["cards_in_hand"] = np.array([])

    game_state = merge_game_player_state(game_state, player_state)

    return game_state



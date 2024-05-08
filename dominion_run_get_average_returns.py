from Dominion_game import Dominion
from Dominion_game import make_card_set

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

# This training python code is made to train Deep SARSA and cummulate the expected returns over 100 games. averaged over N epochs.



Dominion_game = Dominion()
Dominion_game.card_set = make_card_set([16, 11, 8, 25, 29, 14, 23, 10, 22, 15])

player_random1 = random_player(player_name="Ogus_bogus_man")

Sarsa_player = Deep_SARSA(player_name="Deep_sarsa")
# sarsa_player2 = Deep_SARSA(player_name="Deep_sarsa_2")

# Q_learning_player = Deep_Q_learning(player_name="Deep_Q_learning")
# Expected_SARSA_player = Deep_expected_sarsa(player_name="Deep_expected_sarsa")

# DES_ai = Deep_expected_sarsa(player_name="Deep_expected_sarsa")


# Deep sarsa 2 is trained to get provinces after 20 turns

greedy_test_player = greedy_NN(player_name="Greedy_NN")
# greedy_test_player.load_NN_from_file("NN_models/Deep_sarsa_2_model.keras")
Dominion_game.set_players(Sarsa_player, player_random1) # Training the first player, testing with the second player


trained_player_wins_in_row = 0
test_player_wins_in_row = 0

win_streak_limit = 7

test_game_frequency = 1 # Defines how often the test player should play a game
games_per_epoch = 2 # Defines how many games per epoch
N = 2 # defines how many epochs should be trained on

list_discounted_returns = []


for epoch in range(N):

    discounted_returns = []
    for i in range(games_per_epoch):
        print(f"Game: {i}")


        Dominion_game.set_player2test(greedy_test_player)


        Dominion_game.player1.greedy_mode = False
        
        # Dominion_game.set_player2test(Sarsa_player)
        index_player_won = Dominion_game.play_loop_AI(f"trainer_game_{i}",player_0_is_NN=True, player_1_is_NN=False, verbose=True)


        if index_player_won == 0:
            print("Trained player won!")
        elif index_player_won == 1:
            print("Test player won!")
        else:
            print("Draw!")

        if i % test_game_frequency == 0:
            Dominion_game.set_player2test(player_random1)

            Dominion_game.player1.greedy_mode = True
            index_player_won = Dominion_game.play_loop_AI(f"test_game_{i}",player_0_is_NN=True, player_1_is_NN=False, verbose=True)


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

            
            # Log the discounted return of the game
            discounted_return = Dominion_game.trained_player_discounted_return

            discounted_returns.append(discounted_return)



        print("\n")

    list_discounted_returns.append(discounted_returns)

# Average over the list of discounted returns
    
average_discounted_returns = np.mean(list_discounted_returns, axis=0)


open_file = open("averaged_discounted_rewards.txt", "w")
for reward in average_discounted_returns:
    open_file.write(f"{reward}\n")
open_file.close()
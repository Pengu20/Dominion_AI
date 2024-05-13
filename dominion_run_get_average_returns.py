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


def Evaluate_agent(agent, agent_name, num_games = 200, epochs=10, test_game_frequency=4):
    '''
    This function is made to evaluate three different agents, sarsa, Q-learning and expected SARSA
    '''
    agent_dict = {
        "SARSA": Deep_SARSA,
        "Q-learning": Deep_Q_learning,
        "Expected SARSA": Deep_expected_sarsa
    }


    # This training python code is made to train Deep SARSA and cummulate the expected returns over 100 games. averaged over N epochs
    Dominion_game = Dominion()
    Dominion_game.card_set = make_card_set([16, 11, 8, 25, 29, 14, 23, 10, 22, 15])


    test_game_frequency = test_game_frequency # Defines how often the test player should play a game
    games_per_epoch = num_games # Defines how many games per epoch
    N = epochs # defines how many epochs should be trained on

    list_discounted_returns = []
    average_winrate = []


    for epoch in range(N):

        player_random1 = random_player(player_name="Ogus_bogus_man")
        agent_class = agent_dict[agent](player_name=agent_name)


        Dominion_game.set_players(agent_class, player_random1) # Training the first player, testing with the second player


        discounted_returns = []
        wins = 0
        test_games = 0
        for i in range(games_per_epoch):
            print(f"Epoch: {epoch}, Game: {i} ---- Agent: {agent_name}")


            Dominion_game.player1.greedy_mode = False
            index_player_won = Dominion_game.play_loop_AI(f"trainer_game_{i}",player_0_is_NN=True, player_1_is_NN=False, verbose=False)


            if index_player_won == 0:
                print("Trained player won!")
            elif index_player_won == 1:
                print("Test player won!")
            else:
                print("Draw!")



            if i % test_game_frequency == 0:
                print("---- Testing agent policy ----")
                test_games += 1
                Dominion_game.player1.greedy_mode = True
                index_player_won = Dominion_game.play_loop_AI(f"test_game_{i}",player_0_is_NN=True, player_1_is_NN=False, verbose=True)


                if index_player_won == 0:
                    print("Trained player won!")
                    wins += 1
                elif index_player_won == 1:
                    print("Test player won!")
                else:
                    print("Draw!")

                
                # Log the discounted return of the game
                discounted_return = Dominion_game.trained_player_discounted_return

                discounted_returns.append(discounted_return)



            print("\n")


        average_winrate.append(wins/test_games)
        list_discounted_returns.append(discounted_returns)

    # Average over the list of discounted returns
        
    average_discounted_returns = np.mean(list_discounted_returns, axis=0)
    winrate = np.mean(average_winrate)


    open_file = open(f"averaged_discounted_rewards_{agent_name}.txt", "w")
    for reward in average_discounted_returns:
        open_file.write(f"{reward}\n")
    open_file.close()


    open_file = open(f"averaged_wins.txt_{agent_name}.txt", "w")
    open_file.write(f"{winrate}\n")
    open_file.close()

'''
P1 = multiprocessing.Process(target=Evaluate_agent, args=("SARSA", "Deep_sarsa"))
P2 = multiprocessing.Process(target=Evaluate_agent, args=("Q-learning", "Deep_Q_learning"))
P3 = multiprocessing.Process(target=Evaluate_agent, args=("Expected SARSA", "Deep_expected_sarsa"))
'''

'''
"SARSA": Deep_SARSA,
"Q-learning": Deep_Q_learning,
"Expected SARSA": Deep_expected_sarsa
'''


agent = "Expected SARSA"
agent_name = "Deep_expected_sarsa"
Evaluate_agent(agent, agent_name)


import numpy as np

import pickle
from keras import Sequential
from keras.layers import Dense
import time

class Dominion_reward():
    ''' [summary]
        This class is used to determine the reward based on the state given in the dominion game
    '''

    def __init__(self) -> None:
        pass
        
    def get_reward_from_state(self, game_state):
        ''' [summary]
            This function is used to get the reward from the game state
            Args:
                game_state (struct): [description] This game state is a hand tailored struct that defines all the 
                                                   informations about the game
            Returns:
                reward (int): [description] The reward that the player gets from the game state
        '''

        reward = -1
        Victory_reward = 0 #100 if won -100 if lost
        Victory_points_reward = 2 # 1 per victory point

        Province_difference_reward = 0 # 3 per province difference

        Cards_played_reward = 0 # 3 if you played more than 3 cards
        few_coppers_reward = 0 # 3 if you have less than 3 coppers

        no_copper_reward = 0 # 3 if you have no coppers
        no_estates_reward = 0 # 3 if you have no estates

        gold_reward = 0 # get 1 point for each 3 value of player

        curses_owned = 0 # -1 point per curse

        Dead_action_cards_reward = 0 # -1 point per action card in hand, if you have no actions left

    


        # ---------------- Reward based on game end ----------------
        if   (game_state["main_Player_won"] == 1):
            Victory_reward = 20
        elif (game_state["main_Player_won"] == 1):
            Victory_reward = -20



        # ---------------- Reward based on province difference of players ----------------
        province_main = 0
        province_adv = 0

        for card in game_state["owned_cards"]:
            if card == 5:
                province_main += 1

        for card in game_state["adv_owned_cards"]:
            if card == 5:
                province_adv += 1

        Province_difference_reward = (province_main - province_adv) * 3



        # ---------------- Points for having more victory points than the other players ----------------
        Victory_points_diff = game_state["Victory_points"] - game_state["adv_Victory_points"]
        Victory_points_reward = 2 * np.sign(Victory_points_diff)
        


        # ---------------- reward for playing many cards ----------------
        if len(game_state["played_cards"]) > 3:
            Cards_played_reward = 3



        # ---------------- reward for having no coppers ----------------
        coppers = 0
        for card in game_state["owned_cards"]:
            if card == 0:
                coppers += 1

        if coppers < 3:
            few_coppers_reward = 3
        
        if coppers == 0:
            no_copper_reward = 3
        

        # ---------------- reward for having no estates ----------------
        estates = 0
        for card in game_state["owned_cards"]:
            if card == 3:
                estates += 1

        if estates == 0:
            no_estates_reward = 3


        # ---------------- gold reward ----------------

        gold_reward = int(game_state["value"]/3)



        # ---------------- curse punishment ----------------
        curses = 0
        for card in game_state["owned_cards"]:
            if card == 6:
                curses += 1

        curses_owned = -1 * curses


        # ---------------- dead action cards punishment ----------------
        dead_action_cards = 0
        for card in game_state["cards_in_hand"]:
            if game_state["actions"] == 0:
                if card >= 6 and card != 13:
                    dead_action_cards += 1
        
        Dead_action_cards_reward = -1 * dead_action_cards



        reward_list = np.array([reward, Victory_reward, Victory_points_reward, Province_difference_reward, 
                                Cards_played_reward, few_coppers_reward, no_copper_reward, no_estates_reward, 
                                gold_reward, curses_owned, Dead_action_cards_reward])


        return reward_list


    def struct_generator(self):
        '''
        Not used, is onyl here to trick github copilot
        '''
        self.reward = state = {

            # ----- SUPPLY RELATED -----
        "dominion_cards": self.deck.get_card_set(),
        "supply_amount": np.append(self.standard_supply, np.ones(10) * 10), # 10 kingdom cards with 10 supply each


            # ----- WHAT THE ACTION MEANS -----
        "Unique_actions": None, # This is the unique actions that the player can do. This is based on the cards in the players hand
        "Unique_actions_parameter": 0, # This is the parameter that the unique action needs to be executed (Often not needed default is zero)


            # ----- MAIN PLAYER -----
        "main_Player_won": -1,
        "cards_in_hand": np.array([]),
        "cards_in_deck": 0,
        "known_cards_top_deck": np.array([]),
        "cards_in_discard": np.array([]),
        "owned_cards": np.array([]),
        "played_cards": np.array([]), # "played" cards are cards that are in the current hand
        "actions": 0,
        "buys": 0,
        "value": 0,
        "Victory_points": 0,



            # ----- ADVERSARY PLAYER -----
        "adv_Player_won": -1,
        "adv_cards_in_hand": 0,
        "adv_cards_in_deck": 0,
        "adv_cards_in_discard": 0,
        "adv_owned_cards": np.array([]),
        "Victory_points": 0,
        }




        pass



class Deep_SARSA:
    def __init__(self, player_name) -> None:
        self.rf = Dominion_reward()
        self.initialize_NN()
        self.game_state_history = []
        self.action_history = []


        self.player_name = player_name
        self.file_address = f"reward_history/{self.player_name}_reward_history.txt"
        self.file_sum_expected_rewards = f"reward_history/{self.player_name}_sum_expected_rewards.txt"

        self.played_games = 0


        self.SARSA_update_time = []
        self.convert_state2list_time = []
        self.NN_predict_time = []
        self.NN_training_time = []

        ## Experimental design, where neural network is first updates at the end of the game using SARSA

        self.expected_return_history = []
        self.games_played = 0


        self.sum_expected_return = 0 # This is used to keep track of the sum of expected returns gained by the players


    def initialize_NN(self):
        self.model = Sequential()
        self.model.add(Dense(1024, activation='sigmoid', input_shape=(9000,)))
        self.model.add(Dense(512, activation='sigmoid'))
        self.model.add(Dense(256, activation='sigmoid'))
        self.model.add(Dense(1,activation='linear'))

        self.model.compile( optimizer='Adam',
                            loss='mean_squared_error',
                            metrics='accuracy',
                            loss_weights=None,
                            weighted_metrics=None,
                            run_eagerly=None,
                            steps_per_execution=None,
                            jit_compile=None,
                            )
        
        self.model.summary()


    def update_NN(self, game_state, action, expected_return_updated):
        '''
        This function is used to update the neural network with the new values
        '''


        list_NN_input = self.game_state2list_NN_input(game_state, [action])
        self.model.fit(list_NN_input, np.array([[expected_return_updated]]), epochs=1, verbose=0)

    
    def __game_state_list2NN_input(self, game_state_list, action_list):
        '''
        This function maps the input of the game state to the input of the neural network
        It is assumed that the size of the gamestate value is 9000
        '''

        input_matrix = np.zeros((len(game_state_list),9000))

        for i in range(len(game_state_list)):
            list_NN_input = self.game_state2list_NN_input(game_state_list[i], [action_list[i]])
            input_matrix[i,:] = list_NN_input

        return input_matrix
    
    def __expected_return_list2NN_output(self, expected_return_updated_list):
        '''
        This function is used to convert the expected return list to the output of the neural network
        '''
        output_label = np.zeros([len(expected_return_updated_list),1])

        for i in range(len(expected_return_updated_list)):
            output_label[i,:] = expected_return_updated_list[i]

        return output_label



    def __update_NN_np_mat(self, input_matrix, output_matrix):
        '''
        This function is used to update the neural network using a list of all the values used in the game
        '''
        time_start = time.time()

        self.model.fit(input_matrix, output_matrix, epochs=10, verbose=0)

        self.NN_training_time.append(time.time() - time_start)


    def game_state2list_NN_input(self, game_state, action_list):
        '''
        This function is used to convert the game state to a 
        list that can be used as input for the neural network
        '''

        start_time = time.time()

        binarizeed_gamestate = pickle.dumps(game_state)

        # Convert bytearray to list of integers
        list_NN_input = np.array([byte for byte in binarizeed_gamestate])

        NN_inputs = np.zeros((9000, len(action_list)))
        i = 0

        for action in action_list:
            list_NN_input = np.insert(list_NN_input, 0, action)

            list_NN_input.resize((len(list_NN_input),1))


            # Padding the value to 9000
            input_padded = np.zeros((9000,1))
            input_padded[:len(list_NN_input)] = list_NN_input

            NN_inputs[:,i] = input_padded[:,0]

            i += 1

        self.convert_state2list_time.append(time.time() - start_time)

        return NN_inputs.T # Apparently keras needs the matrix transposed


    def NN_get_expected_return(self, game_state, actions_list):
        '''
        This function gives the value from the neural network to the state action pair
        '''

        list_NN_inputs = self.game_state2list_NN_input(game_state, actions_list)


        expected_return = self.model.predict(list_NN_inputs, verbose=0)



        return expected_return


    def SARSA_update(self, game_state, action):
        '''
        This function is used to update the previous timestep with the new reward
        '''

        start_time = time.time()
        alpha = 0.1 # Learning rate
        gamma = 0.9 # Discount factor


        # SA -> State action
        expected_return = self.NN_get_expected_return(game_state, [action])[0]
        old_expected_return = self.NN_get_expected_return(self.game_state_history[-1], [self.action_history[-1]])[0]

        reward = np.sum(self.rf.get_reward_from_state(game_state))


        # SARSA update
        old_expected_return_updated = old_expected_return + alpha * (reward + gamma*expected_return - old_expected_return)

        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values

        # Store the updated values
        self.expected_return_history.append(old_expected_return_updated)


        self.sum_expected_return += old_expected_return

        self.SARSA_update_time.append(time.time() - start_time)


    def greedy_choice(self, list_of_actions, game_state):
        '''
        Until a neural network can give us the best state action rewards, we will use this function to give us the rewards
        '''
        time_start = time.time()
        expected_return = self.NN_get_expected_return(game_state, list_of_actions)

        self.NN_predict_time.append(time.time() - time_start)

        return list_of_actions[np.argmax(expected_return)]


    def epsilon_greedy_policy(self, list_of_actions, game_state, epsilon):
        '''
        This function is used to get the action from the epsilon greedy policy
        '''
        if np.random.rand() < epsilon:
            return np.random.choice(list_of_actions)
        else:
            return self.greedy_choice(list_of_actions, game_state)


    def choose_action(self, list_of_actions, game_state):

        if self.game_state_history == []:
            self.game_state_history.append(game_state)
            self.action_history.append(np.random.choice(list_of_actions))
            return self.action_history[-1]
        else:
            
            action = self.epsilon_greedy_policy(list_of_actions, game_state, 0.1)
            self.SARSA_update(game_state, action)



            self.game_state_history.append(game_state)
            self.action_history.append(action)

            #Remove the previous old values of game state and action history
            return action



    def write_state_reward_to_file(self, game_state):
        '''
        This function is used to write the current player reward into the reward file
        '''
        reward = self.rf.get_reward_from_state(game_state)

        open_file = open(self.file_address, "a")
        open_file.write(f"{np.sum(reward)}  - {reward}\n")
        open_file.close()


        open_file = open(self.file_sum_expected_rewards, "a")
        open_file.write(f"{np.sum(self.sum_expected_return)}\n")
        self.sum_expected_return = 0
        open_file.close()



        self.played_games += 1

        print("Average times: ")
        sarsa_time = np.array(self.SARSA_update_time)
        convert2list_time = np.array(self.convert_state2list_time)
        NN_predict_time = np.array(self.NN_predict_time)
        NN_training_time = np.array(self.NN_training_time)

        print(f"SARSA update: {np.mean(sarsa_time)} - RUN {len(sarsa_time)} times")
        print(f"Convert to list: {np.mean(convert2list_time)} - RUN {len(convert2list_time)}")
        print(f"NN predict: {np.mean(NN_predict_time)} - RUN {len(NN_predict_time)}")
        print(f"NN training: {np.mean(NN_training_time)} - RUN {len(NN_training_time)}")




        self.SARSA_update_time = []
        self.convert_state2list_time = []
        self.NN_predict_time = []
        self.NN_training_time = []

        if self.played_games % 50 == 0:
            # Save model every 50 games
            self.model.save(f"NN_models/{self.player_name}_model.keras")



    def notify_game_end(self):
        ''' [summary]
            This function is used to notify the player that the game has ended
        '''

        # Deep sarsa will update its neural network with the new values
        
        len_gm = len(self.game_state_history)
        len_ac = len(self.action_history)
        game_state_hist = self.game_state_history[:len_gm-1]
        action_hist = self.action_history[:len_ac-1]

        game_state_data = pickle.loads(pickle.dumps(self.game_state_history[-1]))

        game_ID = "game_" + str(self.games_played)

        input_matrix = self.__game_state_list2NN_input(game_state_hist, action_hist)
        output_matrix = self.__expected_return_list2NN_output(self.expected_return_history)

        file = open(f"Q_table_data/input_data/input_data_{game_ID}.txt", "wb")
        pickle.dump(input_matrix, file)
        file.close()

        file = open(f"Q_table_data/output_data/output_data_{game_ID}.txt", "wb")
        pickle.dump(output_matrix, file)
        file.close()


        self.__update_NN_np_mat(input_matrix, output_matrix)

        self.game_state_history = []
        self.action_history = []
        self.expected_return_history = []


        self.games_played += 1



class random_player:
    def __init__(self, player_name):
        self.rf = Dominion_reward()
        self.file_address = f"reward_history/{player_name}_reward_history.txt"
        self.player_name = player_name

    def get_name(self):
        return self.player_name

    def get_reward(self, game_state):
        '''
        This function is used to get the reward from the game state
        '''
        reward = self.rf.get_reward_from_state(game_state)

        return reward
    

    def choose_action(self, list_of_actions, game_state):
        return np.random.choice(list_of_actions)


    def write_state_reward_to_file(self, game_state):
        '''
        This function is used to write the current player reward into the reward file
        '''
        reward = self.get_reward(game_state)


        open_file = open(self.file_address, "a")
        open_file.write(f"{np.sum(reward)}  - {reward}\n")
        open_file.close()


    def notify_game_end(self):
        ''' [summary]
            This function is used to notify the player that the game has ended
        '''

        pass


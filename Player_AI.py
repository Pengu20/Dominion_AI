import numpy as np

import pickle
import keras
from keras.layers import Dense
from keras import layers
from keras.regularizers import L1
from keras.regularizers import L2
from keras import Model
from keras import Input
import time

import state_manipulator as sm

import cards_base_ed2

import copy

class Dominion_reward():
    ''' [summary]
        This class is used to determine the reward based on the state given in the dominion game
    '''

    def __init__(self) -> None:
        pass
        
    def get_reward_from_state(self, game_state, previous_game_state):
        ''' [summary]
            This function is used to get the reward from the game state
            Args:
                game_state (struct): [description] This game state is a hand tailored struct that defines all the 
                                                   informations about the game
            Returns:
                reward (int): [description] The reward that the player gets from the game state
        '''

        # The rewards based on cards in deck, should only be given, 
        # when the card enters the deck, not at all times with the given cards

        reward = -20
        Victory_reward = 0 #20 if won -100 if lost, extra 180, if won by provinces

        Victory_points_difference_reward = 0 # 10 per victory point difference
        Victory_points_reward = 0 # 5 per victory point

        treasure_in_hand_reward = 0 # with 5 or more treasure in hand, gain 5 points for each treasure above 5.

        Province_owned_reward = 0 # 3 per province 
        Province_difference_reward = 0 # 5 per province difference

        Cards_played_reward = 0 # 5 if you played more than 3 cards


        no_copper_reward = 0 # 5 if you have no coppers
        no_estates_reward = 0 # 5 if you have no estates

        gold_reward = 0 # get 3 point for each gold in deck

        deck_value_reward = 0 # -10 if you have less than 3 in value

        no_cards_punishment = 0 # -10 if you own less than 5 cards

        curses_owned = 0 # -10 point per curse

        Gained_expensive_cards_reward = 0 # Gain a reward based on the cost of the bought card to the power of 2

        Too_many_cards_punishment = 0 # after 30, give punishment for cards in deck
        


        # ---------------- Reward based on game end ----------------
        if   (game_state["main_Player_won"] == 1):
            Victory_reward = 500

            # If the province pile is empty, the player won by provinces and gets an extra reward
            if game_state["supply_amount"][5] == 0:
                Victory_reward += 500


        elif (game_state["adv_Player_won"] == 1):
            Victory_reward = -150


        # ---------------- Gained card reward ----------------
        
        # Check all new cards gained.
        owned_cards = copy.deepcopy(game_state["owned_cards"])
        pre_owned_cards = copy.deepcopy(previous_game_state["owned_cards"])


        if len(owned_cards) > len(pre_owned_cards):

            for cards in pre_owned_cards:
                if np.where(owned_cards == cards)[0].size > 0:
                    owned_cards = np.delete(owned_cards, np.where(owned_cards == cards)[0][0])
            
            new_cards = owned_cards.astype(int)

            for card in new_cards:
                card_set = copy.deepcopy(game_state["dominion_cards"])

                card_set_idx = sm.card_idx_2_set_idx(card, game_state=copy.deepcopy(game_state))
                card_cost = int(card_set[card_set_idx][2])
                Gained_expensive_cards_reward += card_cost**2

            # Double reward, if the player spend all their money
            if game_state["value"] == 0 and game_state["buys"] == 0 and game_state["Unique_actions"] == "buy":
                Gained_expensive_cards_reward *= 2
            elif game_state["value"] > 0 and game_state["buys"] == 0 and game_state["Unique_actions"] == "buy":
                Gained_expensive_cards_reward /= 2
        else:
            new_cards = []
            
        

        # ---------------- Reward based on province difference of players ----------------
        province_main = 0
        province_adv = 0

        for card in game_state["owned_cards"]:
            if card == 5:
                province_main += 1

        for card in game_state["adv_owned_cards"]:
            if card == 5:
                province_adv += 5
        


        new_province = 0
        for card in new_cards:
            if card == 5:
                new_province += 1


        Province_difference_reward = abs((province_main - province_adv)**1.5) * np.sign(province_main - province_adv)

        Province_owned_reward = (new_province*25)**1.5



        # ---------------- Points for having more victory points than the other players ----------------
        Victory_points_diff = copy.deepcopy(game_state["Victory_points"]) - copy.deepcopy( game_state["adv_Victory_points"])
        Victory_points_difference_reward = 10 * np.sign(Victory_points_diff)


        
        # ---------------- Points for having a high density of victory points ----------------

        
        if len(game_state["owned_cards"]) > 0:
            Victory_points_reward = 10 * copy.deepcopy(game_state["Victory_points"])


        # ---------------- reward for playing many cards ----------------
        Cards_played_reward = (len(game_state["played_cards"])*150)**0.8


        # ---------------- reward for having few/no coppers ----------------
        coppers = 0
        for card in game_state["owned_cards"]:
            if card == 0:
                coppers += 1

        
        if coppers == 0:
            no_copper_reward = 10
        

        # ---------------- reward for having no estates ----------------
        estates = 0
        for card in game_state["owned_cards"]:
            if card == 3:
                estates += 1

        if estates == 0:
            no_estates_reward = 5
        else:
            no_estates_reward = -estates*40

        no_estates_reward = 0

        # ---------------- gold reward ----------------

        gold_cards = 0
        for card in new_cards:
            if card == 2:
                gold_cards += 1
        
        gold_reward = (50*gold_cards**0.5)


        # ---------------- reward for having alot of value (weighted by deck size) ----------------

        coppers = 0
        silvers = 0
        golds = 0

        for card in game_state["owned_cards"]:
            if card == 0:
                coppers += 1
            elif card == 1:
                silvers += 1
            elif card == 2:
                golds += 1

        # deck_value_reward = int((coppers + 2 * silvers + 3 * golds)/len(game_state["owned_cards"])*50)
                


        deck_value_reward = 0

        # ---------------- Punishment for having critically low treasure value ------------+++----

        if (coppers + 2 * silvers + 3 * golds) <= 2:
            deck_value_reward = -30 * (3 - (coppers + 2 * silvers + 3 * golds))


        current_coppers = 0
        current_silvers = 0
        current_gold = 0


        previous_coppers = 0
        previous_silvers = 0
        previous_gold = 0


        # ---------------- reward for having alot of treasure in hand ----------------
        # Based on the difference between the current and the previous gamestate. Can only gain reward in the action state
        if game_state["Unique_actions"] == "take_action" and previous_game_state["Unique_actions"] == "take_action":
            if len(game_state["cards_in_hand"]) > 0:
                for card in game_state["cards_in_hand"]:
                    if card == 0:
                        current_coppers += 1
                    elif card == 1:
                        current_silvers += 1
                    elif card == 2:
                        current_gold += 1


            if len(previous_game_state["cards_in_hand"]) > 0:
                for card in previous_game_state["cards_in_hand"]:
                    if card == 0:
                        previous_coppers += 1
                    elif card == 1:
                        previous_silvers += 1
                    elif card == 2:
                        previous_gold += 1

                treasure_in_hand_current = current_coppers + 2*current_silvers + 3*current_gold
                treasure_in_hand_previous = previous_coppers + 2*previous_silvers + 3*previous_gold
                gained_treasure = treasure_in_hand_current - treasure_in_hand_previous
                treasure_in_hand_reward = int(max(0, gained_treasure))**1.3


        # ---------------- no_cards_punishment ----------------

        if len(game_state["owned_cards"]) < 5:
            no_cards_punishment = -50


        # ---------------- curse punishment ----------------
        curses = 0
        for card in new_cards:
            if card == 6:
                curses += 1

        curses_owned = -300 * curses

        # ---------------- Too many cards punishment ----------------
        deck_length = len(game_state["owned_cards"])
        new_cards = len(new_cards)
        deck_limit = 35

        if new_cards > 0 and deck_length > deck_limit:
            Too_many_cards_punishment = -(deck_length - deck_limit)**1.7


        reward_list = np.array([reward, Victory_reward, Victory_points_reward,
                                Victory_points_difference_reward, Province_owned_reward, 
                                Province_difference_reward, Cards_played_reward, 
                                no_copper_reward, no_estates_reward, 
                                gold_reward, deck_value_reward,
                                Too_many_cards_punishment, no_cards_punishment,
                                curses_owned, Gained_expensive_cards_reward,
                                treasure_in_hand_reward
                                ])

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
        self.file_address = f"reward_history/{self.player_name}/{self.player_name}_reward_history.txt"

        self.file_average_expected_rewards = f"reward_history/{self.player_name}/{self.player_name}_average_expected_rewards.txt"
        self.file_variance_expected_rewards = f"reward_history/{self.player_name}/{self.player_name}_variance_expected_rewards.txt"

        self.file_average_returns = f"reward_history/{self.player_name}/{self.player_name}_average_returns.txt"
        self.file_variance_returns = f"reward_history/{self.player_name}/{self.player_name}_variance_returns.txt"

        self.file_variance_NN_error = f"reward_history/{self.player_name}/{self.player_name}_variance_NN_error.txt"
        self.file_variance_NN_error = f"reward_history/{self.player_name}/{self.player_name}_variance_NN_error.txt"

        self.file_victory_points = f"reward_history/{self.player_name}/{self.player_name}_victory_points.txt"
        self.file_games_won = f"reward_history/{self.player_name}/{self.player_name}_games_won.txt"
        self.file_game_length = f"reward_history/{self.player_name}/{self.player_name}_game_length.txt"
        self.file_Average_NN_error = f"reward_history/{self.player_name}/{self.player_name}_average_NN_error.txt"
        self.file_variance_NN_error = f"reward_history/{self.player_name}/{self.player_name}_variance_NN_error.txt"


        self.delete_all_previous_history()


        self.SARSA_update_time = []
        self.convert_state2list_time = []
        self.NN_predict_time = []
        self.NN_training_time = []
        self.NN_error = []

        ## Experimental design, where neural network is first updates at the end of the game using SARSA


        self.games_played = 0
        self.turns_in_game = 0


        self.all_expected_returns = [] # This is used to keep track of the sum of expected returns gained by the players
        self.all_returns = []

        # DEBUG, so i can see the latest reward

        self.latest_reward = None
        self.latest_action = None
        self.latest_action_type = None
        self.latest_updated_expected_return = None
        self.latest_desired_expected_return = None

        self.greedy_mode = False
        

        self.only_terminate_action = True


    def load_NN_from_file(self, path):
        self.model = keras.models.load_model(path)

    def delete_all_previous_history(self):
        '''
        This funtions opens all the file paths for overwrite, to delete all previous data
        '''


        open_file = open(self.file_address, "w")
        open_file.close()

        open_file = open(self.file_average_expected_rewards, "w")
        open_file.close()

        open_file = open(self.file_variance_expected_rewards, "w")
        open_file.close()

        open_file = open(self.file_victory_points, "w")
        open_file.close()

        open_file = open(self.file_games_won, "w")
        open_file.close()

        open_file = open(self.file_game_length, "w")
        open_file.close()

        open_file = open(self.file_Average_NN_error, "w")
        open_file.close()

        open_file = open(self.file_variance_NN_error, "w")
        open_file.close()

        open_file = open(self.file_average_returns, "w")
        open_file.close()

        open_file = open(self.file_variance_returns, "w")
        open_file.close()

        

    def initialize_NN(self):

        input_1 = keras.Input(shape=(9000,))
        input_2 = keras.Input(shape=(8,))

        Input_layer = Dense(2048, activation='sigmoid',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(input_1)
        Hidden_layer1 = Dense(1024, activation='sigmoid',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(Input_layer)
        Hidden_layer2 = layers.Dropout(0.4)(Hidden_layer1)
        Hidden_layer3 = Dense(512, activation='sigmoid',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(Hidden_layer2)
        Hidden_layer4 = layers.Dropout(0.4)(Hidden_layer3)
        Hidden_layer5 = Dense(256, activation='sigmoid',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(Hidden_layer4)

        #action handling layers
        action_layer1 = Dense(4, activation='sigmoid',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(input_2)
        #action_layer_dropout = layers.Dropout(0.2)(action_layer1) # Super spicey dropout, might be kinda shit
        Concatenated_layer = layers.concatenate([Hidden_layer5, action_layer1], axis=1)

        Hidden_layer6 = Dense(128, activation='sigmoid',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(Concatenated_layer)

        output = Dense(1,activation='linear',kernel_regularizer=L1(0.01),activity_regularizer=L2(0.01))(Hidden_layer6)

        self.model = Model(inputs=[input_1, input_2], outputs=output)


        self.model.compile( optimizer='Adam',
                            loss='huber',
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


        NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state, [action])
        self.model.fit((NN_input_state, NN_input_action), np.array([[expected_return_updated]]), epochs=10, verbose=0)

    
    def game_state_list2NN_input(self, game_state_list, action_list):
        '''
        This function maps the input of the game state to the input of the neural network
        It is assumed that the size of the gamestate value is 9000
        '''

        input_state_matrix = np.zeros((len(game_state_list),9000))
        input_action_matrix = np.zeros((len(action_list),8)) # Number of bits used to represent the action value

        for i in range(len(game_state_list)):
            NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state_list[i], [action_list[i]])
            input_state_matrix[i,:] = NN_input_state

            input_action_matrix[i,:] = NN_input_action


        return input_state_matrix, input_action_matrix
    

    def expected_return_list2NN_output(self, expected_return_updated_list):
        '''
        This function is used to convert the expected return list to the output of the neural network
        '''
        output_label = np.zeros([len(expected_return_updated_list),1])

        for i in range(len(expected_return_updated_list)):
            output_label[i,:] = expected_return_updated_list[i]

        return output_label



    def update_NN_np_mat(self, input_matrix, output_matrix):
        '''
        This function is used to update the neural network using a list of all the values used in the game
        '''
        time_start = time.time()

        self.model.fit(input_matrix, output_matrix, epochs=1, verbose=0)

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

        NN_inputs_state = np.zeros((9000, len(action_list)))
        NN_inputs_actions = np.zeros((8, len(action_list))) # 8 is the bit number representation of the action
        i = 0


        for action in action_list:

            # Padding the value to 9000
            input_padded = np.zeros((9000,1))
            input_padded[:len(list_NN_input)] = np.array(list_NN_input).reshape((len(list_NN_input),1))

            NN_inputs_state[:,i] = input_padded[:,0]

            # Creating the action input
            # Binarise into two complements 8 bit number


            binarised_action = np.binary_repr(action.astype(int), width=8)
            
            for bin in range(8):
                NN_inputs_actions[bin,i] = int(binarised_action[bin])

            i += 1

        self.convert_state2list_time.append(time.time() - start_time)




        return NN_inputs_state.T, NN_inputs_actions.T  # Apparently keras needs the matrix transposed


    def NN_get_expected_return(self, game_state, actions_list):
        '''
        This function gives the value from the neural network to the state action pair
        '''

        NN_input_state, NN_input_action = self.game_state2list_NN_input(game_state, actions_list)


        expected_return = self.model.predict([NN_input_state, NN_input_action], verbose=0)


        return expected_return


    def SARSA_update(self, game_state, action, game_ended=False):
        '''
        This function is used to update the previous timestep with the new reward
        '''

        start_time = time.time()
        self.alpha = 0.1 # Learning rate
        gamma = 0.9 # Discount factor


        # SA -> State action

        if game_ended:
            expected_return = 0
        else:
            expected_return = self.NN_get_expected_return(game_state, [action])[0]

        old_expected_return = self.NN_get_expected_return(self.game_state_history[-1], [self.action_history[-1]])[0]

        reward_list = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])
        reward = np.sum(reward_list)
        self.latest_reward = reward_list


        # SARSA update
        old_expected_return_updated = old_expected_return + self.alpha * (reward + gamma*expected_return - old_expected_return)


        NN_error = (reward + gamma*expected_return - old_expected_return)**2
        self.NN_error.append(NN_error)

        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values



        self.all_expected_returns.append(old_expected_return_updated)
        self.all_returns.append(reward)


        self.turns_in_game += 1
        

        if self.greedy_mode == False:
            # self.update_NN(self.game_state_history[-1], self.action_history[-1], old_expected_return_updated)

            self.batch_size = 16

            # Every batch_size turns we will update the neural network with the batch_size new datasets
            if self.turns_in_game % self.batch_size == 0:
                input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
                output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])


                self.update_NN_np_mat(input_matrix, output_matrix)



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

            # If random choice is choosen, then reduce the probability of choosing action -> -1.
            choice = np.random.choice(list_of_actions)

            # Reroll random choice, if the choice was -1
            if len(list_of_actions) != 1:
                rerolls = 0
                for i in range(rerolls):
                    if choice == -1:
                        choice = np.random.choice(list_of_actions)
                    else:
                        break
                

            return choice
        else:
            return self.greedy_choice(list_of_actions, game_state)


    def choose_action(self, list_of_actions, game_state):

        if self.game_state_history == []:
            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(np.random.choice(list_of_actions))
            return self.action_history[-1]
        else:
            
            if self.greedy_mode:
                action = self.greedy_choice(list_of_actions, game_state)
            else:
                action = self.epsilon_greedy_policy(list_of_actions, copy.deepcopy(game_state), 0.1)
            
            self.SARSA_update(copy.deepcopy(game_state), action)


            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(copy.deepcopy(action))

            #Remove the previous old values of game state and action history
            return action



    def write_state_reward_to_file(self, game_state):
        '''
        This function is used to write the current player reward into the reward file
        '''
        reward = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])

        open_file = open(self.file_address, "a")
        open_file.write(f"{np.sum(reward)}  - {reward}\n")
        open_file.close()


        open_file = open(self.file_average_expected_rewards, "a")
        open_file.write(f"{np.mean(self.all_expected_returns)}\n")
        open_file.close()


        open_file = open(self.file_variance_expected_rewards, "a")
        open_file.write(f"{np.var(self.all_expected_returns)}\n")
        open_file.close()

        open_file = open(self.file_victory_points, "a")
        victory_points = np.sum(game_state["Victory_points"])
        open_file.write(f"{victory_points}\n")
        open_file.close()


        # Log the accuracy of the neural network
        open_file = open(self.file_Average_NN_error, "a")
        open_file.write(f"{np.mean(self.NN_error)}\n")
        open_file.close()

        open_file = open(self.file_variance_NN_error, "a")
        open_file.write(f"{np.var(self.NN_error)}\n")
        open_file.close()

        self.NN_error = []


        open_file = open(self.file_average_returns, "a")
        open_file.write(f"{np.mean(self.all_returns)}\n")
        open_file.close()

        open_file = open(self.file_variance_returns, "a")
        open_file.write(f"{np.var(self.all_returns)}\n")
        open_file.close()

        self.all_returns = []


        open_file = open(self.file_games_won, "a")
        if game_state["main_Player_won"] == 1:
            open_file.write("1\n")
        else:
            open_file.write("0\n")
        open_file.close()


        open_file = open(self.file_game_length, "a")
        open_file.write(f"{np.sum(self.turns_in_game)}\n")
        open_file.close()


        print("Average times: ")
        sarsa_time = np.array(self.SARSA_update_time)
        convert2list_time = np.array(self.convert_state2list_time)
        NN_predict_time = np.array(self.NN_predict_time)
        # NN_training_time = np.array(self.NN_training_time)



        print(f"SARSA update: {np.mean(sarsa_time)} - RUN {len(sarsa_time)} times")
        print(f"Convert to list: {np.mean(convert2list_time)} - RUN {len(convert2list_time)}")
        print(f"NN predict: {np.mean(NN_predict_time)} - RUN {len(NN_predict_time)}")
        # print(f"NN training: {np.mean(NN_training_time)} - RUN {len(NN_training_time)}")





        self.SARSA_update_time = []
        self.convert_state2list_time = []
        self.NN_predict_time = []
        # self.NN_training_time = []

        if self.games_played % 50 == 0:
            # Save model every 50 games
            self.model.save(f"NN_models/{self.player_name}_model.keras")



    def game_end_update(self, game_state):
        '''
        This function is used to update the neural network with the new values
        '''

        self.SARSA_update(game_state, None, game_ended=True)



    def notify_game_end(self, game_state):
        ''' [summary]
            This function is used to notify the player that the game has ended
        '''


        if self.greedy_mode:
            self.write_state_reward_to_file(game_state)
        else:
            # Deep sarsa will update its neural network with the new values
            self.game_end_update(game_state)
        

        # Saving the game data

        # self.update_NN_np_mat(input_matrix, output_matrix)
            
        self.latest_reward = None
        self.latest_action = None
        self.latest_action_type = None
        self.latest_updated_expected_return = None
        self.latest_desired_expected_return = None


        self.game_state_history = []
        self.action_history = []
        self.all_expected_returns = []
        self.turns_in_game = 0

        self.games_played += 1



class greedy_NN(Deep_SARSA):
    '''
    This class is for loading the neural network gained from deep sarsa to make all the greedy actions.
    '''

    def greedy_choice(self, list_of_actions, game_state):
        '''
        Until a neural network can give us the best state action rewards, we will use this function to give us the rewards
        '''

        expected_return = self.NN_get_expected_return(game_state, list_of_actions)

        self.all_expected_returns.append(np.max(expected_return))


        return list_of_actions[np.argmax(expected_return)]


    def choose_action(self, list_of_actions, game_state):


        action = self.greedy_choice(list_of_actions=list_of_actions, game_state=game_state)
        self.turns_in_game += 1

        #Remove the previous old values of game state and action history
        return action
   
   
    def notify_game_end(self, game_state):
        ''' [summary]
            This function is used to notify the player that the game has ended
        '''



class Deep_Q_learning(Deep_SARSA):

    def Q_learning_update(self, game_state, list_of_actions, game_ended=False):
        '''
        This function is used to update the neural network based on the Q_learning_algorithm
        '''

        start_time = time.time()
        alpha = 0.1 # Learning rate
        gamma = 0.95 # Discount factor


        # SA -> State action

        if game_ended:
            expected_return = 0
        else:

            ## Take the next step based on a greedy policy
            action = self.greedy_choice(list_of_actions, game_state)
            expected_return = self.NN_get_expected_return(game_state, [action])[0]

        old_expected_return = self.NN_get_expected_return(self.game_state_history[-1], [self.action_history[-1]])[0]

        reward_list = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])
        reward = np.sum(reward_list)
        self.latest_reward = reward_list

        NN_error = (reward + gamma*expected_return - old_expected_return)**2
        self.NN_error.append(NN_error)
        self.all_returns.append(reward)

        # Defining learning step - Is 0 if the only action available is the terminate action
        learning_step = alpha * (reward + gamma*expected_return - old_expected_return)

        if self.only_terminate_action:
            learning_step = 0



        # Q_learning update
        old_expected_return_updated = old_expected_return + learning_step
        self.all_expected_returns.append(old_expected_return_updated.astype(float)[0])


        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values



        # Print out the learned update for the given action.
        if self.greedy_mode == False:
            # Printing the reward update step.
            self.latest_action = self.action_history[-1]
            self.latest_updated_expected_return = learning_step
            self.latest_action_type = self.game_state_history[-1]["Unique_actions"]
            self.latest_desired_expected_return = self.all_expected_returns[-1]



        self.turns_in_game += 1
        if self.greedy_mode == False:
            # self.update_NN(self.game_state_history[-1], self.action_history[-1], old_expected_return_updated)

            batch_size = 32

            # Every batch_size turns we will update the neural network with the batch_size new datasets
            if self.turns_in_game % batch_size == 0:

                input_matrix = self.game_state_list2NN_input(self.game_state_history[-batch_size:], self.action_history[-batch_size:])
                output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-batch_size:])
                

                self.update_NN_np_mat(input_matrix, output_matrix)



        self.SARSA_update_time.append(time.time() - start_time)


    def choose_action(self, list_of_actions, game_state):

        
        if self.game_state_history == []:
            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(np.random.choice(list_of_actions))
            return self.action_history[-1]
        else:

            if self.greedy_mode:
                action = self.greedy_choice(list_of_actions, game_state)
            else:
                action = self.epsilon_greedy_policy(list_of_actions, game_state, 0.90)


            self.Q_learning_update(game_state, list_of_actions, game_ended=False)



            # Set boolean so the reward function is constricted for the next state
            if len(list_of_actions) == 1 and action == -1:
                self.only_terminate_action = True
            else:
                self.only_terminate_action = False


            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(copy.deepcopy(action))



            #Remove the previous old values of game state and action history
            return action


    def game_end_update(self, game_state):
        '''
        This function is used to update the neural network with the new values
        '''

        self.Q_learning_update(game_state, None, game_ended=True)






class Deep_expected_sarsa(Deep_SARSA):
    
    def expected_sarsa_update(self, game_state, list_of_actions, game_ended=False):
        '''
        This function is used to update the neural network based on the Q_learning_algorithm
        '''

        start_time = time.time()
        alpha = 0.1 # Learning rate
        gamma = 0.95 # Discount factor


        # SA -> State action

        if game_ended:
            expected_return = 0
        else:

            ## Take the next step based on an average.
            best_action = self.greedy_choice(list_of_actions, game_state)
            all_expected_return_actions = self.NN_get_expected_return(game_state, list_of_actions)[0]

            expected_return_best_action = all_expected_return_actions[list_of_actions.index(best_action)]

            average_expected_return_weighted = expected_return_best_action*(1-self.epsilon)
            for action_return in all_expected_return_actions:
                average_expected_return_weighted += action_return*self.epsilon/len(list_of_actions)

            expected_return = average_expected_return_weighted


        old_expected_return = self.NN_get_expected_return(self.game_state_history[-1], [self.action_history[-1]])[0]


        # Save the reward to be sure that they work
        reward_list = self.rf.get_reward_from_state(game_state, self.game_state_history[-1])
        reward = np.sum(reward_list)
        self.latest_reward = reward_list

        NN_error = (reward + gamma*expected_return - old_expected_return)**2
        self.NN_error.append(NN_error)
        self.all_returns.append(reward)


        old_expected_return_updated = old_expected_return + alpha * (reward + gamma*expected_return - old_expected_return)
        self.all_expected_returns.append(old_expected_return_updated.astype(float)[0])


        old_expected_return_updated = np.array(old_expected_return_updated).reshape((1,1))
        # Train the neural network with the new values


        self.turns_in_game += 1


        if self.greedy_mode == False:
            # self.update_NN(self.game_state_history[-1], self.action_history[-1], old_expected_return_updated)

            self.batch_size = 32

            # Every batch_size turns we will update the neural network with the batch_size new datasets
            if self.turns_in_game % self.batch_size == 0:

                input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
                output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])
                self.update_NN_np_mat(input_matrix, output_matrix)

        elif game_ended:
            input_matrix = self.game_state_list2NN_input(self.game_state_history[-1:], self.action_history[-1:])
            output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-1:])
            self.update_NN_np_mat(input_matrix, output_matrix)


        self.SARSA_update_time.append(time.time() - start_time)






    def choose_action(self, list_of_actions, game_state):

        
        if self.game_state_history == []:
            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(np.random.choice(list_of_actions))
            return self.action_history[-1]
        else:

            if self.greedy_mode:
                action = self.greedy_choice(list_of_actions, game_state)
            else:
                action = self.epsilon_greedy_policy(list_of_actions, game_state, 0.7)



            self.expected_sarsa_update(game_state, list_of_actions, game_ended=False)

            if len(list_of_actions) == 1:
                self.only_terminate_action = True
            else:
                self.only_terminate_action = False


            self.game_state_history.append(copy.deepcopy(game_state))
            self.action_history.append(copy.deepcopy(action))

            #Remove the previous old values of game state and action history
            return action


    def game_end_update(self):
        '''
        This function is used to update the neural network with the new values
        '''
        input_matrix = self.game_state_list2NN_input(self.game_state_history[-self.batch_size:], self.action_history[-self.batch_size:])
        output_matrix = self.expected_return_list2NN_output(self.all_expected_returns[-self.batch_size:])
        self.update_NN_np_mat(input_matrix, output_matrix)


        self.expected_sarsa_update(self.game_state_history[-1], list_of_actions=None, game_ended=True)






class random_player:
    def __init__(self, player_name):
        self.rf = Dominion_reward()
        self.file_address = f"reward_history/{player_name}_reward_history.txt"
        self.player_name = player_name
        self.delete_all_previous_history()


    def delete_all_previous_history(self):
        '''
        This funtions opens all the file paths for overwrite, to delete all previous data
        '''

        open_file = open(self.file_address, "w")
        open_file.close()

    def get_name(self):
        return self.player_name

    def get_reward(self, game_state):
        '''
        This function is used to get the reward from the game state
        '''
        reward = self.rf.get_reward_from_state(game_state, game_state)

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


    def notify_game_end(self, game_state):
        ''' [summary]
            This function is used to notify the player that the game has ended
        '''

        pass


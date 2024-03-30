'''
This python file is for training a neural network based on the Q_table_data files
'''


import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

from Player_AI import Deep_SARSA


class Q_table_data_batch_loader:
    '''
    This class is for loading the Q_table_data files in random for training
    '''

    def __init__(self):
        self.trained_indexes = []


    def set_data_loader_path(self, path_input, path_output):
        self.path_output = path_output

        self.path_input = path_input


    def split_data(self, path_input, path_output, test_size=0.2):
        """
        This function takes in the input and output data, and splits them into training and validation set.
        """
        print(path_input)
        input_data = os.listdir(path_input)
        output_data = os.listdir(path_output)

        idx_list = np.arange(len(input_data))

        np.random.shuffle(idx_list)

        input_data_train = input_data[idx_list[:int(len(input_data)*(1-test_size))]]
        output_data_train = output_data[idx_list[:int(len(input_data)*(1-test_size))]]

        input_data_validate = input_data[idx_list[int(len(input_data)*(1-test_size)):]]
        output_data_validate = output_data[idx_list[int(len(input_data)*(1-test_size)):]]

        
        return input_data_train, output_data_train, input_data_validate, output_data_validate



    def load_data(self,path_input, path_output):
        '''
        This function loads the indexes of the input and output data into the class.
        '''
        self.set_data_loader_path(self.path_input, self.path_output)
        input_data_train, output_data_train, input_data_validate, output_data_validate = self.split_data(path_input, path_output)

        self.input_data_train = input_data_train
        self.input_data_validate = input_data_validate

        self.output_data_train = output_data_train
        self.output_data_validate = output_data_validate


        self.ready_2_train_idx = np.arange(len(self.input_data_train))
        np.random.shuffle(self.ready_2_train_idx)



    def get_batch(self, input_data_size, output_data_size, batch_size=16):
        '''
        This function returns a batch of input and output data
        '''

        # Get a batch of data from the indexes that has not been trained on yet.
        if len(self.ready_2_train_idx) < batch_size:
            self.ready_2_train_idx = np.arange(len(self.input_data_train))
            np.random.shuffle(self.ready_2_train_idx)
        
        batch_idx = self.ready_2_train_idx[:batch_size]
        self.ready_2_train_idx = self.ready_2_train_idx[batch_size:]

        train_data_input = np.zeros((0, input_data_size))
        train_data_output = np.zeros((0, output_data_size))


        for idx in batch_idx:
            path = os.path.join(self.path_input, self.input_data_train[idx])
            game_input_data = pickle.load(open(path, 'rb'))

            train_data_input = np.concatenate((train_data_input, game_input_data))

            

            path = os.path.join(self.path_input, self.output_data_train[idx])
            game_input_data = pickle.load(open(path, 'rb'))

            train_data_output = np.concatenate((train_data_output, game_input_data))
        


        return train_data_input, train_data_output


class Deep_RL_trainer:
    '''
    This class is used to train a neural network based on the Q_table_data files
    '''
    def __init__(self, player_AI):
        self.player_AI = player_AI

        self.data_loader = Q_table_data_batch_loader()

    def load_data(self, path_input, path_output):
        self.data_loader.set_data_loader_path(path_input, path_output)
        self.data_loader.load_data(path_input, path_output)
        
    def split_data(self,path_input, path_output, test_size=0.2):
        self.data_loader.split_data(path_input, path_output, test_size=test_size)


    def train(self, epochs=1000, batch_size=32):
        '''
        This function trains a neural network based on the Q_table_data files
        '''
        # The epoch function is shit

        for i in range(epochs):
            in_data, out_data = self.data_loader.get_batch(9000, 1, batch_size=batch_size)

            self.player_AI.update_NN_np_mat(in_data, out_data)
        


trainer = Deep_RL_trainer(Deep_SARSA(player_name="Deep_SARSA_offline_trained"))

input_data_path = "Q_table_data/input_data"
output_data_path = "Q_table_data/output_data"
trainer.load_data(input_data_path, output_data_path)
trainer.split_data(input_data_path, output_data_path)

trainer.train(epochs=1, batch_size=32)
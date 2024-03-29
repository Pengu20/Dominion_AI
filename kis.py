




import numpy as np
import pickle


list = []
file = open("Q_table_data/input_data/input_data.txt", "wb")
val = pickle.dump(list, file)
file.close()


list = []
file = open("Q_table_data/output_data/output_data.txt", "wb")
val = pickle.dump(list, file)
file.close()



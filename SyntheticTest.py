
from DataPreprocessing import DataPreprocessing
#from MyNeuralNetwork import MyNeuralNetwork
import pandas as pd
import numpy as np


    
data_preprocessor = DataPreprocessing()

synthetic_dataset = data_preprocessor.load_dataset("A1-synthetic.txt")
synthetic_dataset = data_preprocessor.standardize_data(synthetic_dataset)
print(synthetic_dataset.head())



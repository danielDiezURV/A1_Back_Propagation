
from DataPreprocessing import DataPreprocessing
#from MyNeuralNetwork import MyNeuralNetwork
import pandas as pd
import numpy as np


    
data_preprocessor = DataPreprocessing()

turbine_dataset = data_preprocessor.load_dataset("A1-turbine.txt")
turbine_dataset = data_preprocessor.scale_dataset(turbine_dataset, 0.1, 0.9)
print(turbine_dataset.head())



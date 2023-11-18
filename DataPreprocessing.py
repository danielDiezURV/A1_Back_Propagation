
import numpy as np
import pandas as pd

class DataPreprocessing:

    def __init__(self):
        self.x_min = 0
        self.x_max = 1
        self.s_min = 0
        self.s_max = 1

    def scale(self, dataset, s_min=0, s_max=1):
        scaled_data = dataset.copy()
        for col_idx in range(dataset.shape[1]):
            col_values = scaled_data[:, col_idx]
            self.x_min = np.min(col_values)
            self.x_max = np.max(col_values)
            self.s_min = s_min
            self.s_max = s_max
            scaled_data[:, col_idx] = [self.s_min + (((self.s_max - self.s_min)/(self.x_max - self.x_min))*(x - self.x_min)) for x in col_values]
        return scaled_data
    
    def inverse_scale(self, dataset):
        scaled_data = dataset.copy()
        for col_idx in range(dataset.shape[1]):
            col_values = scaled_data[:, col_idx]
            scaled_data[:, col_idx] = [((x - self.x_min) / (self.x_max - self.x_min)) * (self.s_max - self.s_min) + self.s_min for x in col_values]
        return scaled_data
    
    def standardize(self, dataset):
        scaled_data = dataset.copy()
        for col_idx in range(dataset.shape[1]):
            col_values = scaled_data.iloc[:, col_idx]
            mean = np.mean(col_values)
            stdev = np.std(col_values)
            scaled_data.iloc[:, col_idx] = col_values.apply(lambda x: (x - mean) / stdev)
        return scaled_data.values


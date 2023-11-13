
import numpy as np
import pandas as pd

class DataPreprocessing:

    def scale_dataset(self, dataset, s_min, s_max):
        for col_idx in range(dataset.shape[1]):
            col_values = dataset.iloc[:, col_idx]
            x_min = min(col_values)
            x_max = max(col_values)
            dataset.iloc[:, col_idx] = col_values.apply(lambda x:  s_min + (((s_max - s_min)/(x_max - x_min))*(x - x_min)))
        return dataset

    def standardize_data(self, dataset):
        for col_idx in range(dataset.shape[1]):
            col_values = dataset.iloc[:, col_idx]
            mean = np.mean(col_values)
            stdev = np.std(col_values)
            dataset.iloc[:, col_idx] = col_values.apply(lambda x: (x - mean) / stdev)
        return dataset
    



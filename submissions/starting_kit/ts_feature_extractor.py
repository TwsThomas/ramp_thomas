import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass
    
    def transform(self, Z):
        """Compute the running average of the last 10 days at the same time
        
        Z is the raw pd.DataFrame
        return x_vector of size 172 (which will be our final prediction here)
        """
        
        Z = Z.fillna(0).drop(columns = 'Id_client')
        
        nb_days = int(Z.shape[0] / 48)
        previous_measure= [-47] + [-47 - i*48 for i in range(1, min([nb_days,10]))]
        
        X_array = Z.iloc[previous_measure].values
        x_vector = np.mean(X_array, axis = 0)
    
        return x_vector    
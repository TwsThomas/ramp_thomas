import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass
    
    def transform(self, Z):
        """Naive : predict the next timestamp from the previous one
        
        Z is the raw pd.DataFrame
        return x_vector of size 172 (which will be our final prediction here)
        """
        
        Z = Z.fillna(0).drop(columns = 'Id_client')
        x_vector = Z.iloc[-1].values
        
        return x_vector    

from __future__ import division, print_function
import numpy as np
import pandas as pd
import rampwf as rw
import pickle

BURN_IN = 480 # minimum rows of X (480 = 10 days)
BATCH_SIZE = 600 # number of instances in one CV batch
N_CV = 3 # number of CV batch
RATIO_TRAIN_TEST = .6
FAST_MODE = True #False # to debug quickly

problem_title = 'Thomas Households Consumptions'
Predictions = rw.prediction_types.make_regression(label_names = np.arange(172))

def get_train_data(path='.', is_train = True):
    """return the dataset
    
    the first RATIO_TRAIN_TEST part is for the train set
    the second RATIO_TRAIN_TEST part is for the test set.
    """

    try:
        with open('data/linky.pickle','rb') as file:
            Z = pickle.load(file)
    except FileNotFoundError:
        print('reading raw data')
        Z = pd.read_excel("data/linky.xlsx")
        try:
            with open('data/linky.pickle','wb') as file:
                pickle.dump(Z, file)
                print('raw data pickled')
        except:
            print("Couldn't pickle raw file")
        
   
    if is_train:
        # we are in train data
        start_row = 0
        end_row = int(Z.shape[0] * RATIO_TRAIN_TEST)
    else:
        start_row = int(Z.shape[0] * RATIO_TRAIN_TEST)  
        end_row = Z.shape[0]   
        
    Z = Z.iloc[start_row:end_row]
    if FAST_MODE:
        Z = Z.iloc[:3000]
     
    # print('get raw data, from row {} to row {}'.format(start_row, end_row))
    Y = Z.drop(columns = 'Id_client')
    Y = Y.fillna(0).values # hack to consider all columns and all rows (we can also drops rows or columns with NaN, or use an Imputer)
    return Z, Y

def get_test_data(path='.'):
    return get_train_data(path='.', is_train = False)


class TimeSeriesCV(object):
    def __init__(self, n_cv=10, n_indices_per_CV = 12):
        self.n_cv = n_cv
        self.n_indices_per_CV = n_indices_per_CV # nb of data point in each cv batch

    def get_cv(self, X, y_array):
        # yield batch of indices

        indices_to_train = np.arange(BURN_IN, X.shape[0]) #
        np.random.seed(42)
        for _ in range(self.n_cv):
            train_test_is = np.random.choice(indices_to_train, size = self.n_indices_per_CV * 2, replace = False)
            train_is = train_test_is[:self.n_indices_per_CV]
            test_is = train_test_is[self.n_indices_per_CV:]
            yield train_is, test_is


cv = TimeSeriesCV(n_cv=N_CV, n_indices_per_CV = BATCH_SIZE)
get_cv = cv.get_cv


class TimeSeriesWorkflow(object):
    # New workflow for time serie.

    def __init__(self, workflow_element_names = ['ts_feature_extractor', 'regressor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, Z, y_array, train_is=None):
        ts_feature_extractor = import_file(module_path, self.element_names[0])
        regressor = import_file(module_path, self.element_names[1])
        fe = ts_feature_extractor.FeatureExtractor()

        # The singularity here is that
        # each Z_matrix is the data from time 0 to time t
        # (Then the prediction need to be at time t+1)
        X_array = []
        for t in train_is:
            Z_matrix = Z.iloc[:t] # raw data from time 0 to time t
            x_vector = fe.transform(Z_matrix) 
            X_array.append(x_vector)

        X_array = np.array(X_array)
        reg = regressor.Regressor()
        reg.fit(X_array, y_array[train_is])
        return (fe, reg)

    def test_submission(self, trained_model, Z):

        (fe, reg) = trained_model

        X_array = []
        # Loop not efficient here :s
        # But mandatory in order that X_array (thus y_pred) and y_test share the same size.
        for t in range(0, Z.shape[0]): 
            if t < BURN_IN:
                t = BURN_IN # need at least BURN_IN data in Z_matrix.
            Z_matrix = Z.iloc[:t]
            x_vector = fe.transform(Z_matrix) 
            X_array.append(x_vector)

        X_array = np.array(X_array)
        y_pred = reg.predict(X_array)
        return y_pred

def import_file(module_path, filename):
    import imp
    import os

    submitted_file = os.path.join(module_path, filename + '.py')
    submitted_path = (
        '.'.join(list(os.path.split(module_path)) + [filename])
        .replace('/', ''))
    return imp.load_source(submitted_path, submitted_file)


workflow = TimeSeriesWorkflow() # without look_ahead as in El Nino
score_types = [
    rw.score_types.RMSE(name='rmse', precision=3),
]


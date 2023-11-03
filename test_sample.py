from utils import *
import os

def test_for_hparam_cominations_count():

    gamma = [0.001, 0.01, 0.1, 1]
    C = [1, 10, 100, 1000]
    h_params={}
    h_params['gamma'] = gamma
    h_params['C'] = C
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    assert len(h_params_combinations) == len(gamma) * len(C)

def dummy_hyperparameter():
    gamma = [0.001, 0.01]
    C = [1]
    h_params={}
    h_params['gamma'] = gamma
    h_params['C'] = C
    h_params_combinations = get_hyperparameter_combinations(h_params)
    return h_params_combinations

def dummy_data():
    X, y = read_digits()
    
    X_train = X[:100, :]
    y_train = y[:100]

    X_dev = X[:50, :]
    y_dev = y[:50]

    return X_train, X_dev, y_train, y_dev 

def test_for_hparam_cominations_values():    
    h_params_combinations = dummy_hyperparameter()
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

def test_model_saving():
    X_train, X_dev, y_train, y_dev = dummy_data()
    h_params_combinations = dummy_hyperparameter()

    _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)   

    assert os.path.exists(best_model_path)

def test_data_splitting():
    X, y = read_digits()
    
    X = X[:100, :]
    y = y[:100]
    
    test_size = .1
    dev_size = .6

    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == len(y_train) == 30) 
    assert (len(X_test) == len(y_test) == 10)
    assert  ((len(X_dev) == len(y_dev) == 60))
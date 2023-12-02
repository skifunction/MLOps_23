from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, tree, datasets, metrics
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    size = len(X)
    X = X.reshape((size, -1))

    #Answer-1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = digits.target
    return X, y 

def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations

def split_data(x, y, test_size, random_state=30):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, shuffle = True, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test =  split_data(X, y, test_size=test_size, random_state=1)
    print(f"train + dev data size = {len(Y_train_Dev)} test size = {len(y_test)}")
    
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=42)
        
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    elif model_type == "tree":
        clf = tree.DecisionTreeClassifier
    elif model_type == "logistic":
        clf = LogisticRegression
    
    model = clf(**model_params)

    model.fit(x, y)
    return model

def tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combinations, model_type="svm"):
    
    best_accuracy = -1
    best_model_path = ""


    for h_params in h_params_combinations:


        model = train_model(X_train, y_train, h_params, model_type=model_type)      
        current_accuracy = predict_and_eval(model, X_dev, y_dev)
        
        if current_accuracy > best_accuracy and model_type != 'logistic':
            best_accuracy = current_accuracy
            best_hparams = h_params
            best_model_path = f"./models/D23CSA003_lr_" +"_".join([f"{a}:{b}" for a, b in h_params.items()]) + ".joblib"
            # best_model_path = f"./models/D23CSA003_lr_{model_type}.joblib"
            best_model = model
        else :
            best_model_path = f"./models/D23CSA003_lr_" +"_".join([f"{b}" for _, b in h_params.items()]) + ".joblib"
                # best_model_path = f"./models/D23CSA003_lr_{model_type}.joblib"
            best_model = model
            dump(best_model, best_model_path)

    dump(best_model, best_model_path) 

    return h_params, best_model_path, best_accuracy 

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted)
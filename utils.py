from sklearn import datasets, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def data():
    return datasets.load_digits()

def split_train_dev_test(X, y, test_size, dev_size):
    train_set = test_size + dev_size
    dev_size = dev_size/(1-test_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= test_size, shuffle=False
    )

    #print(X_train.shape, y_train.shape)

    X_train, dev_x, y_train, dev_y = train_test_split(
        X_train, y_train, test_size=dev_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test, dev_x, dev_y

def training_data(x, y, parameters):
    clf = svm.SVC(gamma=parameters['gamma'], C=parameters['C'])
    clf.fit(x, y)
    return clf

def Decision_tree(x, y, parameters):
    clf = DecisionTreeClassifier(max_depth=parameters['max_depth'], min_samples_split=parameters['min_samples_split'])
    clf.fit(x, y)
    return clf

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    evaluation = metrics.accuracy_score(y_pred = predicted, y_true=y_test)
    return evaluation

def tune_hparams_SVM(X_train, y_train, dev_x, dev_y, list_of_all_param_combination):
    best_accuracy = -1
    best_model=None
    best_hparams = None

    for i in list_of_all_param_combination[0]:
        for j in list_of_all_param_combination[1]:
            model = training_data(X_train, y_train, {'gamma': i,'C': j})

            val_accuracy = predict_and_eval(model, dev_x, dev_y)
            
            if val_accuracy > best_accuracy:
                best_hparams = [i, j]
                best_accuracy = val_accuracy
                best_model = model
    
    return best_hparams, best_model, best_accuracy  
    
def tune_hparams_DT(X_train, y_train, dev_x, dev_y, list_of_all_param_combination):
    best_accuracy = -1
    best_model=None
    best_hparams = None

    for i in list_of_all_param_combination[0]:
        for j in list_of_all_param_combination[1]:
            model = Decision_tree(X_train, y_train, {'max_depth': i,'min_samples_split': j})

            val_accuracy = predict_and_eval(model, dev_x, dev_y)
            
            if val_accuracy > best_accuracy:
                best_hparams = [i, j]
                best_accuracy = val_accuracy
                best_model = model
    
    return best_hparams, best_model, best_accuracy 
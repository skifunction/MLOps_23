from utils import *

epochs = 5
X, y = read_digits()

classifier_param_dict = {}

gamma = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C = [0.1, 1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma
h_params['C'] = C
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_params_combinations

max_depth = [5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations


solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
h_params = {}
h_params['solver'] = solver
h_params_logistic_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['logistic'] = h_params_logistic_combinations

results = []

# test_sizes =  [0.1, 0.2, 0.3]
# dev_sizes  =  [0.1, 0.2, 0.3]

test_sizes =  [0.2]
dev_sizes  =  [0.1]


for cur_run_i in range(epochs):
    
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            
            train_size = 1 - test_size - dev_size                
            
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

            for model_type in classifier_param_dict:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
                y_dev, current_hparams, model_type)        
       
                best_model = load(best_model_path) 

                test_acc = predict_and_eval(best_model, X_test, y_test)
                train_acc = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy

                print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc))
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)

print(pd.DataFrame(results).groupby('model_type').describe().T)
import pdb
from utils import *

digits = data()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gamma = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
c = [1, 1.02, 1.04, 1.07, 1.081]

possible_combination = []

possible_combination.append(gamma)
possible_combination.append(c)

test_size = [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]

for test_s in test_size:
    for dev_s in dev_size:
        train_s = 1 - dev_s - test_s

        X_train, X_test, y_train, y_test, dev_x, dev_y = split_train_dev_test(data, digits.target, test_s, dev_s)
        
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, dev_x, dev_y, possible_combination)
        
        print(f"test_size={test_s} dev_size={dev_s} train_size={train_s:.0f} train_acc={best_accuracy:.2f} dev_acc={best_accuracy:.2f} test_acc={best_accuracy:.2f}")
        print(f"Best Hyperparameters for this run: ( gamma : {best_hparams[0]} , C : {best_hparams[1]} )")
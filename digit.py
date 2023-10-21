import pdb
from utils import *

digits = data()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gamma = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
c = [1, 1.02, 1.04, 1.07, 1.081]

max_depth = [5, 10, 15]
min_samples_split = [2, 3, 4, 5]

possible_combination_SVM = []
possible_combination_DT = []

possible_combination_SVM.append(gamma)
possible_combination_SVM.append(c)

possible_combination_DT.append(max_depth)
possible_combination_DT.append(min_samples_split)

test_size = [0.2]
dev_size = [0.2]

for test_s in test_size:
    for dev_s in dev_size:
        train_s = 1 - dev_s - test_s

        X_train, X_test, y_train, y_test, dev_x, dev_y = split_train_dev_test(data, digits.target, test_s, dev_s)
        
        pred_svm, best_hparams, best_model, best_accuracy = tune_hparams_SVM(X_train, y_train, dev_x, dev_y, possible_combination_SVM)
        pred_DT, best_hparams_DT, best_model_DT, best_accuracy_DT = tune_hparams_DT(X_train, y_train, dev_x, dev_y, possible_combination_DT)
        
        print(f"For SVM test_size={test_s} dev_size={dev_s} train_size={train_s:.0f} train_acc={best_accuracy:.2f} dev_acc={best_accuracy:.2f} test_acc={best_accuracy:.2f}")
        print(f"Best Hyperparameters for SVM this run: ( gamma : {best_hparams[0]} , C : {best_hparams[1]} )")
        
        print(f"For decision tree test_size={test_s} dev_size={dev_s} train_size={train_s:.0f} train_acc={best_accuracy_DT:.2f} dev_acc={best_accuracy_DT:.2f} test_acc={best_accuracy:.2f}")
        print(f"Best Hyperparameters for Decision tree this run: ( criterion : {best_hparams_DT[0]} , Splitter : {best_hparams_DT[1]} )")

confusion_SVM = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_svm)
confusion_SVM.figure_.suptitle("Confusion Matrix for SVM")
print(f"Confusion matrix:\n{confusion_SVM.confusion_matrix}")


confusion_DT = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_DT)
confusion_DT.figure_.suptitle("Confusion Matrix for DT")
print(f"Confusion matrix:\n{confusion_DT.confusion_matrix}")

from decisiontreelearner import DecisionTreeLearner
from utils import cross_validation, remove_values


def test(dataset):
    print("Testing "+ dataset.name + " dataset:")
    print("")
    for i in range(4):
        if i == 0:
            fold_errT, fold_errV = cross_validation(DecisionTreeLearner, dataset, k=10, trials=5)
            print("Testing with no missing values")


        elif i == 1:
            dataset_ = remove_values(dataset, p=0.1)
            fold_errT, fold_errV = cross_validation(DecisionTreeLearner, dataset_, k=10, trials=5)
            print("Testing with 10% missing values")

        elif i == 2:
            dataset_ = remove_values(dataset, p=0.2)
            fold_errT, fold_errV = cross_validation(DecisionTreeLearner, dataset_, k=10, trials=5)
            print("Testing with 20% missing values")

        else:
            dataset_ = remove_values(dataset, p=0.5)
            fold_errT, fold_errV = cross_validation(DecisionTreeLearner, dataset_, k=10, trials=5)
            print("Testing with 50% missing values")

        fold_errT = '%.4f' % (round(fold_errT, 6) * 100)
        fold_errV = '%.4f' % (round(fold_errV, 6) * 100)

        print("Training errors: " + str(fold_errT) + "%")
        print("Validation errors: " + str(fold_errV) + "%")
        print("")

from utils import cross_validation, remove_values, handle_missing_values


def test(dataset, trials=20):
    print("Testing " + dataset.name + " dataset:")
    print("")
    for i in range(4):
        if i == 0:
            print("Testing with no missing values")
            fold_errT, fold_errV = cross_validation(dataset, k=10, trials=trials)

        elif i == 1:
            print("Testing with 10% missing values")
            dataset_ = remove_values(dataset, p=0.1)
            handle_missing_values(dataset_)
            fold_errT, fold_errV = cross_validation(dataset_, k=10, trials=trials)

        elif i == 2:
            print("Testing with 20% missing values")
            dataset_ = remove_values(dataset, p=0.2)
            handle_missing_values(dataset_)
            fold_errT, fold_errV = cross_validation(dataset_, k=10, trials=trials)

        else:
            print("Testing with 50% missing values")
            dataset_ = remove_values(dataset, p=0.5)
            handle_missing_values(dataset_)
            fold_errT, fold_errV = cross_validation(dataset_, k=10, trials=trials)

        fold_errT = '%.4f' % round(fold_errT * 100, 6)
        fold_errV = '%.4f' % round(fold_errV * 100, 6)

        print("Training errors: " + str(fold_errT) + "%")
        print("Validation errors: " + str(fold_errV) + "%")
        print("")

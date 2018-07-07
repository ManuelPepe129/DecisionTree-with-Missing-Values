import csv
import copy
import random
from dataset import DataSet
from collections import Counter
from decisiontreelearner import DecisionTreeLearner


def parser(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data


def remove_values(set, p):
    examples = copy.deepcopy(set.examples)
    for i in range(len(examples)):
        for j in range(len(examples[i])):
            if j != set.target and random.random() <= p:
                examples[i][j] = None
    return DataSet(examples=examples, attrs=set.attrs, values=set.values, target=set.target)


# _____________________________________________________________________________
# Functions for testing learners on examples


def err_ratio(predict, dataset, examples=None, verbose=0):
    """Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct"""
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = predict(dataset.sanitize(example))
        if output == desired:
            right += 1
            if verbose >= 2:
                print('   OK: got {} for {}'.format(desired, example))
        elif verbose:
            print('WRONG: got {}, expected {} for {}'.format(
                output, desired, example))
    return 1 - (right / len(examples))


def train_test_split(dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder."""
    start = int(start)
    end = int(end)
    examples = dataset.examples
    train = examples[:start] + examples[end:]
    val = examples[start:end]
    return train, val


def cross_validation(dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; if trials>1, average over several shuffles.
    Returns Training error, Validataion error"""
    k = k or len(dataset.examples)
    if trials > 1:
        trial_errT = 0
        trial_errV = 0
        for t in range(trials):
            errT, errV = cross_validation(dataset, k, trials=1)
            trial_errT += errT
            trial_errV += errV
        return trial_errT / trials, trial_errV / trials
    else:
        fold_errT = 0
        fold_errV = 0
        n = len(dataset.examples)
        examples = dataset.examples
        for fold in range(k):
            random.shuffle(dataset.examples)
            train_data, val_data = train_test_split(dataset, fold * (n / k),
                                                    (fold + 1) * (n / k))
            dataset.examples = train_data
            h = DecisionTreeLearner(dataset)
            fold_errT += err_ratio(h, dataset, train_data)
            fold_errV += err_ratio(h, dataset, val_data)

            # Reverting back to original once test is completed
            dataset.examples = examples
        return fold_errT / k, fold_errV / k


def handle_missing_values(dataset):
    examples = dataset.examples
    values_per_attribute = []
    for i in range(len(dataset.values)):
        values_per_attribute.append([])
    for example in examples:
        for i in range(len(example)):
            values_per_attribute[i].append(example[i])

    occurrency_counter = []
    most_common_values = []
    for i in range(len(values_per_attribute)):
        occurrency_counter.append(Counter(values_per_attribute[i]))
        occurrency_counter[i].pop(None, None)
        most_common_values.append(occurrency_counter[i].most_common(1))
    for example in examples:
        for i in range(len(example)):
            if example[i] is None:
                example[i] = most_common_values[i][0][0]

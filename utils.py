import csv
import copy
import random
from dataset import DataSet


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
        for j in range(len(examples[i]) - 1):
            if random.random() <= p:
                examples[i][j] = None
    return DataSet(examples=examples, attrs=set.attrs, values=set.values, target=set.target)


def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


# ______________________________________________________________________________

# argmin and argmax


identity = lambda x: x

argmin = min
argmax = max


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return argmin(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return argmax(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items

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


def cross_validation(learner, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; if trials>1, average over several shuffles.
    Returns Training error, Validataion error"""
    k = k or len(dataset.examples)
    if trials > 1:
        trial_errT = 0
        trial_errV = 0
        for t in range(trials):
            errT, errV = cross_validation(learner,dataset, k, trials=1)
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
            h = learner(dataset)
            fold_errT += err_ratio(h, dataset, train_data)
            fold_errV += err_ratio(h, dataset, val_data)

            # Reverting back to original once test is completed
            dataset.examples = examples
        return fold_errT / k, fold_errV / k

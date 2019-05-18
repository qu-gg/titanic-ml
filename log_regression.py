import numpy as np
import pandas as pd
import time

# Hyperparams
NUM_FEATURES = 4
THRESHOLD = .5


# Importing the dataset and labels
dataset = pd.read_csv("data/dataset.csv").to_numpy()
labels = pd.read_csv("data/labels.csv").to_numpy()

# Weights
w = np.ones([NUM_FEATURES, 1])

# Bias
b = np.ones([1])

# Learning rate
lr = .0005


def sigmoid(x):
    # sigmoid function for logistic regression
    return 1.0 / (1.0 + np.exp(-x))


def loss_funct(y, y_hat):
    # defined loss function to optimize
    return -(np.multiply(y, np.log(y_hat)) + np.multiply((1.0 - y), np.log(1.0 - y_hat)))


def test(w, b):
    # tests the accuracy of the network as a ratio of correct guesses over total
    Z = np.dot(w.T, dataset.T) + b
    A = sigmoid(Z)
    A = np.rint(A.T)

    cor = np.count_nonzero(A == labels)
    print("Accuracy: {}".format(cor / len(dataset)))


def vectorized_cost(w, b):
    m = len(dataset) - 1

    # Making preds over the whole dataset
    Z = np.dot(w.T, dataset.T) + b
    A = sigmoid(Z)

    J = - (np.multiply(labels, np.log(A)) + np.multiply(1.0 - labels, np.log(1.0 - A)))
    J = J.sum()

    dZ = np.subtract(A.T, labels)
    dW = np.multiply(dataset, dZ).sum(axis=0)
    dW = np.reshape(dW, [1, 4])
    dB = dZ.sum()

    return J / m, dW / m, dB / m


def test_pred(w, b):
    # tests on the test set and builds the relevant csv submission file
    test = pd.read_csv("data/test_set.csv").to_numpy()
    test_ids = pd.read_csv("data/test_ids.csv").to_numpy()
    test_ids = np.append([892], test_ids)

    # Calculating test results
    Z = np.dot(w.T, test.T) + b
    A = sigmoid(Z)
    A = np.rint(A.T)
    A = A.astype(np.int)
    A = np.reshape(A, [418])

    # Outputting to CSV
    name = input("Enter submission file name: ")

    col = {"PassengerId": test_ids.tolist(), "Survived": A.tolist()}
    df = pd.DataFrame.from_dict(col)
    df.to_csv("submissions/{}.csv".format(name), index=False)


# Training loop
test(w, b)

for epoch in range(10):
    print("Epoch: ", epoch)

    # Shuffling the dataset in unison
    indices = np.arange(dataset.shape[0] - 1)
    np.random.shuffle(indices)

    # Vectorized implementation
    dataset = dataset[indices]
    labels = labels[indices]

    for i in range(len(dataset) - 1):
        x = np.reshape(dataset[i], [NUM_FEATURES, 1])
        y = labels[i]

        y_hat = sigmoid(np.matmul(w.T, x) + b)
        loss = loss_funct(y, y_hat)

        # Compute gradients, update weights
        cost, dw, db = vectorized_cost(w, b)

        w = w - np.multiply(lr, dw.T)
        b = b - np.multiply(lr, db.T)

        if i % 800 == 0:
            print("Loss at step {}: {}".format(i, loss))
            print("Cost: {}".format(cost))


print("Weights: ", w.T, "Bias: ", b)

test(w, b)
test_pred(w, b)

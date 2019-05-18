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
lr = .002


def sigmoid(x):
    # sigmoid function for logistic regression
    return 1.0 / (1.0 + np.exp(-x))


def loss_funct(y, y_hat):
    # defined loss function to optimize
    return -(np.multiply(y, np.log(y_hat)) + np.multiply((1.0 - y), np.log(1.0 - y_hat)))


def backprop(x, y, y_hat):
    # calculated derivative for regression, applied to the weights/bias
    dldz = y_hat - y
    dw = np.multiply(x, dldz)
    db = dldz
    return dw, db


def test(w, b):
    # tests the accuracy of the network as a ratio of correct guesses over total
    Z = np.dot(w.T, dataset.T) + b
    A = sigmoid(Z)
    A = np.rint(A.T)

    cor = np.count_nonzero(A == labels)
    print("Accuracy: {}".format(cor / len(dataset)))


def vectorized_cost(A, w, b):
    start = time.time()

    m = len(dataset) - 1

    J = - (np.multiply(labels, np.log(A)) + np.multiply(1.0 - labels, np.log(1.0 - A)))
    J = J.sum()

    dZ = np.subtract(A.T, labels)
    dW = np.multiply(dataset, dZ).sum(axis=0)
    dB = dZ.sum()
    end = time.time()

    print("Vectorized: ", J.sum() / m, dW / m, dB / m, "Time: ", end - start)
    return J.sum() / m, dW / m, dB / m


def cost_funct(w, b):
    # cost function of the model, taking the average of all gradients and losses
    m = len(dataset) - 1
    dw = db = J = 0

    # vectorized
    Z = np.dot(w.T, dataset.T) + b
    A = sigmoid(Z)
    vectorized_cost(A, w, b)

    # explicit for
    for i in range(m):
        x = np.reshape(dataset[i], [NUM_FEATURES, 1])
        y = labels[i]

        y_hat = sigmoid(np.matmul(w.T, x) + b)
        J += loss_funct(y, y_hat)

        tup = backprop(x, y, y_hat)
        dw += tup[0]
        db += tup[1]

    cost = J / m
    dw = dw / m
    db = db / m
    return cost, dw, db


def test_pred(w, b):
    # tests on the test set and builds the relevant csv submission file
    test = pd.read_csv("data/test_set.csv").to_numpy()
    test_ids = pd.read_csv("data/test_ids.csv").to_numpy()

    col = {"PassengerId": list(), "Survived": list()}

    # Attempt at vectorized version
    Z = np.dot(w.T, test.T) + b
    A = sigmoid(Z)

    col["PassengerId"] = test_ids
    col["Survived"] = A.T
    col["Survived"] = col["Survived"].round()

    name = input("Enter submission file name: ")
    df = pd.DataFrame(col)
    df.to_csv("submissions/{}.csv".format(name), index=False)



# Training loop
test(w, b)

for _ in range(1):
    # Shuffling the dataset in unison
    indices = np.arange(dataset.shape[0] - 1)
    np.random.shuffle(indices)

    dataset = dataset[indices]
    labels = labels[indices]

    for i in range(len(dataset) - 1):
        x = np.reshape(dataset[i], [NUM_FEATURES, 1])
        y = labels[i]

        y_hat = sigmoid(np.matmul(w.T, x) + b)
        loss = loss_funct(y, y_hat)

        # Compute gradients, update weights
        cost, dw, db = cost_funct(w, b)

        w = w - np.multiply(lr, dw)
        b = b - np.multiply(lr, db)

        if i % 800 == 0:
            print("Loss at step {}: {}".format(i, loss))
            print("Cost: {}".format(cost))
        break

test(w, b)
test_pred(w, b)

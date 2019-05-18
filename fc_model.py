import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


dataset = pd.read_csv("data/dataset.csv", index_col=0).to_numpy()
labels = pd.read_csv("data/labels.csv", index_col=0).to_numpy()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


net = Net()
entropy = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005)


def test_set():
    correct = 0
    for i in range(len(dataset) - 1):
        x = torch.from_numpy(dataset[i]).float()
        y = labels[i]
        y_hat = net(x)

        if y_hat > 0.8:
            y_hat = 1
        else:
            y_hat = 0

        if y == y_hat:
            correct += 1

    print("Accuracy: {}".format(correct / len(dataset)))


def train():
    for _ in range(200):
        for i in range(len(dataset) - 1):
            optimizer.zero_grad()

            x = torch.from_numpy(dataset[i]).float()
            y = torch.from_numpy(labels[i]).float()

            y_hat = net(x)
            loss = entropy(y_hat, y)
            loss.backward()
            optimizer.step()

            if i % 800 == 0:
                print("Loss at step {}: {}".format(i, loss))


test_set()
train()
test_set()
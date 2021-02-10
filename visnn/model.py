import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DIRNAME = os.path.abspath(__file__ + "/../vis_data/")
BATCH_SIZE = 5
EPOCHS = 50
LEARNING_RATE = 0.00001
MOMENTUM = 0.9

NLAYERS = 3
LAYERDIMS = [3, 30, 15, 1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VisDataset(Dataset):
    """
    Pytorch dataset for the visibility uncertainty values

        X: relative position of obstacles
        y: the visibility error
    """
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.X_data)

class VisNet(torch.nn.Module):
    """
    Neural net for regression on visibility uncertainty based on
    vehicle relative position (x, y, z)
    """
    def __init__(self):
        super(VisNet, self).__init__()

        self.linears = torch.nn.ModuleList([])

        for i in range(NLAYERS):
            layer = torch.nn.Linear(LAYERDIMS[i], LAYERDIMS[i + 1])
            self.linears.append(layer)

        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        for i in range(NLAYERS):
            x = self.relu(self.linears[i](x))
        return x

def read_data(filename):
    """
    Reads the visibility data (in csv format) from the filename
    inside vis_data folder

    Returns a pandas dataframe split into train (70) and test (30)

    normalize inputs/outputs? or no?

    """
    filepath = os.path.join(DIRNAME, filename)
    df = pd.read_csv(filepath)
    #print(df.columns)
    df = df[df.ERROR < 1000000.0] # temporary? ignores all rows with high errors to avoid
                                # exploding loss values...

    X = df.iloc[:, -4:-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    xscaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = xscaler.fit_transform(X_train)
    X_test = xscaler.transform(X_test)

    yscaler = MinMaxScaler()
    print(y_train.values.reshape(-1, 1))
    y_train = yscaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = yscaler.transform(y_test.values.reshape(-1, 1))

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train, X_test, y_train, y_test = X_train.astype(float), X_test.astype(float), \
                                        y_train.astype(float), y_test.astype(float)

    return X_train, X_test, y_train, y_test

def get_datasets(X_train, X_test, y_train, y_test):
    train_dataset = VisDataset(torch.from_numpy(X_train).float(),
                                torch.from_numpy(y_train).float())
    test_dataset = VisDataset(torch.from_numpy(X_test).float(),
                                torch.from_numpy(y_test).float())

    return train_dataset, test_dataset

def get_dataloader(dataset, batch_size):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=2)
    return loader

def print_stats(epoch, step, loss_sum):
    loss_str = "EPOCH: {:2d}, STEP: {:5d}, LOSS: {:.4f}".format(epoch, step, loss_sum)
    print(loss_str)

def run_nn(writer, loader):
    # define neural net architecture
    net = VisNet()
    print(net.linears)

    criterion = torch.nn.MSELoss() # (y - yhat)^2
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    print("============== BEGIN TRAINING =================")
    # train the neural net
    for epoch in range(EPOCHS):
        for step, data in enumerate(loader):
            # find X, y
            X_batch, y_batch = data

            optimizer.zero_grad()   # zero the optim gradients

            prediction = net(X_batch)
            y_batch = y_batch.view(-1, 1)
            loss = criterion(prediction, y_batch)
            loss.backward()         # backprop
            optimizer.step()        # apply gradients

            niter = epoch * len(loader) + step
            writer.add_scalar("Loss/train", loss, niter)
            print_stats(epoch + 1, step + 1, loss)


if __name__=="__main__":
    X_train, X_test, y_train, y_test = read_data("vis00.csv")

    print("X TRAINING")
    print(X_train)
    print("Y TRAINING")
    print(y_train)

    train_dataset, test_dataset = get_datasets(X_train, X_test, y_train, y_test)
    train_loader = get_dataloader(train_dataset, BATCH_SIZE)
"""
    writer = SummaryWriter()
    run_nn(writer, train_loader)
    writer.flush()
    writer.close()
    """

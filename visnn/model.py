import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pylot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

DIRNAME = os.path.abspath(__file__ + "/../vis_data/")
BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 0.005
MOMENTUM = 0.9

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

def read_data(filename):
    """
    Reads the visibility data (in csv format) from the filename
    inside vis_data folder

    Returns a pandas dataframe split into train (70) and test (30)

    normalize inputs/outputs? or no?

    """
    filepath = os.path.join(DIRNAME, filename)
    df = pd.read_csv(filepath)

    X = df.iloc[:, -4:-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

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
    net = torch.nn.Sequential(
            torch.nn.Linear(3, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 1),
            #torch.nn.ReLU() # clip off at 0?
        ).to(device)

    criterion = torch.nn.MSELoss() # (y - yhat)^2
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # train the neural net
    for epoch in range(EPOCHS):
        for step, data in enumerate(loader):
            # find X, y
            X_batch, y_batch = data

            optimizer.zero_grad()   # zero the optim gradients

            prediction = net(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()         # backprop
            optimizer.step()        # apply gradients

            niter = (epoch + 1) * (step + 1)
            writer.add_scalar("Loss/train", loss, niter)
            print_stats(epoch + 1, step + 1, loss)

if __name__=="__main__":
    X_train, X_test, y_train, y_test = read_data("vis.csv")
    train_dataset, test_dataset = get_datasets(X_train, X_test, y_train, y_test)
    train_loader = get_dataloader(train_dataset, BATCH_SIZE)

    print("============== BEGIN TRAINING =================")
    writer = SummaryWriter()
    run_nn(writer, train_loader)
    writer.flush()
    writer.close()




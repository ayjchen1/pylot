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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
#print("USING DEVICE: ", device)

DIRNAME = os.path.abspath(__file__ + "/../vis_data/")
BATCH_SIZE = 5
EPOCHS = 10000
LEARNING_RATE = 0.0001
MOMENTUM = 0.9

NLAYERS = 3
LAYERDIMS = [3, 100, 50, 1]

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

        # linear(3, 30), relu, linear(30, 15), relu, linear(15, 1)

    def forward(self, x):
        for i in range(NLAYERS - 1):
            x = self.relu(self.linears[i](x))

        x = self.linears[NLAYERS - 1](x) # no relu on last layer
        # add last layer (sigmoid): x -> 0 - 1
        #x = torch.sigmoid(x)
        return x

def normalize_data(X_train, X_test, y_train, y_test):
    xscaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = xscaler.fit_transform(X_train)
    X_test = xscaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def split_data(X, y, test_fraction):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)

    #X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train, X_test, y_train, y_test = X_train.astype(float), X_test.astype(float), \
                                        y_train.astype(float), y_test.astype(float)

    return X_train, X_test, y_train, y_test

def process_df(df):
    THRESHOLD = 1000
    def squash_01(row):
        """
        Squeezes a row's vis error value into 0.001 (close to zero) or 1 (far away)
        """
        if (row['ERROR'] > THRESHOLD):
            return 1
        else:
            return 0

    #df['ERROR'] = df.apply(squash_01, axis=1)

    df = df[df.ERROR < 1000.0] # temporary? ignores all rows with high errors to avoid
    df = df[df.CAMERA == 'FRONT-eval']
    df.reset_index(drop=True)
    # df = df[df.CAMERA == 'BACK-eval']

    return df

def prepare_data(filename):
    """
    Reads the visibility data (in csv format) from the filename
    inside vis_data folder

    Returns a pandas dataframe split into train (70) and test (30)

    normalize inputs/outputs? or no?

    """
    filepath = os.path.join(DIRNAME, filename)
    df = pd.read_csv(filepath)
    df = process_df(df)

    X = df.iloc[:1000, -4:-1]
    y = df.iloc[:1000, -1]
    print(X)

    return split_data(X, y, 0.3)

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

def run_nn(writer, trainloader, testloader):
    # define neural net architecture
    net = VisNet().to(device)
    print(net)

    criterion = torch.nn.MSELoss() # (y - yhat)^2
    #criterion = torch.nn.BCELoss() # binary loss
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print("============== BEGIN TRAINING =================")
    # train the neural net
    for epoch in range(EPOCHS):
        trainloss = 0
        valloss = 0

        # training
        for step, data in enumerate(trainloader):
            # find X, y
            X_batch, y_batch = data
            y_batch = torch.clamp(y_batch, 0, 100)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()   # zero the optim gradients

            prediction = net(X_batch)

            y_batch = y_batch.view(-1, 1)
            loss = criterion(prediction, y_batch)

            loss.backward()         # backprop
            optimizer.step()        # apply gradients

            trainloss += loss

        avg_trainloss = trainloss / len(trainloader)
        writer.add_scalar("Loss/train", avg_trainloss, epoch)

        ##################################################################

        # compute validation loss
        for step, data in enumerate(testloader):
            # find X, y
            X_batch, y_batch = data
            y_batch = torch.clamp(y_batch, 0, 100)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            prediction = net(X_batch)

            y_batch = y_batch.view(-1, 1)
            loss = criterion(prediction, y_batch)

            valloss += loss

        avg_valloss = valloss / len(testloader)
        writer.add_scalar("Loss/validation", avg_valloss, epoch)

        print("EPOCH: {:2d}, TRAIN LOSS: {:.4f}, VAL LOSS: {:.4f}".format(epoch, avg_trainloss, avg_valloss))

def train(filename, log_msg=None):
    X_train, X_test, y_train, y_test = prepare_data(filename)

    print("X TRAINING")
    print(X_train)
    print("Y TRAINING")
    print(y_train)

    train_dataset, test_dataset = get_datasets(X_train, X_test, y_train, y_test)
    train_loader = get_dataloader(train_dataset, BATCH_SIZE)
    test_loader = get_dataloader(test_dataset, BATCH_SIZE)

    if (log_msg is not None):
        logdir = "runs/" + log_msg
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = SummaryWriter()

    run_nn(writer, train_loader, test_loader)
    writer.flush()
    writer.close()


def debug(continuous=True, log_msg=None):
    """
    continuous: boolean
        (true if continuous fake data, false if discrete fake data)
    """
    if (continuous):
        filename = "fakecont.csv"
    else:
        filename = "fakedis.csv"

    train(filename, log_msg)

if __name__ == "__main__":
    train("vis00.csv", "moreneurons2")
    #debug(True) # continuous
    #debug(False, "FAKEBCE2") # discrete
    pass


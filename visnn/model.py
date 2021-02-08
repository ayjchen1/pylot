import torch
import torch.nn.functional as F

import matplotlib.pylot as plt
import pandas as pd
import numpy as np

torch.manual_seed(1)    # reproducible
BATCH_SIZE = 64
NUMEPOCH = 50

def read_data(filename):
    """
    Reads the visibility data (in csv format) from the filename
    inside vis_data folder

    Returns a pandas dataframe ?"

    """
    return []


def process_data(vis_data):
    """
    Processes the data by stripping away extra info and leaving:
        X: the (x, y, z) relative location of the obstacle to the ego vehicle
        y: the visibility error/uncertainty
    """
    return 0


def print_stats(epoch, step, loss_sum):
    loss_str = "EPOCH: {:2d}, STEP: {:5d}, LOSS: {:.4f}".format(epoch, step, loss_sum)
    print(loss_str)

def run_nn():
    x, y = get_dataset() # implement this!

    """
    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2,)
    """

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)

    net = torch.nn.Sequential(
            torch.nn.Linear(3, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 1),
        )

    criterion = torch.nn.MSELoss() # (y - yhat)^2
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train the neural net
    for epoch in range(NUMEPOCH):
        loss_sum = 0

        for step, data in enumerate(loader):
            # find X, y
            X_cur, y_cur = data

            optimizer.zero_grad()   # zero the optim gradients

            prediction = net(X_cur)
            loss = criterion(prediction, y_cur)
            loss.backward()         # backprop
            optimizer.step()        # apply gradients

            loss_sum += loss.item()
            if (step % 100 == 0):
                print_stats(epoch, step, loss_sum)
                loss_sum = 0


if __name__=="__main__":
    run_nn()




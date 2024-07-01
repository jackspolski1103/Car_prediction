import importlib
import torch
import torch.nn as nn 
import numpy as np
import logging
from torch import nn, optim
from src.train_and_test.test import test


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class Model:
    def __init__(self, name, device, input_dim, lr, epocas, batch_size):
        self.name = name
        self.device = device
        self.input_dim = input_dim
        self.lr = lr
        self.epocas = epocas
        self.batch_size = batch_size
        self.model = SimpleNN(input_dim).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def normalize(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1
        return (data - self.min) / (self.max - self.min)

    def train(self, Metadata_train):
        x_train, y_train = self.split_data(Metadata_train)
        x_train = self.normalize(x_train)
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        for epoch in range(self.epocas):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(x_train)
            loss = self.loss(y_pred, y_train.view(-1, 1))
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                logging.info(f'Epoch {epoch} Loss {loss.item()}')

    def predict(self, x):
        x = self.normalize(x)
        x = x.astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y = self.model(x)
        return y.cpu().numpy().flatten()

    def test(self, metadata_test):
        x_test, y_test = self.split_data(metadata_test)
        y_pred = self.predict(x_test)
        return test(y_test, y_pred)
        
    def save_model(self, path_to_save):
        torch.save(self.model, path_to_save)

    def load_model(self, path_to_load):
        self.model = torch.load(path_to_load)

    

    
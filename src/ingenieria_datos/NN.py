import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    

def normalize(data):
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    max[max == min] = min[max == min] + 1
    return (data - min) / (max - min)


def train_model(model, device, x_train,y_train, x_val, y_val, optimizer, criterion, epochs,batch_size, scheduler=None, early_stopping=5):
    #con batch_size
    model.to(device)
    model.train()
    train_losses = []
    val_losses = []
    best_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_hat = model(x_train)
            train_loss = criterion(y_hat, y_train)
            train_losses.append(train_loss.item())
            y_hat = model(x_val)
            val_loss = criterion(y_hat, y_val)
            val_losses.append(val_loss.item())
            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter == early_stopping:
                break
        if scheduler is not None:
            scheduler.step()
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')
    return model, train_losses, val_losses






    
        



def get_hidden_layers(model, X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = normalize(X)
    X= X.astype(np.float32)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    hidden_layer = model.relu1(model.fc1(X))
    return hidden_layer.detach().cpu().numpy()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def split_data(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def train_valid_split(data, valid_percentage=0.2):
    n = len(data)
    n_valid = int(n * valid_percentage)
    n_train = n - n_valid
    np.random.shuffle(data)
    return data[:n_train], data[n_train:]
    


def feature_engineering(data, train=True):
    if train:
        data = np.load('./results/OneHot/metadata_train.npy', allow_pickle=True)
        data_train, data_val = train_valid_split(data, 0.2)
        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]

        X_train = normalize(X_train)
        X_train= X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        

        X_val = data_val[:, :-1]
        y_val = data_val[:, -1]

        X_val = normalize(X_val)
        X_val= X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)



        model = SimpleNN(data.shape[1]-1)
        train_model(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), torch.tensor(X_train), torch.tensor(y_train).view(-1, 1), torch.tensor(X_val), torch.tensor(y_val).view(-1, 1), optim.Adam(model.parameters(), lr=0.01), nn.MSELoss(), 1000, 32)
        hidden_layer = get_hidden_layers(model, data[:, :-1])
        save_model(model, './src/ingenieria_datos/NN.pth')
        y= data[:, -1]
        return np.column_stack((hidden_layer, y))
    else:
        print('entro')
        print(data.shape[1]-1)
        data = np.load('./results/OneHot/metadata_test.npy', allow_pickle=True)
        model = SimpleNN(data.shape[1]-1)
        model = load_model(model, './src/ingenieria_datos/NN.pth')
        hidden_layer = get_hidden_layers(model, data[:, :-1])
        y= data[:, -1]
        return np.column_stack((hidden_layer, y))
    




import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)  # Dropout with a probability of 0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the activation function
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def train(csv_files, labels):
    X_all = []
    y_all = []
    # load file paths and labels
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        X = df.values 
        y = np.full((X.shape[0], 1), label)
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float().view(-1, 1)
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float().view(-1, 1)
    
    model = Net(input_size=X_all.shape[1])
    
    # using BCELoss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # L2 regularization (weight decay)
    
    # training
    for epoch in range(100):
        optimizer.zero_grad()
        model.train()  # Set the model to training mode
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model, X_test, y_test

# Evaluate the model
def test(model, X_test, y_test):
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Accuracy: {accuracy}")

# Load labeling CSV
labeling_csv_path = "binary_labeling.csv"
labeling_df = pd.read_csv(labeling_csv_path)
csv_files = labeling_df['csv_file'].tolist()
labels = labeling_df['label'].tolist()

# Train the model on all CSVs combined
trained_model, X_test_all, y_test_all = train(csv_files, labels)

# Iterate over each CSV file and test individually
for csv_file, label in zip(csv_files, labels):
    print(f"Testing CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    X_test = torch.tensor(df.values).float()
    y_test = torch.tensor(np.full((X_test.shape[0], 1), label)).float().view(-1, 1)
    test(trained_model, X_test, y_test)
    print()
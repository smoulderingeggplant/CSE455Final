import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

input_size = 10  # Number of input features
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # Number of output units (e.g., for binary classification)

# Step 1: Read labeling CSV file
label_data = pd.read_csv("labeling.csv")

# Step 2: Define Neural Network Architecture
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Step 3: Define Loss Function and Optimization Algorithm
model = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Loop through each distinct CSV file
for index, row in label_data.iterrows():
    # Read CSV file containing vector flows
    csv_file = row['csv_file']
    label = row['label']
    
    # Adjust the path to match the directory structure
    csv_file_path = csv_file.strip("/")
    
    # Read CSV file
    flow_data = pd.read_csv(csv_file_path)
    
    # Preprocess the data as needed
    
    # Perform random split of data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(flow_data, label, test_size=0.2, random_state=42)
    
    # Create Dataset and DataLoader
    train_dataset = FlowDataset(X_train, y_train)
    test_dataset = FlowDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train the Neural Network
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the Model
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / total_samples
    print("Accuracy for", csv_file, ":", accuracy)
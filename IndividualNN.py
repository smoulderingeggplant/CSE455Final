import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define your neural network architecture
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load CSV files and train the neural network
def train_and_test(csv_file, label):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Prepare data
    X = df.drop(columns=['Frame'])  # Drop the 'Frame' column
    
    # Construct the label array
    y = np.full((X.shape[0], 1), label)  # Assign the same label value to all rows
    
    # Convert data to PyTorch tensors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train.values).float()
    y_train = torch.tensor(y_train).float().view(-1, 1)
    X_test = torch.tensor(X_test.values).float()
    y_test = torch.tensor(y_test).float().view(-1, 1)
    
    # Initialize the model
    model = Net(input_size=X.shape[1])
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Test the model
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Accuracy: {accuracy}")

"""
# hard code csv files
csv_data = [
    ("output_video1.mp4/flow_vectors.csv", 0),
    ("output_video2.mp4/flow_vectors.csv", 1),
    ("output_video3.mp4/flow_vectors.csv", 1),
    ("output_video4.mp4/flow_vectors.csv", 1),
    ("output_video5.mp4/flow_vectors.csv", 0),
    ("output_video6.mp4/flow_vectors.csv", 0),
    ("output_video7.mp4/flow_vectors.csv", 0),
    ("output_video8.mp4/flow_vectors.csv", 0),
    ("output_video9.mp4/flow_vectors.csv", 1),
    ("output_video10.mp4/flow_vectors.csv", 0),
    ("output_video11.mp4/flow_vectors.csv", 0),
    ("output_video12.mp4/flow_vectors.csv", 0),
    ("output_video13.mp4/flow_vectors.csv", 0),
    ("output_video14.mp4/flow_vectors.csv", 0),
]
"""

labeling_csv_path = "labeling.csv"
labeling_df = pd.read_csv(labeling_csv_path)
csv_data = labeling_df.values.tolist()

for csv_file, label in csv_data:
    train_and_test(csv_file, label)
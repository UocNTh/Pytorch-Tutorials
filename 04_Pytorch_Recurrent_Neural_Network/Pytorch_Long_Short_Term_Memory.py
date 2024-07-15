import torch

import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F 

from torch.utils.data import DataLoader 

import torchvision.datasets as datasets 
import torchvision.transforms as transforms 

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Hyberparameters
input_size = 28 
sequence_length = 28
num_layers = 2 
hidden_size = 256 
num_classes = 10
learning_rate = 0.005
batch_size = 64 
epochs = 5

# Create Model
class Long_Short_Term_Memory(nn.Module) : 
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Long_Short_Term_Memory, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

 
import os 

file_path = __file__

directory = os.path.dirname(file_path)

dataset = os.path.join(directory, 'dataset')

# Load Data
train_data = datasets.MNIST(dataset, train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(dataset, train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Initialize Network
model = Long_Short_Term_Memory(input_size, hidden_size, num_layers, num_classes).to(device=device) 

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
print('Training Model')
for epoch in range(epochs) : 
    # print(f"Epoch [{epoch + 1}/{epochs}]")
    for batch_idx, (data, targets) in enumerate(train_loader) : 
        data = data.to(device = device).squeeze(1) 
        targets = targets.to(device = device)

        # Forward
        scores = model(data) 
        loss = criterion(scores,targets) 

        #Backward 
        optimizer.zero_grad() 
        loss.backward()

        # Gradient Descent or Adam Step
        optimizer.step() 

def check_accurancy(loader, model) : 

    if loader.dataset.train: 
        print('Checking accurancy on training data')

    else : 
        print('Checking accurancy on test data') 
        
    num_correct = 0 
    num_samples = 0 
    model.eval() 

    with torch.no_grad() : 
        for x, y in loader : 
            x = x.to(device= device).squeeze(1)
            y = y.to(device= device)

            scores = model(x) 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')

    model.train() 

check_accurancy(train_loader, model)

check_accurancy(test_loader, model)

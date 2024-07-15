import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

import os

from tqdm import tqdm  # progress bar


file_path = __file__

directory = os.path.dirname(file_path)

class NeuralNetwork(nn.Module) : 
    def __init__(self, input_size, num_classes) : 
        super(NeuralNetwork, self).__init__() 
        self.fc1 = nn.Linear(input_size, 50) 
        self.fc2 = nn.Linear(50, num_classes) 

    def forward(self, x) : 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

input_size = 784 
num_classes = 10 
learning_rate = 0.001 
batch_size = 64
num_epochs = 1 


dataset = os.path.join(directory, 'dataset')


# Load Data
train_data = datasets.MNIST(dataset, train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(dataset, train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Initialize Network
model = NeuralNetwork(input_size=input_size, num_classes=num_classes).to(device=device) 

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
for epoch in range(num_epochs) : 
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)): 
        data = data.to(device = device) 
        targets = targets.to(device = device)

        data = data.reshape(data.shape[0], -1)

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
            x = x.to(device= device)
            y = y.to(device= device)

            x = x.reshape(x.shape[0], - 1 ) 

            scores = model(x) 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')

    model.train() 



check_accurancy(train_loader, model)


check_accurancy(test_loader, model)

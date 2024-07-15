import torch
import torchvision

import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F 

from torch.utils.data import DataLoader 

import torchvision.datasets as datasets 
import torchvision.transforms as transforms 

from tqdm import tqdm  # progress bar

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Hyberparameterss 
input_size = 28 
sequence_length = 28 
num_layers = 2

'''
num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
would mean stacking two LSTMs together to form a `stacked LSTM`,
with the second LSTM taking in outputs of the first LSTM and
computing the final results. Default: 1
'''

hidden_size = 256 # Number of features in the hidden state
num_classes = 10 
learning_rate = 0.001
batch_size = 64 
num_epochs = 2 

# Create a Recurrent Neural Network
class Recurrent_Neural_Network (nn.Module) : 
    def __init__(self, input_size, hidden_size, num_layers, num_classes) : 
        super(Recurrent_Neural_Network, self).__init__() 
        self.hidden_size = hidden_size
        self.num_layers = num_layers 

        '''
        The batch_first parameter specifies the format of the input and output tensor.

        If batch_first = True: 
            The tensors are expected in the shape (batch, seq, feature)
      
        If batch_first = False: 
            The tensors are expected in the shape (seq, batch, feature)
        '''
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes ) 
    
    def forward(self, x): 

        '''Initializing the hidden state. 

        It creates a tensor of zero with the shape (num_layers, batch_size, hidden_size)
        '''
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        # Forward Prop
        out, _ = self.rnn(x, h0) 

        # Flattening 
        out = out.reshape(out.shape[0], -1)
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
model = Recurrent_Neural_Network(input_size, hidden_size, num_layers, num_classes).to(device=device) 

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
print('Training Model')
for epoch in range(num_epochs) : 
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)): 
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

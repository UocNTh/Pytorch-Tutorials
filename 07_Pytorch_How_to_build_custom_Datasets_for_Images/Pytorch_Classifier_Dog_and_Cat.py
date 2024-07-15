import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split 

from Pytorch_Build_Custom_Datasets_for_Images import CatsAndDogsDataset


current_directory = os.path.dirname(os.path.abspath(__file__)) 

def save_checkpoint(state, filename = os.path.join(current_directory,'checkpoint.pth.tar') ): 
    print('---> Saving Checkpoint')
    torch.save(state, filename) 


def load_checkpoint(checkpoint): 
    print('--> Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer']) 


# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 32
in_channels = 3
num_classes = 2 
learning_rate = 0.0005
batch_size = 1024
epochs = 31
load_model = False

# Datasets
dataset = CatsAndDogsDataset(csv_file='07_Pytorch_How_to_build_custom_Datasets_for_Images/dataset/data.csv', 
                             root_dir='07_Pytorch_How_to_build_custom_Datasets_for_Images/dataset/catsanddogs',
                             transform=transforms.ToTensor())

dataset_length = len(dataset) 
train_length = int(dataset_length*0.8)
test_length = int(dataset_length*0.2) + 1 

train_set, test_set = random_split(dataset, [train_length, test_length]) 
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Create CNN 
class CNN(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super(CNN, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.fc1 = nn.Linear(in_features=32*16*16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x) : 
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x 

file_path = __file__
directory = os.path.dirname(file_path)
dataset = os.path.join(directory, 'dataset')

model = CNN(in_channels, num_classes).to(device=device) 

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 

if load_model : 
    load_checkpoint(torch.load(os.path.join(current_directory,'checkpoint.pth.tar'))) 

for epoch in range(epochs): 
    print(f'Epoch: {epoch}') 
    losses = [] 
    if epoch % 3 == 0  : 
        checkpoint = {
            'state_dict': model.state_dict(), 
            'optimizer' : optimizer.state_dict() 
        }
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)): 
        data = data.to(device=device) 
        targets = targets.to(device=device)

        # Forward
        scores = model(data) 
        loss = criterion(scores,targets)
        losses.append(loss.item()) 

        # Backward 
        optimizer.zero_grad() 
        loss.backward() 

        optimizer.step() 
    
    mean_loss = sum(losses)/len(losses) 

    print(f'Loss at epoch {epoch} was {mean_loss:.5f}') 
def check_accurancy(loader, model) : 
    
    num_correct = 0 
    num_samples = 0 
    model.eval() 

    with torch.no_grad() : 
        for x, y in loader : 
            x = x.to(device= device)
            y = y.to(device= device)

            scores = model(x) 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')

    model.train() 

check_accurancy(train_loader, model)
check_accurancy(test_loader, model)


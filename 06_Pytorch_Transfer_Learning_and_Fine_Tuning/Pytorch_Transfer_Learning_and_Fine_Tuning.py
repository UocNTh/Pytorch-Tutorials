import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import os
import torchvision

from tqdm import tqdm



file_path = __file__
directory = os.path.dirname(file_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

in_channels = 3
num_classes = 10 
learning_rate = 0.0001
batch_size = 1024
num_epochs = 5

dataset = os.path.join(directory, 'dataset')

# Load Data
train_data = datasets.CIFAR10(dataset, train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.CIFAR10(dataset, train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

class Identity(nn.Module) : 
    def __init__(self): 
        super(Identity, self).__init__()

    def forward(self, x): 
        return x

model = torchvision.models.vgg16(pretrained = True) 

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(100,10) 
                                ) 
model.to(device=device)                    
print(model) 

epochs = 20

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 


for epoch in range(epochs): 
    print(f'Epoch: {epoch}') 
    losses = [] 

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

            scores = model(x) 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')

    model.train() 

check_accurancy(train_loader, model)
check_accurancy(test_loader, model)

import requests
import os

def download_image(url, folder_path, file_name):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    response = requests.get(url)
    if response.status_code == 200:
        # Lưu ảnh vào thư mục chỉ định
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Ảnh đã được tải về và lưu tại: {file_path}")
    else:
        print(f"Không thể tải ảnh từ URL. Trạng thái: {response.status_code}")


url = 'https://cdn.britannica.com/96/1296-050-4A65097D/gelding-bay-coat.jpg'
folder_path = os.path.join(directory, 'test')
file_name = 'image.jpg'
download_image(url, folder_path, file_name)

from PIL import Image 

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image_path, model):
    # Transform the image to tensor and normalize it
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move image to the same device as the model
    image = image.to(device=device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

image_path = '/home/toe/Documents/Aladdin_Persson_Pytorch_Serie/06_Pytorch_Transfer_Learning_and_Fine_Tuning/test/image1.jpg'
predicted_label = predict_image(image_path, model)

print(f'The predicted label for the image is: {predicted_label}')
print(f'The predicted label for the image is: {CIFAR10_CLASSES[int(predicted_label)]}') 


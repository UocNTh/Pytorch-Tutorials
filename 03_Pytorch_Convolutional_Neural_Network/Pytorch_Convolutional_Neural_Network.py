import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import os

from tqdm import tqdm

file_path = __file__

directory = os.path.dirname(file_path)

# Create Simple CNN 
class CNN(nn.Module) : 
    def __init__ (self, in_channels = 1, num_classes = 10) : 
        super(CNN, self).__init__() 
        # Create First Convolutional Layer 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 

        # Create Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # Create Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 

        # Create Fully Connected Layer

        '''
        16*7*7: The number of the input features to the fully connected layer

        Firstly, the inital image sizr is 28*28 

        After the first convolutional layer and the first max pooling layer,
        the image size becomes 14*14

        After the second convolutional layer and the second max pooling layer,
        the image size becomes 7*7

        So, input_size to the fully connected layer is 16*7*7 
        '''
        self.fc1 = nn.Linear(16*7*7, num_classes) 

    def forward(self, x) : 
        x = F.relu(self.conv1(x))  
        x = self.pool(x) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = x.reshape(x.shape[0], -1) 
        x = self.fc1(x)

        return x 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

in_channels = 1 
num_classes = 10 
learning_rate = 0.001 
batch_size = 64
num_epochs = 5

dataset = os.path.join(directory, 'dataset')

# Load Data
train_data = datasets.MNIST(dataset, train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(dataset, train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Initialize Network
model = CNN().to(device=device) 

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
for epoch in range(num_epochs) : 
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)): 
        data = data.to(device = device) 
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

from PIL import Image 

def predict_image(image_path, model):
    # Transform the image to tensor and normalize it
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                    transforms.Resize((28, 28)),
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

image_path = 'image/Screenshot from 2024-06-06 13-35-09.png'
predicted_label = predict_image(image_path, model)
print(f'The predicted label for the image is: {predicted_label}')
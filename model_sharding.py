import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os

from sklearn.metrics import accuracy_score

class Net(nn.Module):
    def __init__(self, size, input_depth):
        super().__init__()

        self.size = size
        self.input_depth = input_depth
        self.conv_depth = 32
        self.hidden_dim = 1024

        self.conv_layers = nn.Sequential(
                nn.Conv2d(self.input_depth,
                              self.conv_depth,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(self.conv_depth,
                              self.conv_depth * 2,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(self.conv_depth * 2,
                              self.conv_depth * 4,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                nn.ReLU(),
                nn.Dropout(0.5)
            )


        self.cur_size = self.size // 4
        self.cur_depth = self.conv_depth * 4
        self.linear_layers = nn.Sequential(
                nn.Linear(self.cur_size * self.cur_size * self.cur_depth, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 10)
        )

        self.conv_layers.to('cuda:0')
        self.linear_layers.to('cuda:1')

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(x)

        x = x.to('cuda:1')
        x = x.view(-1, self.cur_size * self.cur_size * self.cur_depth)

        x = self.linear_layers(x)

        return torch.sigmoid(x)

datadir = './data'
if os.path.exists(datadir) == False:
    os.mkdir(datadir)

size = 28
input_depth = 1
batch_size = 1024

training_dataset = datasets.MNIST(
            root=datadir,
            train=True,
            download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
            ]))
trainloader = data.DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True)

test_dataset = datasets.MNIST(
            root=datadir,
            train=True,
            download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
            ]))
testloader = data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True)
net = Net(size, input_depth)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 10
for i in range(num_epochs):
    net.train()
    for j, (inputs, labels) in enumerate(trainloader):
        
        inputs = inputs.to('cuda:0')
        labels = labels.to('cuda:1')

        optimizer.zero_grad()

        preds = net(inputs)
        loss = loss_fn(preds, labels)
        loss.backward()

        optimizer.step()

        print("Epoch {}, batch {}: loss = {}\n".format(i + 1, j + 1, loss.data.item()))

net.eval()
ground_truth = []
predictions = []

for inputs, labels in testloader:

    inputs = inputs.to('cuda:0')
    labels = labels.to('cuda:1')
    
    _, preds = F.softmax(net(inputs), dim=1).max(1)

    ground_truth.extend(labels.cpu().numpy())
    predictions.extend(preds.cpu().numpy())

print("Accuracy: {0:.2f}".format(100 * accuracy_score(ground_truth, predictions)))

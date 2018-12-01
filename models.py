import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.dro = nn.Dropout(p=0.2)
        self.p1 = nn.MaxPool2d(2,2)
        self.p2 = nn.MaxPool2d(2,2)
        self.p3 = nn.MaxPool2d(2,2)
        self.p4 = nn.MaxPool2d(2,2)
        self.p5 = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.conv2 = nn.Conv2d(32,  64, 3)
        self.conv3 = nn.Conv2d(64,  128, 3)
        self.conv4 = nn.Conv2d(128,  256, 3)
        self.conv5 = nn.Conv2d(256,  512, 1)
        self.fc1= nn.Linear(6*6*512,1024)
        self.fc2= nn.Linear(1024,136)
        
    def forward(self, x):
        x = self.dro(self.p1(F.selu(self.conv1(x))))
        x = self.dro(self.p2(F.selu(self.conv2(x))))
        x = self.dro(self.p3(F.selu(self.conv3(x))))
        x = self.dro(self.p4(F.selu(self.conv4(x))))
        x = self.dro(self.p5(F.selu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x= self.dro(F.selu(self.fc1(x)))
        x= self.fc2(x)
        return x

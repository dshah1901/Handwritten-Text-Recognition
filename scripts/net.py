import torch.nn as nn
import torch.nn.functional as F

#    Title: CNN Model implementation
#    Author: Aaron Wong
#    Date: Dec 31, 2018 
#    Availability: https://medium.com/hackernoon/traffic-sign-classification-6e7113d9c4d5

class Net(nn.Module):
    #CNN Model
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #Convolution Layer 1
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #Convolution Layer 2
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x)) #Activation Function
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

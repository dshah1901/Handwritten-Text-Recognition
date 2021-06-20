from torch import nn, optim, cuda
from data_processing import *
from net import Net

#Gets the dataset

batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

#Initialize the network and define the optimizer
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#Training the tarining dataset
def train(epoch):
    model.train()
    train_loader = get_data(1)
    for batch_idx, (data, target) in enumerate( train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), 'model/model.pth') #Saving the trained model

#Testing the dataset and getting the accuracy 
def test():
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = get_data(1)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        

    test_loss /= len(test_loader.dataset) 
    return(f'===========================\n Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')

#Training the dataset for 10 epoches
def training(self):
    for epoch in range(1, 10):
        train(epoch)
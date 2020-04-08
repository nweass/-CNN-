import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


                                                                                        #录入训练数据和测试数据
def loadtraindata():
    path = r"./train_data"
    trainset = torchvision.datasets.ImageFolder(path,
    transform=transforms.Compose([ transforms.Resize((227, 227)),

                           transforms.CenterCrop(227),
                           transforms.ToTensor(),
                            transforms.Normalize((0.4840,),(0.2582,))]) )
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                           shuffle=True, num_workers=2)
    return train_dataloader

def loadtestdata():
    path = r"./test_data"
    trainset = torchvision.datasets.ImageFolder(path,
    transform=transforms.Compose([ transforms.Resize((227, 227)),

                           transforms.CenterCrop(227),
                           transforms.ToTensor(),
                            transforms.Normalize((0.4840,),(0.2582,))]) )
    tset_dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                           shuffle=True, num_workers=2)
    return  tset_dataloader
classes = ( 'dog',  'horse')


                                                                                        #定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(13 * 13 * 256, 120)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 13 * 13 * 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

net = Net()
                                                                            #训练模块
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = F.nll_loss(pred, target)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, loss.item()))

                                                                    #测试模块
def test(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                                                        #优化方法和参数
lr = 0.001
momentum = 0.9
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=0.0001)




if __name__ == '__main__':
    for epoch in range(4):                                                          #训练、保存模型
        train(model, device, loadtraindata(), optimizer, epoch)
        test(model, device, loadtestdata())
    print('Finished Training')
    torch.save(model.state_dict(), "ANiMAL_CNN.pt")



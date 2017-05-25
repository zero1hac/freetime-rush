import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img/2 + 0.5

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print ' '.join('%5s' % classes[labels[j]] for j in range(4))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        #print size
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print "Start training"
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i%2000 == 1999:
            print '[%d, %5d] loss: %.3f' %  (epoch + 1, i + 1, running_loss / 2000)
            running_loss = 0.0
print "finished training"


dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print 'GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4))
output = net(Variable(images))

_, predict = torch.max(output.data, 1)
print 'Predicted: ', ' '.join('%5s' % classes[predict[j][0]] for j in range(4))

correct = 0
total = 0
for data in testloader:
    images, labels = data
    output = net(Variable(images))
    _, predict = torch.max(output.data, 1)
    total +=labels.size(0)
    correct += (predict==labels).sum()

print 'Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total)

class_correct = [0.]*10
class_total = [0.]*10

for data in testloader:
    images, labels = data
    output = net(Variable(images))
    _, predict = torch.max(output.data, 1)
    c = (predict==labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] +=c[i]
        class_total[label] += 1

for i in range(10):
    print 'Accuracy of %5s : %2d %%' % (classes[i], 100*class_correct[i]/class_total[i])

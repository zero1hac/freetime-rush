import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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
        print size
        num_features = 1
        for i in size:
            num_features *= i
        return num_features


net = Net()
print net
params = list(net.parameters())
print len(params)
print params[0].size()

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print out

#out.backward(torch.randn(1, 10))
target = Variable(torch.arange(1, 11))

err = nn.MSELoss()

loss = err(out, target)
print loss

print loss.creator
print(loss.creator.previous_functions[0][0])
print(loss.creator.previous_functions[0][0].previous_functions[0][0])

net.zero_grad()
print "Biases before backward pass"
print net.conv1.bias.grad

loss.backward()

print "Biases after backward pass"

print net.conv1.bias.grad


optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

optimizer.step()

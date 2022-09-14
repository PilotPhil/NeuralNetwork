import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 1. conv->relu->maxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        # 2. conv->relu->maxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        #
        x = x.view(-1, self.num_flat_features(x))

        # 3.full connect1
        x = self.fc1(x)
        x = F.relu(x)

        # 3.full connect2
        x = self.fc2(x)
        x = F.relu(x)

        # 3.full connect3
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    # 1.实例化网络
    net = Net()
    # 查看网络结构
    print(net)

    # 2.网络的参数
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    # 3.forward测试
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    print(output)

    # 4.计算loss
    target = torch.randn(10)  # 随机值作为样例
    target = target.view(1, -1)  # 使target和output的shape相同

    criterion = nn.MSELoss()# 均方差误差函数
    loss = criterion(output, target)
    print('loss: '+str(loss))

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    #
    net.zero_grad()  # 清除梯度

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    #
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
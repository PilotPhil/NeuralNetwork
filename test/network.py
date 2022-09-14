import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入是1个通道的灰度图，输出6个通道(feature map)，使用5x5的卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 第二个卷积层也是5x5，有16个通道
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 32x32 -> 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 14x14 -> 10x10 -> 5x5
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了batch维度之外的其它维度。
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

output = net(input)
target = torch.arange(1, 11)  # 随便伪造的一个“真实值”
target = target.view(1, -1)  # 把它变成output的shape(1, 10)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchNumberNet(nn.Module):
    def __init__(self):
        super(TorchNumberNet, self).__init__()

        # 1.conv
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.relu0 = nn.ReLU()

        # 2.max polling
        self.maxPool = nn.MaxPool2d((2, 2))

        # 3.full connect
        self.fc1 = nn.Linear(8 * 13 * 13, 676)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(676, 338)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(338, 10)

        # 4.softmax
        self.softmax = nn.Softmax(dim=1)

        # 5.criterion
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # 1.conv 1*28*28 -> 8*26*26
        x = self.conv1(x)
        x = self.relu0(x)

        # 2.max polling 8*26*26 -> 8*13*13
        x = self.maxPool(x)

        # 3.full connect 8*13*13 -> 10
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        # 4.softmax 10 -> 10(all>0)
        x = self.softmax(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    # Quadrant information
    print('Quadrant information: ')
    net = TorchNumberNet()
    print(net)
    print('')

    # test forward
    print('test forward: ')
    input = torch.randn(1, 1, 28, 28)
    output = net.forward(input)
    print(output)
    print('')

    # params
    print('params: ')
    params = list(net.parameters())
    print(len(params))

    # test loss
    target = torch.randn(1, 10)  # aka real value
    loss = net.criterion(output, target)
    print('loss: ' + str(loss))

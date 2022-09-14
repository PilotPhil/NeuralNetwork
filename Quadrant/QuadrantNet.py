import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrantNet(nn.Module):
    def __init__(self):
        super(QuadrantNet, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # test net
    net = QuadrantNet()
    print('net structure: ')
    print(net)
    print('')

    input=torch.randn(1,2)
    output=net(input)
    print(output)


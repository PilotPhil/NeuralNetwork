import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrantNet(nn.Module):
    def __init__(self):
        super(QuadrantNet, self).__init__()

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
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


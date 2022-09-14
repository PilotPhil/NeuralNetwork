import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberNet(nn.Module):
    def __init__(self, lr):
        super(NumberNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3),  # 1*1*28*28 -> 1*8*26*26
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 1*8*26*26 -> 1*8*13*13
            nn.Flatten(),
            nn.Linear(8 * 13 * 13, 10),
            nn.Softmax(dim=1)
        )

        self.lr = lr

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.parameters(), self.lr)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = NumberNet()
    print(net)

    input = torch.randn(1, 1, 28, 28)
    output = net(input)
    print(output)

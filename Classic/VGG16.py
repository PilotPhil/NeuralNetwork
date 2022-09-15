import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, outputNum: int):
        super(VGG16, self).__init__()
        self.outputNum=outputNum

        self.criterion = nn.MSELoss()

        self.modle = nn.Sequential(
            # Block1
            # 1x1x224x224 -> 1x64x224x224
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x64x224x224 -> 1x64x224x224
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x64x224x224 -> 1x64x112x112
            nn.MaxPool2d(stride=2, kernel_size=2),

            # 1x64x112x112 -> 1x128x112x112
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x128x112x112 -> 1x128x112x112
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x128x112x112 -> 1x128x56x56
            nn.MaxPool2d(stride=2, kernel_size=2),

            # 1x128x56x56 -> 1x256x56x56
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x256x56x56 -> 1x256x56x56
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x256x56x56 -> 1x256x28x28
            nn.MaxPool2d(stride=2, kernel_size=2),

            # 1x256x28x28 -> 1x512x28x28
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x512x28x28 -> 1x512x28x28
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x512x28x28 -> 1x512x14x14
            nn.MaxPool2d(stride=2, kernel_size=2),

            # 1x512x14x14 -> 1x512x14x14
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x512x14x14 -> 1x512x14x14
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 1x512x14x14 -> 1x512x7x7
            nn.MaxPool2d(stride=2, kernel_size=2),

            # 1x512x7x7
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, self.outputNum),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.modle(x)
        return x


if __name__ == '__main__':
    net = VGG()
    print(net)

    input = torch.randn(1, 1, 224, 224)
    output = net(input)

    print(output)

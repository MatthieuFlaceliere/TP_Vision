from torch import flatten
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, in_channel, output_dim):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, 3)
        self.relu1 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, 2)
        self.relu2 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.BN3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2)
        self.relu4 = nn.ReLU()
        self.BN4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.relu5 = nn.ReLU()
        self.BN5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, 2)
        self.relu6 = nn.ReLU()
        self.BN6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.relu7 = nn.ReLU()
        self.BN7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, 2)
        self.relu8 = nn.ReLU()
        self.BN8 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(73728, 120)
        self.relufc1 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relufc2 = nn.ReLU()

        self.fc3 = nn.Linear(84, output_dim)
        self.act_output = nn.Softmax(dim=1)


    def forward(self, x):

        z1 = self.conv1(x)
        a1 = self.relu1(z1)
        a1 = self.BN1(a1)

        z2 = self.conv2(a1)
        a2 = self.relu2(z2)
        a2 = self.BN2(a2)

        z3 = self.conv3(a2)
        a3 = self.relu3(z3)
        a3 = self.BN3(a3)

        z4 = self.conv4(a3)
        a4 = self.relu4(z4)
        a4 = self.BN4(a4)

        z5 = self.conv5(a4)
        a5 = self.relu5(z5)
        a5 = self.BN5(a5)

        z6 = self.conv6(a5)
        a6 = self.relu6(z6)
        a6 = self.BN6(a6)

        z7 = self.conv7(a6)
        a7 = self.relu7(z7)
        a7 = self.BN7(a7)

        z8 = self.conv8(a7)
        a8 = self.relu8(z8)
        a8 = self.BN8(a8)

        a8 = flatten(a8, 1)

        z9 = self.fc1(a8)
        a9 = self.relufc1(z9)

        z10 = self.fc2(a9)
        a10 = self.relufc2(z10)

        z11 = self.fc3(a10)
        y = self.act_output(z11)

        return y
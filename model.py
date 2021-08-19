import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 2)
        self.classifier = nn.Linear(2, 10)

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        feature = self.fc1(x)
        output = self.classifier(feature)
        return feature, output


if __name__ == "__main__":
    from torchsummary import summary
    model = Model().cuda()
    summary(model, input_size=(3, 32, 32))
    import torch
    torch.tensor

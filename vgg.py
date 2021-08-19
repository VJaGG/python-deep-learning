import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        block1_conv1 = F.relu(self.block1_conv1(x))
        block1_conv2 = F.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.block1_pool(block1_conv2)

        block2_conv1 = F.relu(self.block2_conv1(block1_pool))
        block2_conv2 = F.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.block2_pool(block2_conv2)

        block3_conv1 = F.relu(self.block3_conv1(block2_pool))
        block3_conv2 = F.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = F.relu(self.block3_conv3(block3_conv2))
        block3_pool = self.block3_pool(block3_conv3)

        block4_conv1 = F.relu(self.block4_conv1(block3_pool))
        block4_conv2 = F.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = F.relu(self.block4_conv3(block4_conv2))
        block4_pool = self.block4_pool(block4_conv3)

        block5_conv1 = F.relu(self.block5_conv1(block4_pool))
        block5_conv2 = F.relu(self.block5_conv2(block5_conv1))
        block5_conv3 = F.relu(self.block5_conv3(block5_conv2))
        block5_pool = self.block5_pool(block5_conv3)
    
        features = {'block1_conv1': block1_conv1,
                    'block2_conv1': block2_conv1,
                    'block3_conv1': block3_conv1,
                    'block4_conv1': block4_conv1,
                    'block5_conv1': block5_conv1}

        x = self.avgpool(block5_pool)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, block5_conv3, features


if __name__ == "__main__":
    # x = torch.rand(1, 3, 224, 224)
    # model = VGG16()
    # pred = model(x)
    # print(pred.shape)
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.Tensor([2])
    z = x + y
    z.backward()
    print(z)










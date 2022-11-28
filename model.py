import torch
import torch.nn as nn
from layers import ResidualLayer, EdgeDetectionLayer, LaplacianLayer

class BasicNN(nn.Module):
    def __init__(self, initial_size=158 * 320):
        super(BasicNN, self).__init__()
        self.flatten = nn.Flatten()
        self.classifiy = nn.Sequential(
            nn.Linear(initial_size, initial_size // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(initial_size // 4),
            nn.Linear(initial_size // 4, initial_size // 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(initial_size // 16),
            nn.Linear(initial_size // 16, initial_size // 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(initial_size // 64),
            nn.Linear(initial_size // 64, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        assert x.shape[1] == 1, 'Image need to only have 1 channel'
        # assert x.shape[2] == 64, 'Input image shape needs to be 64 * 128'
        # assert x.shape[3] == 128, 'Input image shape needs to be 64 * 128'

        flatten_input = self.flatten(x)
        out = self.classifiy(flatten_input)
        return out

class EdgeDetectionNN(BasicNN):
    def __init__(self):
        super().__init__(initial_size=158 * 320 // 4)
        self.edge_detect = EdgeDetectionLayer(1)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        feature = self.edge_detect(x)
        return super().forward(feature)


class LaplacianNN(BasicNN):
    def __init__(self):
        super().__init__(initial_size=158 * 320 // 4)
        self.edge_detect = LaplacianLayer(1)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        feature = self.edge_detect(x)
        return super().forward(feature)


class BaselineNN(BasicNN):
    def __init__(self):
        super().__init__(initial_size=158 * 320 // 4)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        return super().forward(x)


class BaselineCNN(BasicNN):
    def __init__(self):
        super().__init__(initial_size=158 * 320 // 4)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.feature = nn.Conv2d(1, 1, kernel_size=3, padding='same', bias=False)

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        feature = self.feature(x)
        return super().forward(feature)

class ClassifierPlusPretrain(nn.Module):
    def __init__(self, pretrainlayer, feature_channel=64, domainfeature = False):
        super(ClassifierPlusPretrain, self).__init__()
        self.pretrainlayer = pretrainlayer
        self.domainfeatureextract = nn.Sequential(
            ResidualLayer(feature_channel, 64),
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        self.global_pool = nn.AdaptiveMaxPool2d(7)
        self.flatten = nn.Flatten()
        self.domainfeature = domainfeature
        self.cn1 = nn.Conv2d(feature_channel, 64, 1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(64)
        self.feature_channel = feature_channel

    def forward(self, img):
        pretrainfeature = self.pretrainlayer(img)
        if self.domainfeature:
            feature = self.domainfeatureextract(pretrainfeature)
        else:
            if self.feature_channel != 64:
                feature = self.relu(self.batchnorm(self.cn1(pretrainfeature)))
            else:
                feature = self.batchnorm(pretrainfeature)

        feature = self.global_pool(feature)

        flatten_feature = self.flatten(feature)

        return self.final_classifier(flatten_feature)
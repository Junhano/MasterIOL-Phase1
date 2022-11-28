import torch.nn as nn
import math
import torch

def same_padding(kernel_size):
  return math.floor((kernel_size - 1) / 2)


class EdgeDetectionLayer(nn.Module):
    def __init__(self,input_channel):
        super().__init__()
        self.convx = nn.Conv2d(input_channel,input_channel, kernel_size=3, padding='same',bias=False)
        self.convy = nn.Conv2d(input_channel,input_channel, kernel_size=3, padding='same',bias=False)
        edge_detection_x = [[-1,0,1] for _ in range(3)]
        edge_detection_y = [[1,1,1],[0,0,0],[-1,-1,-1]]
        for conv, kernel in [(self.convx, edge_detection_x),(self.convy, edge_detection_y)]:
            new_weights = torch.Tensor([[kernel for _1 in range(input_channel)] for _2 in range(input_channel)])
            conv.weight = nn.parameter.Parameter(new_weights,requires_grad=False)
    def forward(self,x):
        return self.convx(x) + self.convy(x)


class LaplacianLayer(nn.Module):
    def __init__(self,input_channel):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,input_channel, kernel_size=3, padding='same',bias=False)
        kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        new_weights = torch.Tensor([[kernel for _1 in range(input_channel)] for _2 in range(input_channel)])
        self.conv.weight = nn.parameter.Parameter(new_weights,requires_grad=False)
    def forward(self,x):
        return self.conv(x)#compose x and y edge detection

class ResidualLayer(nn.Module):#Resnet paper for reference: https://arxiv.org/pdf/1512.03385.pdf
    def __init__(self, input_channel, output_channel, kernel_size = 3):
        super(ResidualLayer, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, padding = same_padding(kernel_size))
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size = kernel_size, padding = same_padding(kernel_size))
        self.batchnorm1 = nn.BatchNorm2d(output_channel)
        self.batchnorm2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(input_channel, output_channel, kernel_size = 1)

    def forward(self, x):
        residual = x
        if self.input_channel != self.output_channel:
            residual = self.skip(residual)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batchnorm1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out += residual
        out = self.batchnorm2(out)
        return out
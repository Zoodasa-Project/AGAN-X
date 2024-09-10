import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, num_res_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        
        upsample_blocks = [UpsampleBlock(64, 2) for _ in range(2)]  # 2번의 2배 업스케일링으로 4배 달성
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        x = self.prelu(self.conv1(x))
        residual = x
        x = self.res_blocks(x)
        x = self.bn(self.conv2(x))
        x += residual
        x = self.upsample_blocks(x)
        x = self.conv3(x)
        return torch.tanh(x)

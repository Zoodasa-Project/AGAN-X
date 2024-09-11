import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, channels, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.time_embed = nn.Embedding(num_steps, 128)
        self.unet = UNet(channels, 128)

    def forward(self, x, t):
        time_emb = self.time_embed(t)
        return self.unet(x, time_emb)

class UNet(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.down1 = DownBlock(channels, 64, time_emb_dim)
        self.down2 = DownBlock(64, 128, time_emb_dim)
        self.down3 = DownBlock(128, 256, time_emb_dim)
        self.up1 = UpBlock(256, 128, time_emb_dim)
        self.up2 = UpBlock(128, 64, time_emb_dim)
        self.up3 = UpBlock(64, channels, time_emb_dim)

    def forward(self, x, t):
        d1 = self.down1(x, t)
        d2 = self.down2(d1, t)
        d3 = self.down3(d2, t)
        u1 = self.up1(d3, d2, t)
        u2 = self.up2(u1, d1, t)
        return self.up3(u2, x, t)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        time_emb = F.relu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        return self.pool(h)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        time_emb = F.relu(self.time_mlp(t))
        return h + time_emb.unsqueeze(-1).unsqueeze(-1)

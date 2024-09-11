import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features[:23].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        return F.mse_loss(self.vgg(x), self.vgg(y))

class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_pred, fake_pred):
        real_loss = self.loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.loss(fake_pred, torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) / 2

def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(torch.nn.Module):
    
    def __init__(self, f1=9, f2=1, f3=5, n1=64, n2=32, c=3, input_size=33):
        super().__init__()
        self.feature = nn.Conv2d(c, n1, f1, stride=1, padding=0)
        self.remap = nn.Conv2d(n1, n2, f2, stride=1, padding=0)
        self.reconstruct = nn.Conv2d(n2, c, f3, stride=1, padding=0)
        self.margin = (f1 + f2 + f3 - 3) // 2

    # Input: N, C, H, W
    # Output: N, C, H-12, W-12
    def forward(self, x):
        x = self.feature(x)
        x = F.relu(x, inplace=True)
        x = self.remap(x)
        x = F.relu(x, inplace=True)
        x = self.reconstruct(x)
        return x

        




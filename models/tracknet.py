import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """
    Standard block with 2D Convolution, Batch Normalization, and ReLU.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class TrackNetV3(nn.Module):
    """
    TrackNetV3 Architecture for ball detection in sports videos.
    Input : (B, 9, H, W)  — 3 RGB frames stacked
    Output: (B, 3, H, W)  — Gaussian heatmap per frame
    Peak of heatmap = ball location (subpixel accurate)
    """
    def __init__(self, in_frames=3, out_frames=3):
        super().__init__()
        in_ch = in_frames * 3
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, 64),
            ConvBNReLU(64, 64)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBNReLU(256, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512)
        )
        self.dec4 = nn.Sequential(
            ConvBNReLU(512 + 256, 256),
            ConvBNReLU(256, 256)
        )
        self.dec3 = nn.Sequential(
            ConvBNReLU(256 + 128, 128),
            ConvBNReLU(128, 128)
        )
        self.dec2 = nn.Sequential(
            ConvBNReLU(128 + 64, 64),
            ConvBNReLU(64, 64)
        )
        self.dec1 = nn.Sequential(
            ConvBNReLU(64, 64),
            nn.Conv2d(64, out_frames, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d4 = F.interpolate(e4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        
        d3 = F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        
        d2 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        return torch.sigmoid(self.dec1(d2))

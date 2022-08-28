import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# Unet block
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, input_):
        return self.model(input_)
    
class UNetUp(nn.Module):
    def __init__(self, in_channels,out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias = False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input_, skip_):
        x = self.layers(input_)
        x = torch.cat([x, skip_], axis = 1)
        return x
    
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512)
        
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1,0,1,0)),  # 왼쪽, 위 제로패딩
            nn.Conv2d(128, out_channels, 4, padding=1),  # 위에서 패딩 한칸씩 했으므로 원본 크기를 유지한다.
            nn.Tanh()
        )
        
    def forward(self, input_):  # input is sketch
        d1 = self.down1(input_)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        output = self.up_final(u7)  # [B x 3 x H x W]
        return output
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalization=True):
            block = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalization:
                block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2))
            return block
        
        self.discriminator = nn.Sequential(
            *discriminator_block(in_channels*2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)            
        )
        
    def forward(self, condition, gt):
        x = torch.cat([condition, gt], axis=1)
        return self.discriminator(x)   
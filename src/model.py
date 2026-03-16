# Modelo Unet 2D para Audio Super Resolution

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Módulo de atención para ponderar skip connections"""
    def __init__(self, F_g, F_l, F_int):

        super().__init__()

        # Basado en el paper "https://arxiv.org/abs/1804.03999"
        # https://github.com/ozan-oktay/Attention-Gated-Networks
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # Interpolar g para que tenga el mismo tamaño que x
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Calcular la atención
        att = self.sigmoid(self.psi(self.relu(self.W_g(g) + self.W_x(x))))

        return x * att


class DilatedBlock(nn.Module):
    """Bloque con capas convolucionales dilatadas para capturar contexto de largo alcance sin perder resolución."""
    def __init__(self, channels):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7,3), padding=(3,1), dilation=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(7,3), padding=(6,2), dilation=(2,2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(7,3), padding=(12,4), dilation=(4,4)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x) + x


class UNetAudio2D(nn.Module):
    """Arquitectura UNet 2D adaptada para audio super resolution."""
    def __init__(self):

        super().__init__()

        # Encoder
        # Entrada: (B, 2, F, T)
        self.enc1 = self.conv_block(2, 32)
        self.pool1 = nn.MaxPool2d((2,2))

        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d((2,2))

        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d((2,2))

        self.enc4 = self.conv_block(128, 256)
        self.pool4 = nn.MaxPool2d((2,2))

        # Bottleneck
        self.bottleneck_conv = self.conv_block(256, 512)
        self.bottleneck_dilated = DilatedBlock(512)

        # Decoder
        self.up4 = self.up_block(512,256)
        self.dec4 = self.conv_block(512,256)

        self.up3 = self.up_block(256,128)
        self.dec3 = self.conv_block(256,128)

        self.up2 = self.up_block(128,64)
        self.dec2 = self.conv_block(128,64)

        self.up1 = self.up_block(64,32)
        self.dec1 = self.conv_block(64,32)

        # Attention gates
        self.att4 = AttentionGate(256,256,128)
        self.att3 = AttentionGate(128,128,64)
        self.att2 = AttentionGate(64,64,32)
        self.att1 = AttentionGate(32,32,16)

        # Output
        self.final = nn.Conv2d(32,2,kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        """Bloque convolucional"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(7,3), padding=(3,1)),
            nn.GroupNorm(out_ch//4, out_ch),    # GroupNorm para batchs pequeños
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(7,3), padding=(3,1)),
            nn.GroupNorm(out_ch//4, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def up_block(self, in_ch, out_ch):
        """Bloque de upsampling en frecuencia y tiempo"""
        return nn.Sequential(
            nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True))

    def match_size(self, x, ref):
        """Asegura que x tenga el mismo tamaño que ref"""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        return x

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck_conv(self.pool4(e4))
        b = self.bottleneck_dilated(b)

        # Decoder con skip connections y attention gates
        up4 = self.match_size(self.up4(b), e4)
        d4 = self.dec4(torch.cat([up4, self.att4(up4, e4)], dim=1))

        up3 = self.match_size(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([up3, self.att3(up3, e3)], dim=1))

        up2 = self.match_size(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([up2, self.att2(up2, e2)], dim=1))

        up1 = self.match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([up1, self.att1(up1, e1)], dim=1))

        return self.final(d1) + x
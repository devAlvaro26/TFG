# Modelo UNet 2D para Audio Super-Resolución

import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    """Módulo de atención para ponderar las skip connections en función de la relevancia de las características."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: feature del decoder (upsampled), x: skip connection del encoder
        att = self.sigmoid(self.psi(self.relu(self.W_g(g) + self.W_x(x))))
        return x * att  # Pondera la skip connection
        
class UNetAudio2D(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Encoder (Downsampling) ---
        # Entrada: (B, 2, F, T) — 2 canales: real e imaginario
        self.enc1 = self.conv_block(2, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = self.conv_block(256, 512)

        # --- Decoder (Upsampling) ---
        self.up4 = nn.Sequential(nn.Conv2d(512, 256 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True),)
        self.dec4 = self.conv_block(512, 256)   # 256 (up) + 256 (skip)
        # Pixel shuffle añadido en base a https://github.com/kuleshov/audio-super-res 
        # Solución a los artefactos de checkerboard

        self.up3 = nn.Sequential(nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True),)
        self.dec3 = self.conv_block(256, 128)   # 128 (up) + 128 (skip)

        self.up2 = nn.Sequential(nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True),)
        self.dec2 = self.conv_block(128, 64)    # 64 (up) + 64 (skip)

        self.up1 = nn.Sequential(nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True),)
        self.dec1 = self.conv_block(64, 32)     # 32 (up) + 32 (skip)

        # --- Attention Gates (uno por nivel de skip connection) ---
        self.att4 = AttentionGate(256,256,128)
        self.att3 = AttentionGate(128,128,64)
        self.att2 = AttentionGate(64,64,32)
        self.att1 = AttentionGate(32,32,16)

        # --- Output ---
        self.final = nn.Conv2d(32, 2, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_ch//4, num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_ch//4, num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder con skip connections y attention gates
        up4_out = self.up4(b)
        d4 = self.dec4(torch.cat([up4_out, self.att4(up4_out, e4)], dim=1))

        up3_out = self.up3(d4)
        d3 = self.dec3(torch.cat([up3_out, self.att3(up3_out, e3)], dim=1))

        up2_out = self.up2(d3)
        d2 = self.dec2(torch.cat([up2_out, self.att2(up2_out, e2)], dim=1))

        up1_out = self.up1(d2)
        d1 = self.dec1(torch.cat([up1_out, self.att1(up1_out, e1)], dim=1))

        return self.final(d1) + x
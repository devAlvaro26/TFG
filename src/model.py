import torch
import torch.nn as nn

class UNetAudio(nn.Module):
    def __init__(self):
        super(UNetAudio, self).__init__()

        # --- Encoder (Downsampling) ---
        # Input: 1 channel (Audio waveform)
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = self.conv_block(32, 64)
        self.pool3 = nn.MaxPool1d(2)

        # --- Bottleneck ---
        self.bottleneck = self.conv_block(64, 128)

        # --- Decoder (Upsampling) ---
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64) # 64+64 input channels

        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32) # 32+32 input channels

        self.up1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16) # 16+16 input channels

        # --- Output ---
        self.final = nn.Conv1d(16, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Skip Connections
        d3 = self.up3(b)
        # Concatenate d3 with e3 (Skip Connection)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
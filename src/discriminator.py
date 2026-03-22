# Discriminador Multi-Escala (MSD)
# Basado en HiFi-GAN y AERO: AUDIO SUPER RESOLUTION IN THE SPECTRAL DOMAIN
# https://arxiv.org/pdf/2010.05646
# https://arxiv.org/pdf/2211.12232

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorS(nn.Module):
    """
    Sub-discriminador a una escala específica.
    """
    def __init__(self, use_spectral_norm=False):

        super().__init__()
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = [] # feature maps para Feature Matching Loss
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Discriminador Multi-Escala (MSD).
    Evaluar audio en distintas escalas
    """
    def __init__(self):

        super().__init__()
        
        # Multi discriminadores basado en AERO
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True), # Escala original (1x)
            DiscriminatorS(),                       # Escala reducida (0.5x)
        ])
        
        # Average pooling para reducir la resolución de la forma de onda
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        """
        y: Waveform real (Ground Truth) -> Shape (B, 1, T)
        y_hat: Waveform generada -> Shape (B, 1, T)
        """
        y_d_rs = []     # Resultados discriminador para real
        y_d_gs = []     # Resultados discriminador para generado
        fmap_rs = []    # Feature maps reales
        fmap_gs = []    # Feature maps generados

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)

            # Pasar por el discriminador
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
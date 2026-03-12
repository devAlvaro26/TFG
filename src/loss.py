# Loss para Audio Super-Resolución
# Este script contiene las funciones de pérdida para entrenar la red neuronal, calculando
# la pérdida en el dominio del tiempo y en el dominio de la frecuencia.

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_mag(x, fft_size, hop_size, win_length):
    """Calcula el STFT y devuelve la magnitud."""
    device = x.device
    x = x.cpu()
    window = torch.hann_window(win_length).cpu()

    stft_res = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    
    return torch.abs(stft_res).to(device)


class SpectralConvergenceLoss(nn.Module):
    """Pérdida de convergencia espectral."""

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)


class LogMagnitudeLoss(nn.Module):
    """Pérdida de magnitud logarítmica."""

    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))


class STFTLoss(nn.Module):
    """Pérdida STFT."""
    def __init__(self, fft_size, hop_size, win_length):

        super().__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length

        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft_mag(x, self.fft_size, self.hop_size, self.win_length)
        y_mag = stft_mag(y, self.fft_size, self.hop_size, self.win_length)

        sc = self.sc_loss(x_mag, y_mag)
        mag = self.mag_loss(x_mag, y_mag)

        return sc, mag


class MultiResolutionSTFTLoss(nn.Module):
    """Pérdida STFT multi-resolución."""
    def __init__(self):

        super().__init__()

        # Calcular STFT a diferentes resoluciones para capturar diferentes frecuencias
        self.stft_losses = nn.ModuleList([
            STFTLoss(512, 128, 512),
            STFTLoss(1024, 256, 1024),
            STFTLoss(2048, 512, 2048),
        ])

    def forward(self, x, y):
        sc_total = 0
        mag_total = 0

        for loss in self.stft_losses:
            sc, mag = loss(x, y)
            sc_total += sc
            mag_total += mag

        sc_total /= len(self.stft_losses)
        mag_total /= len(self.stft_losses)

        return sc_total, mag_total


class HighFrequencyLoss(nn.Module):
    """Pérdida de alta frecuencia."""
    def __init__(self, sr=44100, n_fft=1024, fmin=4000):

        super().__init__()

        # Crear máscara para ponderar más las altas frecuencias
        freqs = torch.linspace(0, sr/2, n_fft//2 + 1)
        mask = (freqs >= fmin).float()

        self.register_buffer("mask", mask.view(1, -1, 1))

    def forward(self, pred_mag, target_mag):
        return torch.mean(torch.abs(pred_mag - target_mag) * self.mask)


class CombinedLoss(nn.Module):
    """Pérdida combinada de las anteriores."""
    def __init__(self, lambda_mrstft=1.0, lambda_hf=1.5, lambda_complex=0.5):

        super().__init__()

        self.lambda_mrstft = lambda_mrstft
        self.lambda_hf = lambda_hf
        self.lambda_complex = lambda_complex

        self.mrstft = MultiResolutionSTFTLoss()
        self.hf_loss = HighFrequencyLoss()

    def forward(self, pred, target):
        # pred/target shape (B,2,F,T)
        
        # Descartar bins de frecuencia acolchados para coincidir con n_fft // 2 + 1 esperado
        valid_f = self.hf_loss.mask.shape[1]
        pred = pred[:, :, :valid_f, :]
        target = target[:, :, :valid_f, :]

        pred_real = pred[:,0]
        pred_imag = pred[:,1]

        tgt_real = target[:,0]
        tgt_imag = target[:,1]

        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
        tgt_mag = torch.sqrt(tgt_real**2 + tgt_imag**2 + 1e-8)

        # complex L1
        complex_l1 = F.l1_loss(pred, target)

        # HF loss
        hf = self.hf_loss(pred_mag, tgt_mag)

        # Reconstruir audio aproximado (ISTFT simplificada)
        device = pred.device
        pred_complex = torch.complex(pred_real.cpu(), pred_imag.cpu())
        tgt_complex = torch.complex(tgt_real.cpu(), tgt_imag.cpu())

        window = torch.hann_window(1024).cpu()
        pred_audio = torch.istft(pred_complex, n_fft=1024, hop_length=256, window=window).to(device)
        tgt_audio = torch.istft(tgt_complex, n_fft=1024, hop_length=256, window=window).to(device)

        sc, mag = self.mrstft(pred_audio, tgt_audio)

        mrstft_loss = sc + mag

        return self.lambda_mrstft * mrstft_loss + self.lambda_hf * hf + self.lambda_complex * complex_l1
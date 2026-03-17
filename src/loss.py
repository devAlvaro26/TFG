# Loss para Audio Super-Resolución
# Este script contiene las funciones de pérdida para entrenar la red neuronal, calculando
# la pérdida en el dominio del tiempo y en el dominio de la frecuencia.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks

NFFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

def stft_mag(x, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """Calcula el STFT y devuelve la magnitud."""
    device = x.device
    x = x.cpu()
    window = torch.hann_window(win_length).cpu()

    stft_res = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
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
    def __init__(self, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):

        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft_mag(x, self.n_fft, self.hop_length, self.win_length)
        y_mag = stft_mag(y, self.n_fft, self.hop_length, self.win_length)

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
    def __init__(self, sr=44100, n_fft=NFFT, fmin=4000):

        super().__init__()

        # Crear máscara para ponderar más las altas frecuencias
        freqs = torch.linspace(0, sr/2, n_fft//2 + 1)
        mask = (freqs >= fmin).float()

        self.register_buffer("mask", mask.view(1, -1, 1))

    def forward(self, pred_mag, target_mag):
        return torch.mean(torch.abs(pred_mag - target_mag) * self.mask)


class MelLoss(nn.Module):
    """Pérdida de mel spectrogram."""
    def __init__(self, sr=44100, n_fft=NFFT, n_mels=80):

        super().__init__()

        mel_fb = melscale_fbanks(n_freqs=n_fft//2+1, f_min=0, f_max=sr/2, n_mels=n_mels, sample_rate=sr)
        self.register_buffer("mel_fb", mel_fb)

    def forward(self, pred_mag, tgt_mag):
        # pred_mag/tgt_mag shape: (B, F, T)
        pred_mel = torch.matmul(pred_mag.transpose(1, 2), self.mel_fb)
        tgt_mel  = torch.matmul(tgt_mag.transpose(1, 2), self.mel_fb)
        return F.l1_loss(torch.log(pred_mel + 1e-7), torch.log(tgt_mel + 1e-7))


class CombinedLoss(nn.Module):
    """Pérdida combinada de MRSTFT, HF, Complex y Mel."""
    def __init__(self, lambda_mrstft=1.0, lambda_hf=1.5, lambda_complex=0.5, lambda_mel=0.5):

        super().__init__()

        self.lambda_mrstft = lambda_mrstft
        self.lambda_hf = lambda_hf
        self.lambda_complex = lambda_complex
        self.lambda_mel = lambda_mel

        self.mrstft = MultiResolutionSTFTLoss()
        self.hf_loss = HighFrequencyLoss()
        self.mel_loss = MelLoss()

    def denormalize(self, stft):
        """Inversa de la log-compresión: sign(x) * (exp(|x|) - 1)."""
        sign = torch.sign(stft)
        return sign * (torch.exp(torch.abs(stft)) - 1)

    def forward(self, pred, target):
        # pred/target shape (B,2,F,T)
        
        # Descartar bins de frecuencia para coincidir con n_fft // 2 + 1 esperado
        valid_f = self.hf_loss.mask.shape[1]
        pred = pred[:, :, :valid_f, :]
        target = target[:, :, :valid_f, :]

        # L1 loss
        complex_l1 = F.l1_loss(pred, target)

        # Desnormalizar
        pred_denorm = self.denormalize(pred)
        target_denorm = self.denormalize(target)

        pred_real = pred_denorm[:,0]
        pred_imag = pred_denorm[:,1]

        tgt_real = target_denorm[:,0]
        tgt_imag = target_denorm[:,1]

        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
        tgt_mag = torch.sqrt(tgt_real**2 + tgt_imag**2 + 1e-8)

        # HF loss
        hf = self.hf_loss(pred_mag, tgt_mag)

        # MRSTFT loss
        device = pred.device
        pred_complex = torch.complex(pred_real.cpu(), pred_imag.cpu())
        tgt_complex = torch.complex(tgt_real.cpu(), tgt_imag.cpu())

        window = torch.hann_window(NFFT).cpu()
        pred_audio = torch.istft(pred_complex, n_fft=NFFT, hop_length=HOP_LENGTH, window=window).to(device)
        tgt_audio = torch.istft(tgt_complex, n_fft=NFFT, hop_length=HOP_LENGTH, window=window).to(device)

        sc, mag = self.mrstft(pred_audio, tgt_audio)

        mrstft_loss = sc + mag

        # Mel spectrogram loss
        mel_loss = self.mel_loss(pred_mag, tgt_mag)

        return self.lambda_mrstft * mrstft_loss + self.lambda_hf * hf + self.lambda_complex * complex_l1 + self.lambda_mel * mel_loss
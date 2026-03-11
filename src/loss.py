import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class STFTMagnitudeLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, freq_emphasis=True):
        """
        Pérdida espectral multi-objetivo: convergencia espectral (alpha),
        magnitud logarítmica L1 (beta) y MSE complejo (gamma).

        freq_emphasis: si True, pondera las altas frecuencias con más peso,
                       ya que son las que la red debe reconstruir en super-resolución.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.freq_emphasis = freq_emphasis

    def _magnitude(self, x):
        """Magnitud espectral a partir de (B, 2, F, T) → (B, F, T)."""
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-7)

    def _freq_weights(self, n_freqs, device):
        """
        Máscara de énfasis frecuencial: escala exponencial a lo largo de las frecuencias.
        Shape: (1, F, 1) — broadcast sobre batch y tiempo.
        """
        weights = torch.exp(torch.linspace(1.0, 2.0, n_freqs, device=device))
        return weights.view(1, -1, 1)

    def forward(self, y_hat, y):
        # y_hat, y: (B, 2, F, T) — canal 0: real, canal 1: imaginario

        mag_hat = self._magnitude(y_hat)  # (B, F, T)
        mag_y = self._magnitude(y)        # (B, F, T)

        # Énfasis en altas frecuencias
        if self.freq_emphasis:
            w = self._freq_weights(mag_y.size(1), mag_y.device)
            mag_hat_w = mag_hat * w
            mag_y_w = mag_y * w
        else:
            w = None
            mag_hat_w = mag_hat
            mag_y_w = mag_y

        # 1. Spectral Convergence — calculada por muestra y luego promediada
        sc_loss = (torch.linalg.matrix_norm(mag_y_w - mag_hat_w, ord="fro", dim=(-2, -1)) / (torch.linalg.matrix_norm(mag_y_w, ord="fro", dim=(-2, -1)) + 1e-7)).mean()

        # 2. Log-Magnitude L1 ponderada
        log_diff = torch.abs(torch.log10(1 + mag_hat) - torch.log10(1 + mag_y))
        if self.freq_emphasis:
            log_loss = (log_diff * w).mean()
        else:
            log_loss = log_diff.mean()

        # 3. Complex MSE — supervisión directa de fase y magnitud sobre (real, imag)
        complex_mse = F.mse_loss(y_hat, y)

        return (self.alpha * sc_loss) + (self.beta * log_loss) + (self.gamma * complex_mse)


class MelSpectrogramLoss(nn.Module):
    """
    Pérdida L1 sobre mel-spectrogramas en escala logarítmica.
    Calcula los mel-spectrogramas directamente a partir del STFT (B, 2, F, T)
    sin necesidad de ISTFT.
    """
    def __init__(self, n_mels=80, n_fft=1024, sample_rate=44100):
        super().__init__()
        # Crear mel filterbank: (n_freqs, n_mels)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer('mel_fb', mel_fb)   # (F_real, n_mels)

    def forward(self, y_hat, y):
        # y_hat, y: (B, 2, F, T) — pueden tener F > F_real por el padding
        n_freqs = self.mel_fb.size(0)   # F_real = n_fft//2 + 1

        # Magnitud truncada a los bins reales
        mag_hat = torch.sqrt(y_hat[:, 0, :n_freqs, :] ** 2 + y_hat[:, 1, :n_freqs, :] ** 2 + 1e-7)
        mag_y = torch.sqrt(y[:, 0, :n_freqs, :] ** 2 + y[:, 1, :n_freqs, :] ** 2 + 1e-7)

        # Proyectar a mel: (B, F_real, T) @ (F_real, n_mels) -> (B, n_mels, T)
        mel_hat = torch.matmul(mag_hat.transpose(-1, -2), self.mel_fb).transpose(-1, -2)
        mel_y = torch.matmul(mag_y.transpose(-1, -2), self.mel_fb).transpose(-1, -2)

        # L1 en escala log
        return F.l1_loss(torch.log(mel_hat + 1e-7), torch.log(mel_y + 1e-7))


class CombinedLoss(nn.Module):
    """
    Pérdida combinada que opera directamente sobre el STFT (B, 2, F, T)
    
    Combina:
      - STFTMagnitudeLoss: spectral convergence + log-magnitude L1 + complex MSE
      - MelSpectrogramLoss: L1 sobre mel-spectrogramas en escala log (perceptual)
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, mel_weight=0.0,
                 n_mels=80, n_fft=1024, sample_rate=44100):
        super().__init__()
        self.stft_loss = STFTMagnitudeLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.mel_loss = MelSpectrogramLoss(n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate)
        self.mel_weight = mel_weight

    def forward(self, y_hat, y):
        loss_stft = self.stft_loss(y_hat, y)
        if self.mel_weight > 0:
            loss_mel = self.mel_loss(y_hat, y)
            return loss_stft + self.mel_weight * loss_mel
        return loss_stft


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return F.mse_loss(y_hat, y)
# Loss para Audio Super-Resolución y el Discriminador
# Este script contiene las funciones de pérdida para entrenar la red neuronal, calculando
# la pérdida en el dominio del tiempo y en el dominio de la frecuencia.
# Basado en la Loss implementada en AERO: https://github.com/slp-rl/aero

import torch
import torch.nn as nn
import torch.nn.functional as F

NFFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
SAMPLE_RATE = 44100
FRAGMENT_LENGTH = 65536

class STFTLoss(nn.Module):
    """Pérdida STFT."""
    def __init__(self, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):

        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x, y):
        # x, y shape: (B, T)
        device = x.device
        if device.type in ['privateuseone', 'dml']:
            # Workaround para DirectML
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            window_cpu = self.window.cpu()
            
            x_stft = torch.stft(x_cpu, self.n_fft, self.hop_length, self.win_length, window_cpu, return_complex=True)
            y_stft = torch.stft(y_cpu, self.n_fft, self.hop_length, self.win_length, window_cpu, return_complex=True)

            x_mag = torch.abs(x_stft).to(device)
            y_mag = torch.abs(y_stft).to(device)
        else:
            x_stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)
            y_stft = torch.stft(y, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)

            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)

        # Spectral Convergence Loss
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-7)
        
        # Log-Magnitude Loss
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Pérdida STFT multi-resolución.
    Resoluciones basadas en AERO
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):


        super().__init__()


        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        # STFT diferentes resoluciones
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, ss, wl))

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        
        return sc_loss + mag_loss


class CombinedLoss(nn.Module):
    """Pérdida combinada de L1 y MR-STFT."""

    def __init__(self, n_fft=NFFT, hop_length=HOP_LENGTH, lambda_l1=1.0, lambda_mrstft=1.0):

        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.lambda_l1 = lambda_l1
        self.lambda_mrstft = lambda_mrstft
        
        # Buffer ISTFT
        self.register_buffer('base_window', torch.hann_window(n_fft))
        
        # MRSTFT
        self.mrstft_loss = MultiResolutionSTFTLoss()

    def _denormalize_stft(self, stft):
        """Inversa de log-compresión."""
        real = stft[:, 0]
        imag = stft[:, 1]
        
        mag_compressed = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_cos = real / mag_compressed
        phase_sin = imag / mag_compressed
        
        magnitude = torch.exp(mag_compressed) - 1
        
        return torch.stack([magnitude * phase_cos, magnitude * phase_sin], dim=1)

    def _stft_to_waveform(self, stft_ri):
        """Aplica ISTFT."""
        # Deshacer padding en frecuencia
        valid_freq_bins = self.n_fft // 2 + 1
        if stft_ri.size(2) > valid_freq_bins:
            stft_ri = stft_ri[:, :, :valid_freq_bins, :]
            
        # Deshacer padding en tiempo
        valid_time_frames = FRAGMENT_LENGTH // self.hop_length + 1
        if stft_ri.size(3) > valid_time_frames:
            stft_ri = stft_ri[:, :, :, :valid_time_frames]

        device = stft_ri.device
        if device.type in ['privateuseone', 'dml']:
            # Workaround para DirectML
            stft_ri_cpu = stft_ri.cpu()
            stft_complex = torch.complex(stft_ri_cpu[:, 0], stft_ri_cpu[:, 1])
            waveform = torch.istft(
                stft_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.base_window.cpu(),
                length=FRAGMENT_LENGTH
            )
            return waveform.to(device)
        else:
            stft_complex = torch.complex(stft_ri[:, 0], stft_ri[:, 1])
            waveform = torch.istft(
                stft_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.base_window,
                length=FRAGMENT_LENGTH
            )
            return waveform

    def get_waveforms(self, pred_stft, target_stft):
        """Pasar de STFT log-comprimido a waveform"""
        pred_linear = self._denormalize_stft(pred_stft)
        target_linear = self._denormalize_stft(target_stft)
        
        pred_wav = self._stft_to_waveform(pred_linear)
        target_wav = self._stft_to_waveform(target_linear)
    
        # Recortar a la misma longitud mínima
        min_len = min(pred_wav.size(-1), target_wav.size(-1))
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]
    
        # Asegurar shape (B, 1, T) para el discriminador 1D
        if pred_wav.ndim == 2:
            pred_wav = pred_wav.unsqueeze(1)
            target_wav = target_wav.unsqueeze(1)
        
        return pred_wav, target_wav

    def forward(self, pred_stft, target_stft, pred_wav=None, target_wav=None):
        # Si no se proporcionan waveforms precalculadas, calcular internamente
        if pred_wav is None or target_wav is None:
            # Descomprimir
            pred_linear = self._denormalize_stft(pred_stft)
            target_linear = self._denormalize_stft(target_stft)
            
            # Reconstrucción a Waveform
            pred_wav = self._stft_to_waveform(pred_linear)
            target_wav = self._stft_to_waveform(target_linear)
        
        # Recortar a longitud mínima
        min_len = min(pred_wav.size(-1), target_wav.size(-1))
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]
        
        # L1 loss
        loss_l1 = F.l1_loss(pred_wav, target_wav)
        
        # MRSTFT loss
        if pred_wav.ndim == 3:
            pred_wav = pred_wav.squeeze(1)
            target_wav = target_wav.squeeze(1)
        loss_mrstft = self.mrstft_loss(pred_wav, target_wav)
        
        return self.lambda_l1 * loss_l1 + self.lambda_mrstft * loss_mrstft


# Pérdidas del discriminador
class DiscriminatorLoss(nn.Module):
    """Pérdidas del discriminador."""
    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        """Pérdida LSGAN para el discriminador."""
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr_loss = torch.mean((1 - dr)**2)
            dg_loss = torch.mean(dg**2)
            loss += (dr_loss + dg_loss)
        return loss

    @staticmethod
    def generator_loss(disc_generated_outputs):
        """Pérdida adversarial para el generador."""
        loss = 0
        for dg in disc_generated_outputs:
            loss += torch.mean((1 - dg)**2)
        return loss

    @staticmethod
    def feature_matching_loss(fmap_r, fmap_g):
        """Pérdida L1 entre los feature maps del discriminador."""
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2 # 2x FM loss
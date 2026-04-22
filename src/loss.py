# Loss para Audio Super-Resolución y el Discriminador
# Este script contiene las funciones de pérdida para entrenar la red neuronal, calculando
# la pérdida en el dominio del tiempo y en el dominio de la frecuencia.
# Loss de validación basado en la Loss implementada en AERO: https://github.com/slp-rl/aero
# Loss de entrenamiento basado en la Loss implementada en HiFiGAN: https://github.com/jik876/hifi-gan

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_pesq import PesqLoss
from torchaudio.transforms import MelSpectrogram
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

NFFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
SAMPLE_RATE = 44100
FRAGMENT_LENGTH = 65536

class STFTLoss(nn.Module):
    """Pérdida de Magnitud Logarítmica y Convergencia Espectral (STFT)."""
    def __init__(self, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
        """
        Inicializa los parámetros de la STFT.
        Args:
            n_fft (int): Tamaño de la FFT.
            hop_length (int): Salto entre ventanas.
            win_length (int): Tamaño de la ventana.
        """
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
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)
        
        # Log-Magnitude Loss
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-8), torch.log(y_mag + 1e-8))

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Pérdida STFT multi-resolución basada en AERO."""
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        """
        Inicializa las pérdidas STFT para múltiples resoluciones.
        Args:
            fft_sizes (list[int]): Tamaños de FFT para cada resolución.
            hop_sizes (list[int]): Saltos entre ventanas para cada resolución.
            win_lengths (list[int]): Tamaños de ventana para cada resolución.
        """
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
    """Pérdida combinada L1 temporal y Multi-Resolución STFT."""
    def __init__(self, n_fft=NFFT, hop_length=HOP_LENGTH, lambda_l1=1.0, lambda_mrstft=1.0):
        """
        Inicializa la pérdida combinada.
        Args:
            n_fft (int): Tamaño de la FFT.
            hop_length (int): Salto entre ventanas.
            lambda_l1 (float): Peso de la pérdida L1.
            lambda_mrstft (float): Peso de la pérdida MRSTFT.
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.lambda_l1 = lambda_l1
        self.lambda_mrstft = lambda_mrstft
        
        # Buffer ISTFT
        self.register_buffer('base_window', torch.hann_window(n_fft))
        
        # MRSTFT
        self.mrstft_loss = MultiResolutionSTFTLoss()

    def _denormalize_stft(self, stft_ri):
        """
        Inversa de la compresión logarítmica de la magnitud.
        Args:
            stft_ri (torch.Tensor): Tensor STFT comprimido (B, 2, F, T).
        Returns:
            torch.Tensor: Tensor STFT en escala lineal (B, 2, F, T).
        """
        real = stft_ri[:, 0]
        imag = stft_ri[:, 1]
        
        mag_compressed = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_cos = real / mag_compressed
        phase_sin = imag / mag_compressed
        
        magnitude = torch.expm1(mag_compressed)
        
        return torch.stack([magnitude * phase_cos, magnitude * phase_sin], dim=1)

    def _stft_to_waveform(self, stft_ri):
        """
        Aplica la Inversa del STFT (ISTFT) para recuperar la forma de onda.
        Args:
            stft_ri (torch.Tensor): Tensor STFT real/imaginario.
        Returns:
            torch.Tensor: Tensor de la forma de onda reconstruida.
        """
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
        """
        Pasa de representaciones STFT log-comprimidas a formas de onda en el tiempo.
        Args:
            pred_stft (torch.Tensor): STFT predicción del modelo.
            target_stft (torch.Tensor): STFT objetivo del modelo.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Las formas de onda (B, 1, T)
                para predicción y objetivo.
        """
        # Deshacer compresión logarítmica
        pred_linear = self._denormalize_stft(pred_stft)
        target_linear = self._denormalize_stft(target_stft)
        
        # Reconstrucción a waveform
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
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, lambda_adv=1.0, lambda_fm=2.0, lambda_mel=45.0, n_mels=80):
        """
        Inicializa las pérdidas del discriminador.
        Args:
            sample_rate (int): Frecuencia de muestreo del audio.
            n_fft (int): Tamaño de la FFT.
            hop_length (int): Salto entre ventanas.
            win_length (int): Tamaño de la ventana.
            lambda_adv (float): Peso de la pérdida adversarial.
            lambda_fm (float): Peso del feature matching.
            lambda_mel (float): Peso de la pérdida Mel.
            n_mels (int): Número de bandas Mel.
        """
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            center=True,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        """
        Pérdida LSGAN del discriminador.
        Args:
            disc_real_outputs (list[torch.Tensor]): Salida del discriminador para inputs reales.
            disc_generated_outputs (list[torch.Tensor]): Salida para inputs generados.
        Returns:
            torch.Tensor: Valor de la pérdida total del discriminador.
        """
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr_loss = torch.mean((1 - dr)**2)
            dg_loss = torch.mean(dg**2)
            loss += (dr_loss + dg_loss)
        return self.lambda_adv * loss

    def generator_loss(self, disc_generated_outputs):
        """
        Pérdida LSGAN del generador.
        Args:
            disc_generated_outputs (list[torch.Tensor]): Salida del discriminador para inputs generados.
        Returns:
            torch.Tensor: Valor de la pérdida adversarial del generador.
        """
        loss = 0
        for dg in disc_generated_outputs:
            loss += torch.mean((1 - dg)**2)
        return self.lambda_adv * loss

    def feature_matching_loss(self, fmap_r, fmap_g):
        """
        Pérdida L1 entre los feature maps del discriminador.
        Args:
            fmap_r (list[list[torch.Tensor]]): Mapa de características sobre el audio real.
            fmap_g (list[list[torch.Tensor]]): Mapa de características sobre el audio generado.
        Returns:
            torch.Tensor: Valor acumulado de las diferencias L1.
        """
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return self.lambda_fm * loss

    def mel_spectrogram_loss(self, pred_wav, target_wav):
        """
        Pérdida L1 entre los Mel-spectrogramas del predictor y el objetivo.
        Args:
            pred_wav (torch.Tensor): Waveform predicción del modelo.
            target_wav (torch.Tensor): Waveform objetivo del modelo.
        Returns:
            torch.Tensor: Valor de la pérdida L1 entre los Mel-spectrogramas.
        """
        if pred_wav.ndim == 3:
            pred_wav = pred_wav.squeeze(1)
        if target_wav.ndim == 3:
            target_wav = target_wav.squeeze(1)

        device = pred_wav.device
        # Workaround para DirectML
        if device.type in ['privateuseone', 'dml']:
            self.mel_spec = self.mel_spec.cpu()
            pred_mel = torch.log(self.mel_spec(pred_wav.cpu()) + 1e-8)
            target_mel = torch.log(self.mel_spec(target_wav.cpu()) + 1e-8)
            mel_loss = F.l1_loss(pred_mel, target_mel).to(device)
        else:
            pred_mel = torch.log(self.mel_spec(pred_wav) + 1e-8)
            target_mel = torch.log(self.mel_spec(target_wav) + 1e-8)
            mel_loss = F.l1_loss(pred_mel, target_mel)

        return self.lambda_mel * mel_loss


class LossMetrics(nn.Module):
    """Métricas de calidad de audio."""
    @staticmethod
    def sisdr_loss(pred_wav, target_wav):
        """Pérdida SI-SDR."""
        if pred_wav.ndim == 3 and pred_wav.size(1) == 1:
            pred_wav = pred_wav.squeeze(1)
            target_wav = target_wav.squeeze(1)
        sisdr = ScaleInvariantSignalDistortionRatio().to(pred_wav.device)
        return sisdr(pred_wav, target_wav)
    
    @staticmethod
    def pesq_loss(pred_wav, target_wav):
        """Pérdida PESQ."""
        if pred_wav.ndim == 3 and pred_wav.size(1) == 1:
            pred_wav = pred_wav.squeeze(1)
            target_wav = target_wav.squeeze(1)
        pesq = PesqLoss(0.5, sample_rate=SAMPLE_RATE)
        return pesq.mos(target_wav.cpu(), pred_wav.cpu()).mean().item()

    @staticmethod
    def lsd_loss(pred_wav, target_wav, n_fft=NFFT, hop_length=HOP_LENGTH, eps=1e-8):
        """Pérdida LSD."""
        device = pred_wav.device
        if device.type in ['privateuseone', 'dml']:
            # Workaround para DirectML
            window = torch.hann_window(n_fft).cpu()
            pred_wav = pred_wav.cpu()
            target_wav = target_wav.cpu()
        else:
            window = torch.hann_window(n_fft).to(device)

        if pred_wav.ndim == 3 and pred_wav.size(1) == 1:
            pred_wav = pred_wav.squeeze(1)
            target_wav = target_wav.squeeze(1)
        
        Sx = torch.abs(torch.stft(pred_wav, n_fft, hop_length, window=window, return_complex=True))
        Sy = torch.abs(torch.stft(target_wav, n_fft, hop_length, window=window, return_complex=True))
        lsd = torch.mean((torch.log(Sx + eps) - torch.log(Sy + eps)) ** 2, dim=(-2, -1))
        return torch.mean(torch.sqrt(lsd)).item()
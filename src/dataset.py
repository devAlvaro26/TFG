# Script para cargar el dataset
# Devuelve pares de STFT (real + imaginario) con shape (2, F, T)

import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class AudioSuperResDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, segment_length=65536, n_fft=1024, hop_length=256):
        """
        Clase para cargar pares de audio (Alta Calidad - HR y Baja Calidad - LR)
        y devolver sus representaciones STFT.

        hr_dir: Directorio de los archivos en alta calidad (Ground Truth)
        lr_dir: Directorio de los archivos en baja calidad (Input)
        segment_length: Longitud del segmento de audio para el entrenamiento (en muestras). Debe ser compatible con n_fft y hop_length.
        n_fft: Tamaño de la FFT para el STFT.
        hop_length: Salto entre ventanas STFT.
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length

        # POOL_FACTOR = 2^4 = 16 (4 capas de pooling en la UNet 2D)
        self.pool_factor = 16

        # Contar ficheros automaticamente
        self.files = [
            f for f in os.listdir(hr_dir)
            if f.endswith('.wav') and os.path.exists(os.path.join(lr_dir, f))
        ]

    def __len__(self):
        return len(self.files)

    def _waveform_to_stft(self, waveform):
        """
        Convierte una waveform (1, L) a un tensor STFT de shape (2, F, T)
        donde canal 0 = parte real y canal 1 = parte imaginaria.
        """
        # STFT
        stft = torch.stft(
            waveform.squeeze(0),          #(L,)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )  #stft shape: (F, T)

        # Separar parte real e imaginaria
        stft_ri = torch.stack([stft.real, stft.imag], dim=0)
        return stft_ri

    def _normalize_stft(self, stft_ri):
        """
        Log-compresión del STFT para reducir el rango dinámico.
        Preserva el signo: sign(x) * log1p(|x|)
        Esto facilita enormemente la convergencia de la red.
        """
        sign = torch.sign(stft_ri)
        return sign * torch.log1p(torch.abs(stft_ri))

    def _pad_stft_to_pool_factor(self, stft):
        """
        Asegura que tanto F como T sean divisibles por pool_factor.
        Padding con ceros si es necesario.
        """
        _, freq_bins, time_frames = stft.shape

        # Pad frecuencia
        pad_f = (self.pool_factor - (freq_bins % self.pool_factor)) % self.pool_factor
        # Pad tiempo
        pad_t = (self.pool_factor - (time_frames % self.pool_factor)) % self.pool_factor

        if pad_f > 0 or pad_t > 0:
            stft = F.pad(stft, (0, pad_t, 0, pad_f))

        return stft

    def __getitem__(self, idx):
        filename = self.files[idx]
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        # Cargar audio en alta y baja calidad
        waveform_hr, sr_hr = torchaudio.load(hr_path)
        waveform_lr, sr_lr = torchaudio.load(lr_path)

        # Uniformizar a mono
        if waveform_hr.size(0) > 1:
            waveform_hr = waveform_hr.mean(dim=0, keepdim=True)
        if waveform_lr.size(0) > 1:
            waveform_lr = waveform_lr.mean(dim=0, keepdim=True)

        # Forzar 44.1kHz en ambos para alinear cortes
        if sr_hr != 44100:
            resampler = torchaudio.transforms.Resample(sr_hr, 44100)
            waveform_hr = resampler(waveform_hr)

        if sr_lr != 44100:
            resampler = torchaudio.transforms.Resample(sr_lr, 44100)
            waveform_lr = resampler(waveform_lr)

        # Asegurar que los archivos HR y LR tengan la misma longitud
        if len(waveform_hr) != len(waveform_lr):
            raise ValueError("Los archivos HR y LR no tienen la misma longitud.")

        min_len = min(waveform_hr.size(1), waveform_lr.size(1))

        # Descartar segmentos de silencio (evita sesgar el loss hacia cero)
        if waveform_hr.abs().max() < 0.01:
            return self.__getitem__((idx + 1) % len(self))

        # Normalizar a [-1, 1]
        max_val = max(waveform_hr.abs().max(), waveform_lr.abs().max()) + 1e-8
        waveform_hr = waveform_hr / max_val
        waveform_lr = waveform_lr / max_val

        # Random Crop o Padding
        if min_len < self.segment_length:
            # Padding: Si el audio es más corto, rellenamos con ceros a la derecha.
            pad_amount = self.segment_length - min_len
            waveform_hr = F.pad(waveform_hr[:, :min_len], (0, pad_amount))
            waveform_lr = F.pad(waveform_lr[:, :min_len], (0, pad_amount))
        else:
            # Random Crop: Si es más largo, elegimos un trozo aleatorio.
            # Esto ayuda al modelo a generalizar mejor (Data Augmentation implícito).
            start = torch.randint(0, min_len - self.segment_length, (1,)).item()
            waveform_hr = waveform_hr[:, start:start + self.segment_length]
            waveform_lr = waveform_lr[:, start:start + self.segment_length]

        # Calcular STFT de ambas waveforms
        stft_lr = self._waveform_to_stft(waveform_lr)
        stft_hr = self._waveform_to_stft(waveform_hr)

        # Normalizar STFT con log-compresión para reducir rango dinámico
        stft_lr = self._normalize_stft(stft_lr)
        stft_hr = self._normalize_stft(stft_hr)

        # Asegurar que F y T son divisibles por pool_factor (16)
        stft_lr = self._pad_stft_to_pool_factor(stft_lr)
        stft_hr = self._pad_stft_to_pool_factor(stft_hr)

        # Devolver STFT: Input LR y Target HR, ambos con shape (2, F, T)
        return stft_lr, stft_hr
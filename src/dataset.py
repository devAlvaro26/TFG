# Script para cargar el dataset
# Devuelve pares de STFT (real + imaginario) con shape (2, F, T) en fragmentos de 1.5s

import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

NFFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 44100
FRAGMENT_LENGTH = 65536

class AudioSuperResDataset(Dataset):
    """
    Carga pares de audio en Alta Calidad (HR) y Baja Calidad (LR),
    los divide en fragmentos de FRAGMENT_LENGTH muestras y devuelve
    sus representaciones STFT con shape (2, F, T).
    """
    def __init__(self, hr_dir, lr_dir, fragment_length=FRAGMENT_LENGTH, n_fft=NFFT, hop_length=HOP_LENGTH):
        """
        Inicializa el dataset cargando y fragmentando los archivos de audio.
        Args:
            hr_dir (str): Directorio de los archivos en alta calidad (Ground Truth).
            lr_dir (str): Directorio de los archivos en baja calidad (Input).
            fragment_length (int): Longitud del fragmento a procesar en muestras.
            n_fft (int): Tamaño de la FFT para el STFT.
            hop_length (int): Salto entre ventanas STFT.
        """

        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.fragment_length = fragment_length
        self.n_fft = n_fft
        self.hop_length = hop_length

        # POOL_FACTOR = 2^4 = 16 (4 capas de pooling en la UNet 2D)
        self.pool_factor = 16

        # Cache de resamplers
        self._resamplers = {}

        # Ventana para STFT
        self.window = torch.hann_window(self.n_fft)

        # Contar ficheros
        files = [
            f for f in os.listdir(hr_dir)
            if f.endswith('.wav') and os.path.exists(os.path.join(lr_dir, f))
        ]

        # Cargar ficheros y dividir en fragmentos
        self.fragments = []  # lista (hr_fragment, lr_fragment)
        for filename in files:
            hr_path = os.path.join(hr_dir, filename)
            lr_path = os.path.join(lr_dir, filename)

            # Cargar audio
            waveform_hr, sr_hr = torchaudio.load(hr_path)
            waveform_lr, sr_lr = torchaudio.load(lr_path)

            # Uniformizar a mono
            if waveform_hr.size(0) > 1:
                waveform_hr = waveform_hr.mean(dim=0, keepdim=True)
            if waveform_lr.size(0) > 1:
                waveform_lr = waveform_lr.mean(dim=0, keepdim=True)

            # Resamplear a 44.1 kHz
            if sr_hr != SAMPLE_RATE:
                key = ('hr', sr_hr)
                if key not in self._resamplers:
                    self._resamplers[key] = torchaudio.transforms.Resample(sr_hr, SAMPLE_RATE)
                waveform_hr = self._resamplers[key](waveform_hr)

            if sr_lr != SAMPLE_RATE:
                key = ('lr', sr_lr)
                if key not in self._resamplers:
                    self._resamplers[key] = torchaudio.transforms.Resample(sr_lr, SAMPLE_RATE)
                waveform_lr = self._resamplers[key](waveform_lr)

            # Normalizar a [-1, 1]
            max_val = max(waveform_hr.abs().max(), waveform_lr.abs().max()) + 1e-8
            waveform_hr = waveform_hr / max_val
            waveform_lr = waveform_lr / max_val

            # Calcular número de fragmentos
            min_len = min(waveform_hr.size(1), waveform_lr.size(1))
            num_fragments = min_len // self.fragment_length

            for i in range(num_fragments):
                # Obtener fragmentos
                start = i * self.fragment_length
                end = start + self.fragment_length
                frag_hr = waveform_hr[:, start:end]
                frag_lr = waveform_lr[:, start:end]

                # Descartar fragmentos de silencio
                if frag_hr.abs().max() < 1e-4:
                    continue

                self.fragments.append((frag_hr, frag_lr))

    def __len__(self):
        """Devuelve el número de fragmentos del conjunto."""
        return len(self.fragments)

    def _waveform_to_stft(self, waveform):
        """Convierte una waveform (1, L) a un tensor STFT de shape (2, F, T).
        El canal 0 corresponde a la parte real y el canal 1 a la parte imaginaria.
        Args:
            waveform (torch.Tensor): Tensor de audio de entrada con shape (1, L).
        Returns:
            torch.Tensor: Tensor STFT con partes real e imaginaria apiladas con shape (2, F, T).
        """
        stft = torch.stft(
            waveform.squeeze(0),          #(L,)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )  #stft shape: (F, T)

        # Apilar parte real e imaginaria
        stft_ri = torch.stack([stft.real, stft.imag], dim=0)
        return stft_ri

    def _normalize_stft(self, stft_ri):
        """
        Compresión logarítmica de la magnitud de la STFT preservando la fase.
        Args:
            stft_ri (torch.Tensor): Tensor STFT real/imaginario de dimensiones (2, F, T).
        Returns:
            torch.Tensor: Tensor comprimido (2, F, T).
        """
        real = stft_ri[0]
        imag = stft_ri[1]
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_cos = real / magnitude  # cos(phase)
        phase_sin = imag / magnitude  # sin(phase)
        
        mag_compressed = torch.log1p(magnitude)
        
        return torch.stack([mag_compressed * phase_cos, mag_compressed * phase_sin], dim=0)

    def _pad_stft_to_pool_factor(self, stft, pool_factor=16):
        """
        Aplica padding a STFT para que F y T sean divisibles por pool_factor.
        Args:
            stft (torch.Tensor): Tensor STFT (2, F, T).
            pool_factor (int): Factor de pooling.
        Returns:
            torch.Tensor: Tensor STFT con dimensiones divisibles por pool_factor.
        """
        _, freq_bins, time_frames = stft.shape

        # Pad frecuencia
        pad_f = (pool_factor - (freq_bins % pool_factor)) % pool_factor
        # Pad tiempo
        pad_t = (pool_factor - (time_frames % pool_factor)) % pool_factor

        if pad_f > 0 or pad_t > 0:
            stft = F.pad(stft, (0, pad_t, 0, pad_f), mode='reflect')    # Padding mediante reflexión

        return stft

    def __getitem__(self, idx):
        frag_hr, frag_lr = self.fragments[idx]

        # Calcular STFT de ambas waveforms
        stft_lr = self._waveform_to_stft(frag_lr)
        stft_hr = self._waveform_to_stft(frag_hr)

        # Normalizar STFT con log-compresión
        stft_lr = self._normalize_stft(stft_lr)
        stft_hr = self._normalize_stft(stft_hr)

        # Asegurar que F y T son divisibles por pool_factor (16)
        stft_lr = self._pad_stft_to_pool_factor(stft_lr, self.pool_factor)
        stft_hr = self._pad_stft_to_pool_factor(stft_hr, self.pool_factor)

        # Devolver STFT: Input LR y Target HR, ambos con shape (2, F, T)
        return stft_lr, stft_hr
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
    Cargar pares de audio (Alta Calidad - HR y Baja Calidad - LR),
    dividirlos en fragmentos de FRAGMENT_LENGTH muestras y devolver
    sus representaciones STFT con shape (2, F, T).

    hr_dir: Directorio de los archivos en alta calidad (Ground Truth)
    lr_dir: Directorio de los archivos en baja calidad (Input)
    fragment_length: Longitud del fragmento a procesar en muestras
    n_fft: Tamaño de la FFT para el STFT.
    hop_length: Salto entre ventanas STFT.
    """
    def __init__(self, hr_dir, lr_dir, fragment_length=FRAGMENT_LENGTH, n_fft=NFFT, hop_length=HOP_LENGTH):

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

            # Cargar los audios
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

            # Dividir en fragmentos de fragment_length muestras
            min_len = min(waveform_hr.size(1), waveform_lr.size(1))
            num_fragments = min_len // self.fragment_length

            for i in range(num_fragments):
                start = i * self.fragment_length
                end = start + self.fragment_length
                frag_hr = waveform_hr[:, start:end]
                frag_lr = waveform_lr[:, start:end]

                # Descartar fragmentos de silencio
                if frag_hr.abs().max() < 1e-4:
                    continue

                self.fragments.append((frag_hr, frag_lr))

        print(f"[Dataset] {len(files)} ficheros: {len(self.fragments)} fragmentos de {FRAGMENT_LENGTH} muestras.")

    def __len__(self):
        return len(self.fragments)

    def _waveform_to_stft(self, waveform):
        """
        Convierte una waveform (1, L) a un tensor STFT de shape (2, F, T)
        donde canal 0 = parte real y canal 1 = parte imaginaria.
        """
        stft = torch.stft(
            waveform.squeeze(0),          #(L,)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )  #stft shape: (F, T)

        # Separar parte real e imaginaria
        stft_ri = torch.stack([stft.real, stft.imag], dim=0)
        return stft_ri

    def _normalize_stft(self, stft_ri):
        """
        Log-compresión del STFT para reducir el rango dinámico.
        Preserva el signo: sign(x) * log1p(|x|)
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
        frag_hr, frag_lr = self.fragments[idx]

        # Calcular STFT de ambas waveforms
        stft_lr = self._waveform_to_stft(frag_lr)
        stft_hr = self._waveform_to_stft(frag_hr)

        # Normalizar STFT con log-compresión para reducir rango dinámico
        stft_lr = self._normalize_stft(stft_lr)
        stft_hr = self._normalize_stft(stft_hr)

        # Asegurar que F y T son divisibles por pool_factor (16)
        stft_lr = self._pad_stft_to_pool_factor(stft_lr)
        stft_hr = self._pad_stft_to_pool_factor(stft_hr)

        # Devolver STFT: Input LR y Target HR, ambos con shape (2, F, T)
        return stft_lr, stft_hr
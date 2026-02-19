# Script para cargar el dataset

import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class AudioSuperResDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, segment_length=65536):
        """
        Clase para cargar pares de audio (Alta Calidad - HR y Baja Calidad - LR).

        hr_dir: Directorio de los archivos en alta calidad (Ground Truth)
        lr_dir: Directorio de los archivos en baja calidad (Input)
        segment_length: Longitud del segmento de audio para el entrenamiento (en muestras). Debe ser divisible por 8 debido a las 3 capas de pooling en UNet (2^3 = 8).
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        
        # Contar ficheros automaticamente
        self.files = [f for f in os.listdir(hr_dir) if f.endswith('.wav') and os.path.exists(os.path.join(lr_dir, f))]
        
        # Longitud del segmento de audio
        self.segment_length = segment_length

    def __len__(self):
        # Devuelve el tamaño total del dataset
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        # Cargar audio en alta y baja calidad
        waveform_hr, sr_hr = torchaudio.load(hr_path)
        waveform_lr, sr_lr = torchaudio.load(lr_path)

        # Forzar 44.1kHz en ambos para alinear cortes
        if sr_hr != 44100:
            resampler = torchaudio.transforms.Resample(sr_hr, 44100)
            waveform_hr = resampler(waveform_hr)
            
        if sr_lr != 44100:
            resampler = torchaudio.transforms.Resample(sr_lr, 44100)
            waveform_lr = resampler(waveform_lr)

        # Uniformizar archivos
        if waveform_hr.size(0) > 1:
            waveform_hr = waveform_hr.mean(dim=0, keepdim=True)
        if waveform_lr.size(0) > 1:
            waveform_lr = waveform_lr.mean(dim=0, keepdim=True)

        min_len = min(waveform_hr.size(1), waveform_lr.size(1))
        
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
            waveform_hr = waveform_hr[:, start:start+self.segment_length]
            waveform_lr = waveform_lr[:, start:start+self.segment_length]

        # Normalizar a [-1, 1]
        waveform_hr = waveform_hr / (waveform_hr.abs().max() + 1e-8)
        waveform_lr = waveform_lr / (waveform_lr.abs().max() + 1e-8)

        # Devolver waveforms directamente: shape (1, segment_length)
        # Input: LR waveform, Target: HR waveform
        return waveform_lr, waveform_hr
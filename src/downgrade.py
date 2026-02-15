# Script para bajar la frecuencia de muestreo de los archivos de audio y crear el dataset de entrenamiento

import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample

fs = 8000 #Frecuencia de muestreo objetivo
HR_DIR = 'D:/Proyectos/TFG/data/train/HR'
LR_DIR = 'D:/Proyectos/TFG/data/train/LR'

files = [f for f in os.listdir(HR_DIR) if f.endswith('.wav')]
for file in files:
    fs_original, audio = wav.read(os.path.join(HR_DIR, file))

    # Resampliar a nuevo fs
    num_samples_nuevo = int(len(audio) * fs / fs_original)
    audio_resampled = resample(audio, num_samples_nuevo).astype(np.int16)

    # Guardar con la nueva frecuencia de muestreo
    wav.write(os.path.join(LR_DIR, file), fs, audio_resampled)

    print(f"{file} resampleado a {fs} Hz")
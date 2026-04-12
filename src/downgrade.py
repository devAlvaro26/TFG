# Script para degradar archivos de audio HR a baja resolución

import os
import torchaudio
from torchaudio import functional as F

TRAIN_HR_DIR = './data/dataset/train/HR'
TRAIN_LR_DIR = './data/dataset/train/LR'
TEST_HR_DIR = './data/dataset/test/HR'
TEST_LR_DIR = './data/dataset/test/LR'
TARGETS_SR = [8000, 11025, 22050] # Frecuencias de muestreo objetivo para degradar

files_train = [f for f in os.listdir(TRAIN_HR_DIR) if f.endswith('.wav')]
files_test = [f for f in os.listdir(TEST_HR_DIR) if f.endswith('.wav')]

print(f"Degradando {len(files_train)} archivos de entrenamiento")
for i, (file) in enumerate(files_train):
    waveform, sr = torchaudio.load(os.path.join(TRAIN_HR_DIR, file))
    
    # Degradar un archivo a cada frecuencia de muestreo objetivo
    TARGET_SR = TARGETS_SR[i%len(TARGETS_SR)]
    downsampled = F.resample(waveform, sr, TARGET_SR)
    
    torchaudio.save(
        os.path.join(TRAIN_LR_DIR, file),
        downsampled, TARGET_SR,
        bits_per_sample=16,
        encoding="PCM_S"
    )
    print(f"{file}: degradado a {TARGET_SR}Hz")

print(f"Degradando {len(files_test)} archivos de validación")
for i, (file) in enumerate(files_test):
    waveform, sr = torchaudio.load(os.path.join(TEST_HR_DIR, file))
    
    TARGET_SR = TARGETS_SR[i%len(TARGETS_SR)]
    downsampled = F.resample(waveform, sr, TARGET_SR)
    
    torchaudio.save(
        os.path.join(TEST_LR_DIR, file),
        downsampled, TARGET_SR,
        bits_per_sample=16,
        encoding="PCM_S"
    )
    print(f"{file}: degradado a {TARGET_SR}Hz")
import os
import torchaudio
import torchaudio.functional as F

HR_DIR = './data/train/HR'
LR_DIR = './data/train/LR'
TARGET_SR = 8000

files = [f for f in os.listdir(HR_DIR) if f.endswith('.wav')]
for file in files:
    waveform, sr = torchaudio.load(os.path.join(HR_DIR, file))
    
    # Degradar a 8kHz
    downsampled = F.resample(waveform, sr, TARGET_SR)
    
    torchaudio.save(
        os.path.join(LR_DIR, file),
        downsampled, TARGET_SR,
        bits_per_sample=16,
        encoding="PCM_S"
    )
    print(f"{file}: degradado a {TARGET_SR}Hz")
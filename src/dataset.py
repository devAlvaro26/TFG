import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class AudioSuperResDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, segment_length=65536):
        """
        hr_dir: High Resolution wav folder
        lr_dir: Low Resolution wav folder
        segment_length: Audio segment length (must be divisible by 8 for 3 pooling layers)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        
        # Filter files that exist in both folders
        self.files = [f for f in os.listdir(hr_dir) if f.endswith('.wav') and os.path.exists(os.path.join(lr_dir, f))]
        
        # Segment length must be divisible by 8 (2^3 for 3 pooling layers)
        self.segment_length = segment_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        # Load High-Res and Low-Res audio
        waveform_hr, sr_hr = torchaudio.load(hr_path)
        waveform_lr, sr_lr = torchaudio.load(lr_path)

        # Force 44.1kHz in both to align cuts
        if sr_hr != 44100:
            resampler = torchaudio.transforms.Resample(sr_hr, 44100)
            waveform_hr = resampler(waveform_hr)
            
        if sr_lr != 44100:
            resampler = torchaudio.transforms.Resample(sr_lr, 44100)
            waveform_lr = resampler(waveform_lr)

        # Convert to mono if stereo (take average of channels)
        if waveform_hr.size(0) > 1:
            waveform_hr = waveform_hr.mean(dim=0, keepdim=True)
        if waveform_lr.size(0) > 1:
            waveform_lr = waveform_lr.mean(dim=0, keepdim=True)

        # Cut or pad to segment_length (required for batching)
        min_len = min(waveform_hr.size(1), waveform_lr.size(1))
        
        if min_len < self.segment_length:
            # Pad if audio is shorter than segment_length
            pad_amount = self.segment_length - min_len
            waveform_hr = F.pad(waveform_hr[:, :min_len], (0, pad_amount))
            waveform_lr = F.pad(waveform_lr[:, :min_len], (0, pad_amount))
        else:
            # Random cut if audio is longer
            start = torch.randint(0, min_len - self.segment_length, (1,)).item()
            waveform_hr = waveform_hr[:, start:start+self.segment_length]
            waveform_lr = waveform_lr[:, start:start+self.segment_length]

        # Normalize to [-1, 1] range
        waveform_hr = waveform_hr / (waveform_hr.abs().max() + 1e-8)
        waveform_lr = waveform_lr / (waveform_lr.abs().max() + 1e-8)

        # Return waveforms directly: shape (1, segment_length)
        # Input: LR waveform, Target: HR waveform
        return waveform_lr, waveform_hr
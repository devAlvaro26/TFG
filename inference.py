import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import numpy as np
from src.model import UNetAudio

MODEL_PATH = 'unet_superres.pth'
TEST_DIR = './data/test'       # Test directory
OUTPUT_DIR = './results'       # Results directory
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_audio(tensor, path, sample_rate):
    if tensor.ndim == 3: 
        tensor = tensor.squeeze(0)
    torchaudio.save(path, tensor.cpu(), sample_rate)

def save_waveform_plot(lr, sr, filename, sample_rate):
    """
    Generates a comparative waveform plot.
    """
    wave_lr = lr.squeeze().cpu().numpy()
    wave_sr = sr.squeeze().cpu().numpy()

    # Calculate time axis
    duration_secs = len(wave_lr) / sample_rate
    time_axis = np.linspace(0, duration_secs, len(wave_lr))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Low Res (Input)
    axs[0].plot(time_axis, wave_lr, color='steelblue', linewidth=0.5)
    axs[0].set_title("Input (Low Res)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_ylim(-1, 1)
    axs[0].grid(True, alpha=0.3)

    # Super Res (Output)
    axs[1].plot(time_axis, wave_sr, color='darkorange', linewidth=0.5)
    axs[1].set_title("Output (Super Res - Reconstructed)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(-1, 1)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def save_spectrogram_plot(lr_waveform, sr_waveform, filename, sample_rate, n_fft=1024, hop_length=256):
    """
    Generates a comparative spectrogram plot from waveforms.
    Converts 1D waveforms to spectrograms for visualization.
    """
    # Create spectrogram transform
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
    
    # Generate spectrograms from waveforms
    # Ensure tensors are on CPU and have correct shape
    lr_wave = lr_waveform.cpu() if lr_waveform.is_cuda else lr_waveform
    sr_wave = sr_waveform.cpu() if sr_waveform.is_cuda else sr_waveform
    
    # Calculate spectrograms
    spec_lr = spec_transform(lr_wave).squeeze().numpy()
    spec_sr = spec_transform(sr_wave).squeeze().numpy()
    
    # Convert to dB scale for better visualization
    spec_lr_db = 10 * np.log10(spec_lr + 1e-10)
    spec_sr_db = 10 * np.log10(spec_sr + 1e-10)

    # Calculations for the axes
    nyquist = sample_rate / 2
    num_frames = spec_lr.shape[1]
    duration_secs = (num_frames * hop_length) / sample_rate 
    
    # [xmin, xmax, ymin, ymax]
    extent_val = [0, duration_secs, 0, nyquist] 

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Low Res (Input)
    im1 = axs[0].imshow(spec_lr_db, origin='lower', aspect='auto', cmap='magma', extent=extent_val)
    axs[0].set_title("Input (Low Res)")
    axs[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axs[0], label="Amplitude (dB)")

    # Super Res (Output)
    im2 = axs[1].imshow(spec_sr_db, origin='lower', aspect='auto', cmap='magma', extent=extent_val)
    axs[1].set_title("Output (Super Res)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_xlabel("Time (s)")
    fig.colorbar(im2, ax=axs[1], label="Amplitude (dB)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def inference():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNetAudio().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print("Error: Model not found (.pth)")
        return
    model.eval()

    # Search files in the test directory
    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.wav')]
    if not files:
        print(f"No .wav files found in {TEST_DIR}")
        return

    print(f"Found {len(files)} files to process.")

    # Loop through each file
    for filename in files:
        file_path = os.path.join(TEST_DIR, filename)
        print(f"Processing: {filename}...")

        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Ensure 44.1kHz
        if sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
            sample_rate = 44100

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform[0:1, :]

        # Clip to 10 seconds max to avoid memory overflow
        MAX_SEC = 10
        if waveform.size(1) > sample_rate * MAX_SEC:
            waveform = waveform[:, :sample_rate * MAX_SEC]

        # Pad to be divisible by 8 (for 3 pooling layers with factor 2)
        original_length = waveform.size(1)
        pad_amount = (8 - (original_length % 8)) % 8
        if pad_amount > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Normalize input
        max_val = waveform.abs().max() + 1e-8
        waveform_normalized = waveform / max_val

        # Add batch dimension: shape (1, 1, length)
        waveform_input = waveform_normalized.unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            predicted_waveform = model(waveform_input)

        # Post-processing
        # Remove batch dimension
        predicted_waveform = predicted_waveform.squeeze(0)
        
        # Remove padding
        if pad_amount > 0:
            predicted_waveform = predicted_waveform[:, :original_length]
            waveform_normalized = waveform_normalized[:, :original_length]

        # Denormalize (scale back to original range)
        predicted_waveform = predicted_waveform * max_val

        # Clamp to valid audio range
        predicted_waveform = torch.clamp(predicted_waveform, -1.0, 1.0)

        print(f"  Output stats - Min: {predicted_waveform.min():.4f}, Max: {predicted_waveform.max():.4f}, Mean: {predicted_waveform.mean():.4f}")

        # Save Results
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_DIR, base_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save audios
        # Input
        save_audio(waveform[:, :original_length], os.path.join(save_path, 'input.wav'), 44100)
        # Super resolution output
        save_audio(predicted_waveform, os.path.join(save_path, 'super_res.wav'), 44100)

        # Save waveform comparison plot
        save_waveform_plot(
            waveform_normalized[:, :original_length],
            predicted_waveform.cpu() / max_val,  # Normalize for plotting
            os.path.join(save_path, 'waveform.png'),
            sample_rate
        )

        # Save spectrogram comparison plot
        save_spectrogram_plot(
            waveform_normalized[:, :original_length],
            predicted_waveform.cpu() / max_val,  # Normalize for plotting
            os.path.join(save_path, 'spectrogram.png'),
            sample_rate
        )

    print(f"Done. Check the {OUTPUT_DIR} folder")

if __name__ == "__main__":
    inference()
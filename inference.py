import os
import shutil
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from src.model import UNetAudio

# --- Configuration ---
MODEL_PATH = 'unet_superres.pth'
TEST_DIR = './data/test'
OUTPUT_DIR = './results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TARGET_SR = 44100       # Target sample rate (must match training)
MAX_SECONDS = 10        # Max duration to process (avoids memory overflow)
POOL_FACTOR = 8         # 2^3 for 3 pooling layers in UNet


def save_audio(tensor, path, sample_rate):
    """
    Save a waveform tensor as a 16-bit PCM WAV file.
    """
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    torchaudio.save(path, tensor.cpu(), sample_rate, bits_per_sample=16, encoding="PCM_S")


def postprocess(predicted, max_val, original_length, pad_amount):
    """
    Post-process model output: remove batch dim, unpad, denormalize, and clamp.
    """
    # Remove batch dimension: (1, 1, L) → (1, L)
    predicted = predicted.squeeze(0)

    # Remove padding
    if pad_amount > 0:
        predicted = predicted[:, :original_length]

    # Denormalize and clamp to valid audio range
    predicted = (predicted * max_val).clamp(-1.0, 1.0)

    return predicted


def save_waveform_plot(lr, sr, filename, sample_rate):
    """
    Generate a comparative waveform plot (Input vs Super-Res output).
    """
    wave_lr = lr.squeeze().cpu().numpy()
    wave_sr = sr.squeeze().cpu().numpy()

    time_lr = np.linspace(0, len(wave_lr) / sample_rate, len(wave_lr))
    time_sr = np.linspace(0, len(wave_sr) / sample_rate, len(wave_sr))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    axs[0].plot(time_lr, wave_lr, color='steelblue', linewidth=0.5)
    axs[0].set_title("Input (Low Res)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_ylim(-1, 1)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time_sr, wave_sr, color='darkorange', linewidth=0.5)
    axs[1].set_title("Output (Super Res)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(-1, 1)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_spectrogram_plot(lr_waveform, sr_waveform, filename, lr_sample_rate, sr_sample_rate, n_fft=1024, hop_length=256):
    """
    Generate a comparative spectrogram plot from waveforms.
    Each waveform can have a different sample rate.
    """
    spec_transform_lr = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
    spec_transform_sr = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)

    lr_wave = lr_waveform.cpu()
    sr_wave = sr_waveform.cpu()

    spec_lr = spec_transform_lr(lr_wave).squeeze().numpy()
    spec_sr = spec_transform_sr(sr_wave).squeeze().numpy()

    # Convert to dB scale
    spec_lr_db = 10 * np.log10(spec_lr + 1e-10)
    spec_sr_db = 10 * np.log10(spec_sr + 1e-10)

    nyquist_lr = lr_sample_rate / 2
    nyquist_sr = sr_sample_rate / 2
    duration_lr = (spec_lr.shape[1] * hop_length) / lr_sample_rate
    duration_sr = (spec_sr.shape[1] * hop_length) / sr_sample_rate
    extent_lr = [0, duration_lr, 0, nyquist_lr]
    extent_sr = [0, duration_sr, 0, nyquist_sr]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    im1 = axs[0].imshow(spec_lr_db, origin='lower', aspect='auto', cmap='magma', extent=extent_lr)
    axs[0].set_title("Input (Low Res)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_ylim(0, 20000)
    fig.colorbar(im1, ax=axs[0], label="Amplitude (dB)")

    im2 = axs[1].imshow(spec_sr_db, origin='lower', aspect='auto', cmap='magma', extent=extent_sr)
    axs[1].set_title("Output (Super Res)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_ylim(0, 20000)
    axs[1].set_xlabel("Time (s)")
    fig.colorbar(im2, ax=axs[1], label="Amplitude (dB)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def inference():
    """
    Run super-resolution inference on all .wav files in TEST_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNetAudio().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    model.eval()

    # Discover test files
    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.wav')]
    if not files:
        print(f"No .wav files found in {TEST_DIR}")
        return

    print(f"Found {len(files)} file(s) to process.\n")

    resamplers = {}

    for filename in files:
        file_path = os.path.join(TEST_DIR, filename)
        print(f"Processing: {filename}")

        # Load audio
        waveform, original_sr = torchaudio.load(file_path)

        # Save original waveform (before resampling) for input spectrogram
        original_waveform = waveform.clone()
        if original_waveform.size(0) > 1:
            original_waveform = original_waveform.mean(dim=0, keepdim=True)
        max_samples_orig = original_sr * MAX_SECONDS
        if original_waveform.size(1) > max_samples_orig:
            original_waveform = original_waveform[:, :max_samples_orig]

        # Resample if necessary (using cached resampler)
        if original_sr != TARGET_SR:
            if original_sr not in resamplers:
                 resamplers[original_sr] = torchaudio.transforms.Resample(original_sr, TARGET_SR).to(DEVICE)
            
            waveform = resamplers[original_sr](waveform)
            current_sr = TARGET_SR
        else:
            current_sr = original_sr

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate to MAX_SECONDS
        max_samples = TARGET_SR * MAX_SECONDS
        if waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]

        # Pad to be divisible by POOL_FACTOR
        original_length = waveform.size(1)
        pad_amount = (POOL_FACTOR - (original_length % POOL_FACTOR)) % POOL_FACTOR
        if pad_amount > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Normalize to [-1, 1]
        max_val = waveform.abs().max() + 1e-8
        waveform_norm = waveform / max_val

        # Save normalized input without padding for plotting
        input_for_plot = waveform_norm[:, :original_length].clone()

        # Inference
        waveform_input = waveform_norm.unsqueeze(0).to(DEVICE)  # (1, 1, L)
        with torch.no_grad():
            predicted = model(waveform_input)

        # Post-process
        predicted = postprocess(predicted, max_val, original_length, pad_amount)

        print(f"Output stats — Min: {predicted.min():.4f}, Max: {predicted.max():.4f}, Mean: {predicted.mean():.4f}")

        # Save results
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(save_path, exist_ok=True)

        # Copy original low-quality input file
        shutil.copy2(file_path, os.path.join(save_path, 'input.wav'))

        # Save reconstructed super-resolution output
        save_audio(predicted, os.path.join(save_path, 'super_res.wav'), TARGET_SR)

        # Normalize predicted for plotting (same scale as input)
        predicted_for_plot = predicted.cpu() / max_val

        # Waveform comparison (original vs reconstructed)
        save_waveform_plot(
            input_for_plot,
            predicted_for_plot,
            os.path.join(save_path, 'waveform.png'),
            TARGET_SR,
        )

        # Spectrogram comparison (original at native SR vs reconstructed at target SR)
        save_spectrogram_plot(
            original_waveform,
            predicted_for_plot,
            os.path.join(save_path, 'spectrogram.png'),
            lr_sample_rate=original_sr,
            sr_sample_rate=TARGET_SR,
        )

        print(f"Saved to {save_path}/\n")

    print(f"Done. Check the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    inference()
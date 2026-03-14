# Script para realizar inferencia
# Este script cargará el modelo entrenado y generará resultados

import os
import torch
import torchaudio
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt
from src.model import UNetAudio2D

MODEL_PATH = 'unet2D_superres.pt'   # Archivo del modelo entrenado
INF_DIR = './data/inference'        # Archivos de entrada
OUTPUT_DIR = './results'            # Archivos de salida

try:
    import torch_directml
    has_dml = torch_directml.is_available()
except ImportError:
    has_dml = False

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif has_dml:
    DEVICE = torch_directml.device()
else:
    DEVICE = 'cpu'

TARGET_SR = 44100       # Target sample rate
POOL_FACTOR = 16        # 2^4 para 4 capas de pooling en UNet 2D
N_FFT = 1024            # Tamaño de la FFT para STFT
HOP_LENGTH = 256        # Salto entre ventanas STFT


def save_audio(tensor, path, sample_rate):
    """Guarda una forma de onda en un archivo de audio."""
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    torchaudio.save(path, tensor.cpu(), sample_rate, bits_per_sample=16, encoding="PCM_S")


def waveform_to_stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convierte una forma de onda a un STFT con canales real e imaginario."""
    original_device = waveform.device
    waveform_cpu = waveform.cpu()
    if waveform_cpu.ndim == 2:
        waveform_cpu = waveform_cpu.squeeze(0)

    window = torch.hann_window(n_fft, device='cpu')
    stft = torch.stft(
        waveform_cpu,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    # Devolver como tensor de 2 canales: real e imaginario
    return torch.stack([stft.real, stft.imag], dim=0).to(original_device)

def stft_to_waveform(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE):
    """Convierte un STFT con canales real e imaginario de vuelta a forma de onda usando ISTFT."""
    stft_cpu = stft.cpu()
    
    stft_complex = torch.complex(stft_cpu[0], stft_cpu[1])
    window = torch.hann_window(n_fft, device='cpu')
    waveform = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
    )
    return waveform.unsqueeze(0).to(device)


def normalize_stft(stft):
    """Log-compresión del STFT."""
    sign = torch.sign(stft)
    return sign * torch.log1p(torch.abs(stft))

def denormalize_stft(stft):
    """Inversa de la log-compresión: sign(x) * (exp(|x|) - 1)."""
    sign = torch.sign(stft)
    return sign * (torch.exp(torch.abs(stft)) - 1)


def pad_stft(stft, pool_factor=POOL_FACTOR):
    """Ajusta el STFT para que sea divisible por el factor de pooling."""
    _, freq_bins, time_frames = stft.shape
    pad_f = (pool_factor - (freq_bins % pool_factor)) % pool_factor
    pad_t = (pool_factor - (time_frames % pool_factor)) % pool_factor

    if pad_f > 0 or pad_t > 0:
        stft = torch.nn.functional.pad(stft, (0, pad_t, 0, pad_f))

    return stft, freq_bins, time_frames


def save_waveform_plot(lr, sr, filename, sample_rate):
    """Genera un gráfico de la forma de onda de entrada y salida."""
    wave_lr = lr.squeeze().cpu().numpy()
    wave_sr = sr.squeeze().cpu().numpy()

    time_lr = np.linspace(0, len(wave_lr) / sample_rate, len(wave_lr))
    time_sr = np.linspace(0, len(wave_sr) / sample_rate, len(wave_sr))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    axs[0].plot(time_lr, wave_lr, color='steelblue', linewidth=0.5)
    axs[0].set_title("Input (Low Res)")
    axs[0].set_ylabel("Amplitud")
    axs[0].set_ylim(-1, 1)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time_sr, wave_sr, color='darkorange', linewidth=0.5)
    axs[1].set_title("Output (Super Res)")
    axs[1].set_ylabel("Amplitud")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylim(-1, 1)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_spectrogram_plot(lr_waveform, sr_waveform, filename, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Genera un gráfico del espectrograma de entrada y salida."""
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)

    spec_lr = spec_transform(lr_waveform.cpu()).squeeze().numpy()
    spec_sr = spec_transform(sr_waveform.cpu()).squeeze().numpy()

    # Escala dB
    spec_lr_db = 10 * np.log10(spec_lr + 1e-10)
    spec_sr_db = 10 * np.log10(spec_sr + 1e-10)

    vmin = min(spec_lr_db.min(), spec_sr_db.min())
    vmax = max(spec_lr_db.max(), spec_sr_db.max())

    nyquist = sample_rate / 2
    duration_lr = (spec_lr.shape[1] * hop_length) / sample_rate
    duration_sr = (spec_sr.shape[1] * hop_length) / sample_rate
    extent_lr = [0, duration_lr, 0, nyquist]
    extent_sr = [0, duration_sr, 0, nyquist]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    im1 = axs[0].imshow(spec_lr_db, origin='lower', aspect='auto', cmap='magma', extent=extent_lr, vmin=vmin, vmax=vmax)
    axs[0].set_title("Input (Low Res)")
    axs[0].set_ylabel("Frecuencia (Hz)")
    axs[0].set_ylim(0, nyquist)
    fig.colorbar(im1, ax=axs[0], label="Amplitud (dB)")

    im2 = axs[1].imshow(spec_sr_db, origin='lower', aspect='auto', cmap='magma', extent=extent_sr, vmin=vmin, vmax=vmax)
    axs[1].set_title("Output (Super Res)")
    axs[1].set_ylabel("Frecuencia (Hz)")
    axs[1].set_ylim(0, nyquist)
    axs[1].set_xlabel("Tiempo (s)")
    fig.colorbar(im2, ax=axs[1], label="Amplitud (dB)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def inference():
    """Realiza inferencia en los archivos de entrada y guardar los resultados."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cargar modelo
    print(f"Cargando modelo desde {MODEL_PATH}...")
    model = UNetAudio2D().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)) # weights_only=False para compatibilidad con AMD
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo '{MODEL_PATH}'.")
        return
    model.eval()

    # Procesar archivos de inferencia
    files = [f for f in os.listdir(INF_DIR) if f.endswith('.wav')]
    if not files:
        print(f"No se encontraron archivos .wav en {INF_DIR}")
        return

    print(f"Encontrados {len(files)} archivos para procesar.")

    resamplers = {}

    for filename in files:
        file_path = os.path.join(INF_DIR, filename)
        print(f"Procesando: {filename}")

        waveform, original_sr = torchaudio.load(file_path)

        # Resampleo a TARGET_SR si es necesario
        if original_sr != TARGET_SR:
            if original_sr not in resamplers:
                resamplers[original_sr] = torchaudio.transforms.Resample(original_sr, TARGET_SR)
            waveform = resamplers[original_sr](waveform)

        # Pasar a mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalización por amplitud máxima
        max_val = waveform.abs().max().item() + 1e-8
        waveform_norm = waveform / max_val

        original_length = waveform_norm.size(1)
        input_for_plot = waveform_norm.clone()

        # STFT -> Normalizar -> Modelo -> Desnormalizar -> ISTFT
        stft_input = waveform_to_stft(waveform_norm.to(DEVICE))
        stft_input = normalize_stft(stft_input)  # Log-compresión (igual que en dataset)
        stft_padded, orig_f, orig_t = pad_stft(stft_input)
        stft_batch = stft_padded.unsqueeze(0)

        # Inferencia
        with torch.no_grad():
            predicted_stft = model(stft_batch)

        predicted_stft = predicted_stft.squeeze(0)[:, :orig_f, :orig_t]
        predicted_stft = denormalize_stft(predicted_stft)  # Inversa de log-compresión
        predicted_waveform = stft_to_waveform(predicted_stft)

        # Truncar a longitud original
        predicted_waveform = predicted_waveform[:, :original_length].cpu()

        # Guardar predicción normalizada para gráficos
        predicted_norm = predicted_waveform.clone()

        # Denormalizar y guardar audio
        predicted_waveform = (predicted_waveform * max_val).clamp(-1.0, 1.0)

        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(save_path, exist_ok=True)

        # Copiar el archivo de entrada original para referencia
        copy2(file_path, os.path.join(save_path, 'input.wav'))
        save_audio(predicted_waveform, os.path.join(save_path, 'super_res.wav'), TARGET_SR)

        # Guardar gráficos de forma de onda y espectrograma
        save_waveform_plot(
            input_for_plot,
            predicted_norm,
            os.path.join(save_path, 'waveform.png'),
            TARGET_SR,
        )

        save_spectrogram_plot(
            input_for_plot,
            predicted_norm,
            os.path.join(save_path, 'spectrogram.png'),
            sample_rate=TARGET_SR,
        )

        print(f"Guardado en {save_path.replace(os.sep, '/')}/")

    print(f"Listo. Revisa la carpeta '{OUTPUT_DIR.replace(os.sep, '/')}'.")


if __name__ == "__main__":
    try:
        inference()
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        DEVICE = 'cpu'
        print(f"Error durante la inferencia. Intentando en CPU...\n{e}")
        inference()
# Script para realizar inferencia
# Este script cargará el modelo entrenado y generará resultados

import os
import argparse
import torch
import torchaudio
import numpy as np
from shutil import copy2
import matplotlib.pyplot as plt
from src.model import UNetAudio2D

MODEL_PATH = 'unet2D_superres.pt'   # Archivo del modelo entrenado
INF_DIR = './data/inference'        # Archivos de entrada
OUTPUT_DIR = './results'            # Archivos de salida
TARGET_SR = 44100                   # Target sample rate
POOL_FACTOR = 16                    # 2^4 para 4 capas de pooling en UNet 2D
N_FFT = 2048                        # Tamaño de la FFT para STFT
HOP_LENGTH = 512                    # Salto entre ventanas STFT
FRAGMENT_LENGTH = 65536             # Longitud de los fragmentos en muestras

def get_device(force_device=None):
    """Elige el dispositivo donde se ejecutará el entrenamiento."""
    if force_device:
        return force_device
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except:
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

def parse_args():
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Inferencia de super-resolución de audio con UNet 2D',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Directorios
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Ruta al archivo del modelo entrenado (.pt)')
    parser.add_argument('--input', type=str, default=INF_DIR, help='Directorio con los archivos de audio de entrada para inferencia')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Directorio donde se guardarán los resultados')
    # Hiperparámetros
    parser.add_argument('--sample-rate', type=int, default=TARGET_SR, help='Frecuencia de muestreo objetivo (Hz)')
    parser.add_argument('--n-fft', type=int, default=N_FFT, help='Tamaño de la FFT para la STFT')
    parser.add_argument('--hop-length', type=int, default=HOP_LENGTH, help='Salto entre ventanas STFT')
    parser.add_argument('--fragment-length', type=int, default=FRAGMENT_LENGTH, help='Longitud de los fragmentos en muestras')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'directml'], help='Dispositivo de ejecución (auto-detecta si no se especifica)')
    parser.add_argument('--pool-factor', type=int, default=POOL_FACTOR, help='Factor de pooling (2^n capas de pooling)')
    return parser.parse_args()


def save_audio(tensor, path, sample_rate):
    """Guarda una forma de onda en un archivo de audio."""
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(path, tensor.cpu(), sample_rate, bits_per_sample=16, encoding="PCM_S")


def waveform_to_stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convierte una forma de onda a un STFT con canales real e imaginario.
    Args:
        waveform (torch.Tensor): Tensor de audio de entrada.
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Salto entre ventanas STFT.
    Returns:
        torch.Tensor: Tensor STFT con shape (2, F, T).
    """
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

def stft_to_waveform(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, length=None):
    """
    Convierte un STFT con canales real e imaginario a forma de onda usando ISTFT.
    Args:
        stft (torch.Tensor): Tensor STFT con shape (2, F, T).
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Salto entre ventanas STFT.
        length (int | None): Longitud exacta de la waveform de salida.
    Returns:
        torch.Tensor: Waveform reconstruida con shape (1, T).
    """
    stft_cpu = stft.cpu()

    stft_complex = torch.complex(stft_cpu[0], stft_cpu[1])
    window = torch.hann_window(n_fft, device='cpu')
    waveform = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=length,  # Forzar longitud exacta para evitar desfases en el corte
    )
    return waveform.unsqueeze(0)  # (1, T)


def normalize_stft(stft):
    """
    Aplica log-compresión a la magnitud de la STFT preservando la fase.
    Args:
        stft (torch.Tensor): Tensor STFT con shape (2, F, T).
    Returns:
        torch.Tensor: Tensor STFT comprimido con misma forma.
    """
    real, imag = stft[0], stft[1]
    magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase_cos = real / magnitude
    phase_sin = imag / magnitude
    mag_compressed = torch.log1p(magnitude)
    return torch.stack([mag_compressed * phase_cos, mag_compressed * phase_sin], dim=0)

def denormalize_stft(stft):
    """
    Aplica la inversa de la log-compresión de la STFT.
    Args:
        stft (torch.Tensor): Tensor STFT comprimido con shape (2, F, T).
    Returns:
        torch.Tensor: Tensor STFT en escala lineal con misma forma.
    """
    real, imag = stft[0], stft[1]
    mag_compressed = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase_cos = real / mag_compressed
    phase_sin = imag / mag_compressed
    magnitude = torch.expm1(mag_compressed)
    return torch.stack([magnitude * phase_cos, magnitude * phase_sin], dim=0)


def pad_stft(stft, pool_factor=POOL_FACTOR):
    """
    Ajusta el STFT para que sea divisible por pool_factor.
    Args:
        stft (torch.Tensor): Tensor STFT con shape (2, F, T).
        pool_factor (int): Factor de pooling.
    Returns:
        tuple[torch.Tensor, int, int]: STFT con padding, número original
            de bins de frecuencia y frames temporales.
    """
    _, freq_bins, time_frames = stft.shape
    pad_f = (pool_factor - (freq_bins % pool_factor)) % pool_factor
    pad_t = (pool_factor - (time_frames % pool_factor)) % pool_factor

    if pad_f > 0 or pad_t > 0:
        stft = torch.nn.functional.pad(stft, (0, pad_t, 0, pad_f), mode='reflect')

    return stft, freq_bins, time_frames


def save_waveform_plot(lr, sr, filename, sample_rate):
    """
    Genera y guarda un gráfico de la forma de onda de entrada y salida.
    Args:
        lr (torch.Tensor): Waveform de baja resolución.
        sr (torch.Tensor): Waveform de super resolución.
        filename (str): Ruta del archivo de salida.
        sample_rate (int): Frecuencia de muestreo.
    """
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
    """
    Genera y guarda un gráfico del espectrograma de entrada y salida.
    Args:
        lr_waveform (torch.Tensor): Waveform de baja resolución.
        sr_waveform (torch.Tensor): Waveform de super resolución.
        filename (str): Ruta del archivo de salida.
        sample_rate (int): Frecuencia de muestreo.
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Salto entre ventanas STFT.
    """
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


def process_audio_in_chunks(model, stft, orig_f, orig_t, chunk_frames, overlap=64):
    """
    Procesa el STFT por chunks con overlap para evitar artefactos en los bordes.
    Args:
        model (nn.Module): Modelo de super resolución.
        stft (torch.Tensor): STFT de entrada con shape (2, F, T).
        orig_f (int): Número original de bins de frecuencia (antes de padding).
        orig_t (int): Número original de frames temporales (antes de padding).
        chunk_frames (int): Tamaño de cada chunk en frames.
        overlap (int): Número de frames de solapamiento entre chunks.
    Returns:
        torch.Tensor: STFT procesado con shape (2, F, T).
    """
    _, F, T = stft.shape
    hop = chunk_frames - overlap
    output = torch.zeros_like(stft)
    weights = torch.zeros(T, device=stft.device)

    # Ventana trapezoidal
    window = torch.ones(chunk_frames, device=stft.device)
    if overlap > 0:
        window[:overlap] = torch.linspace(0, 1, overlap, device=stft.device)
        window[-overlap:] = torch.linspace(1, 0, overlap, device=stft.device)

    start = 0
    while start < T:
        end = min(start + chunk_frames, T)
        chunk = stft[:, :, start:end]

        # Pad si el chunk es más corto que chunk_frames
        pad_t = chunk_frames - chunk.shape[-1]
        if pad_t > 0:
            pad_mode = 'reflect' if pad_t < chunk.shape[-1] else 'replicate' # En el caso de que sea mayor que 0, se usa replicate (ultimo chunk)
            chunk = torch.nn.functional.pad(chunk, (0, pad_t), mode=pad_mode)

        with torch.no_grad():
            pred_chunk = model(chunk.unsqueeze(0)).squeeze(0)

        actual_len = end - start
        
        w = window[:actual_len].clone()
        # Evitar fade-in en el primer bloque de audio
        if start == 0 and overlap > 0:
             w[:overlap] = 1.0
        # Evitar fade-out en el último bloque de audio
        if end == T and overlap > 0:
             w[-overlap:] = 1.0

        output[:, :, start:end] += pred_chunk[:, :, :actual_len] * w
        weights[start:end] += w

        start += hop

    # Normalizar por los pesos acumulados
    output = output / weights.unsqueeze(0).unsqueeze(0)
    return output


def inference(args):
    """Realiza inferencia en los archivos de entrada y guarda los resultados."""
    device = get_device(args.device)
    n_fft = args.n_fft
    hop_length = args.hop_length
    target_sr = args.sample_rate
    pool_factor = args.pool_factor
    fragment_length = args.fragment_length
    os.makedirs(args.output, exist_ok=True)

    # Cargar modelo
    print(f"Cargando modelo desde {args.model}...")
    model = UNetAudio2D().to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))   # weights_only=False para compatibilidad con DirectML
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo '{args.model}'.")
        return
    model.eval()

    # Procesar archivos de inferencia
    files = [f for f in os.listdir(args.input) if f.endswith('.wav')]
    if not files:
        print(f"No se encontraron archivos .wav en {args.input}")
        return

    print(f"Encontrados {len(files)} archivos para procesar.")

    resamplers = {}

    for filename in files:
        file_path = os.path.join(args.input, filename)
        print(f"Procesando: {filename}")

        waveform, original_sr = torchaudio.load(file_path)

        # Resampleo a target_sr si es necesario
        if original_sr != target_sr:
            if original_sr not in resamplers:
                resamplers[original_sr] = torchaudio.transforms.Resample(original_sr, target_sr)
            waveform = resamplers[original_sr](waveform)

        # Pasar a mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalización por amplitud máxima
        max_val = waveform.abs().max().item() + 1e-8
        waveform_norm = waveform / max_val

        original_length = waveform_norm.size(1)
        input_for_plot = waveform_norm.clone()

        # Preparación STFT
        stft_input = waveform_to_stft(waveform_norm.to(device), n_fft=n_fft, hop_length=hop_length)
        stft_input = normalize_stft(stft_input)  # Log-compresión (igual que en dataset)
        stft_padded, orig_f, orig_t = pad_stft(stft_input, pool_factor=pool_factor)

        # frames equivalentes a FRAGMENT_LENGTH muestras
        chunk_frames = fragment_length // hop_length
        # Asegurar que chunk_frames sea múltiplo del factor de pooling
        chunk_frames = chunk_frames + (pool_factor - (chunk_frames % pool_factor)) % pool_factor

        predicted_stft = process_audio_in_chunks(model, stft_padded, orig_f, orig_t, chunk_frames=chunk_frames)
        predicted_stft = predicted_stft[:, :orig_f, :orig_t]

        predicted_stft = denormalize_stft(predicted_stft)  # Inversa de log-compresión

        # ISTFT con longitud exacta para evitar artefactos de corte
        predicted_waveform = stft_to_waveform(predicted_stft, n_fft=n_fft, hop_length=hop_length, length=original_length)   # (1, T)

        # Prevención de valores NaN o infinitos
        predicted_waveform = torch.nan_to_num(predicted_waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        predicted_waveform = torch.clamp(predicted_waveform, -1.0, 1.0)

        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(args.output, base_name)
        os.makedirs(save_path, exist_ok=True)

        # Guardar resultados
        copy2(file_path, os.path.join(save_path, 'input.wav'))
        save_audio(predicted_waveform, os.path.join(save_path, 'super_res.wav'), target_sr)

        # Guardar gráficos de forma de onda y espectrograma
        save_waveform_plot(
            input_for_plot,
            predicted_waveform,
            os.path.join(save_path, 'waveform.png'),
            target_sr
        )

        save_spectrogram_plot(
            input_for_plot,
            predicted_waveform,
            os.path.join(save_path, 'spectrogram.png'),
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        print(f"Guardado en {save_path.replace(os.sep, '/')}/")

    print(f"Listo. Revisa la carpeta '{args.output.replace(os.sep, '/')}'.")


if __name__ == "__main__":
    args = parse_args()
    try:
        inference(args)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        args.device = 'cpu'
        print(f"Error durante la inferencia. Intentando en CPU...\n{e}")
        inference(args)
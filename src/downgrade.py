import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample

fc = 4000  # frecuencia de corte en Hz
fs = fc*2

for i in range(1, 12):
    fs_original, audio = wav.read(f"train{i}.wav")

    # Pasar a float para procesado
    audio = audio.astype(np.float64)

    N = len(audio)

    # FFT
    X = np.fft.fft(audio)

    # Vector de frecuencias
    freqs = np.fft.fftfreq(N, d=1/fs_original)

    # Máscara paso bajo
    mask = np.abs(freqs) <= fc

    # Aplicar máscara
    X_filtered = X * mask

    # IFFT (señal real)
    audio_filtered = np.fft.ifft(X_filtered).real

    # Resampliar a fs_nuevo = fc * 2
    num_samples_nuevo = int(len(audio_filtered) * fs / fs_original)
    audio_resampled = resample(audio_filtered, num_samples_nuevo)

    # Normalizar y volver a int16
    audio_resampled = np.clip(audio_resampled, -32768, 32767)
    audio_resampled = audio_resampled.astype(np.int16)

    # Guardar con la nueva frecuencia de muestreo
    wav.write(f"D:\\PruebasTFG\\data\\train\\LR\\train{i}.wav", fs, audio_resampled)

    print(f"train{i}.wav → filtrado a {fc} Hz y resampled a {fs} Hz")
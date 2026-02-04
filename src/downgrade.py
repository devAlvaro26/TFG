import numpy as np
import scipy.io.wavfile as wav

fc = 4500  # frecuencia de corte en Hz

for i in range(1, 12):
    fs, audio = wav.read(f"train{i}-lr.wav")

    # Pasar a float para procesado
    audio = audio.astype(np.float64)

    N = len(audio)

    # FFT
    X = np.fft.fft(audio)

    # Vector de frecuencias
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Máscara paso bajo
    mask = np.abs(freqs) <= fc

    # Aplicar máscara
    X_filtered = X * mask

    # IFFT (señal real)
    audio_filtered = np.fft.ifft(X_filtered).real

    # Normalizar y volver a int16
    audio_filtered = np.clip(audio_filtered, -32768, 32767)
    audio_filtered = audio_filtered.astype(np.int16)

    # Guardar
    wav.write(f"train{i}.wav", fs, audio_filtered)

    print(f"train{i}.wav → filtrado a {fc} Hz")
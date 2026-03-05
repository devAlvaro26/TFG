# Superresolución de Audio con UNet 2D

Este proyecto implementa un modelo de Deep Learning basado en una arquitectura UNet 2D que opera sobre representaciones STFT (Short-Time Fourier Transform) para realizar Superresolución de Audio. El objetivo es reconstruir el contenido de alta frecuencia a partir de entradas de audio de baja resolución, mejorando cualquier archivo de audio de baja resolución a 44.1kHz.

## Características

*   **Arquitectura UNet 2D**: Aplicada al procesamiento de la magnitud real e imaginaria del STFT.
*   **Superresolución**: Escala el audio desde frecuencias de muestreo más bajas a un objetivo de 44.1kHz.
*   **Pérdida Multi-objetivo**: Utiliza `STFTMagnitudeLoss` (Spectral Convergence + Log-Magnitude L1 + Complex MSE) para una reconstrucción mejorada.
*   **Aprendizaje Residual**: El modelo aprende a predecir el contenido faltante (residuo) que se suma a la entrada de baja resolución.
*   **Inferencia y Visualización**:
    - Genera archivos de audio super-resueltos.
    - Produce gráficos comparativos de forma de onda (Entrada vs. Salida).
    - Produce gráficos comparativos de espectrograma para visualizar la reconstrucción de frecuencias.

## Estructura del Proyecto

```
.
├── data/                   # Directorio para dataset y archivos de prueba
│   ├── train/              # Datos de entrenamiento
│   │   ├── HR/             # Audio de Alta Resolución (Ground Truth)
│   │   └── LR/             # Audio de Baja Resolución (Input)
│   └── test/               # Archivos de audio de prueba para inferencia
├── results/                # Directorio de salida para resultados de inferencia
├── src/                    # Módulos de código fuente
│   ├── dataset.py          # Clase Dataset: Carga audio y lo convierte a STFT (Real/Imag)
│   ├── model.py            # Definición del modelo UNetAudio2D
│   ├── loss.py             # Función de pérdida STFT multi-objetivo
│   └── downgrade.py        # Script para degradar el audio (para generar LR)
|
├── inference.py            # Script para ejecutar inferencia en datos de prueba
├── train.py                # Script para entrenar el modelo
├── requirements.txt        # Dependencias de Python
└── unet2D_superres.pth     # Checkpoint del modelo entrenado
```

## Instalación

1.  Clonar el repositorio.
2.  Instalar las dependencias requeridas:

    ```bash
    pip install -r requirements.txt
    ```
3. Opcional: Instalar torch-directml para GPU de AMD:

    ```bash
    pip install torch-directml==0.2.5.dev240914
    ```

## Uso

### 1. Entrenamiento

Para entrenar el modelo, es necesario un dataset de pares de archivos de audio de Alta Resolución (HR) y Baja Resolución (LR).

1.  Colocar los archivos wav de **Alta Resolución** en `./data/train/HR/`.
2.  Colocar los archivos wav correspondientes de **Baja Resolución** en `./data/train/LR/`.
    *   *Nota: Los nombres de archivo deben coincidir exactamente entre las carpetas HR y LR.*
3.  Ejecutar el script de entrenamiento:

    ```bash
    python train.py
    ```

El script entrenará el modelo y guardará el mejor checkpoint en `unet2D_superres.pth`. Utiliza un sistema de *Early Stopping* si la pérdida no mejora durante varias épocas.

### 2. Inferencia

Para probar el modelo en nuevos archivos de audio:

1.  Colocar los archivos de entrada `.wav` en `./data/test/`.
2.  Ejecutar el script de inferencia:

    ```bash
    python inference.py
    ```

3.  Los resultados se guardarán en la carpeta `./results/`. Para cada archivo de entrada, se creará una subcarpeta que contiene:
    *   `input.wav`: La entrada original de baja resolución.
    *   `super_res.wav`: La salida super-resuelta del modelo.
    *   `waveform.png`: Una comparación visual de las formas de onda.
    *   `spectrogram.png`: Una comparación visual de los espectrogramas.

## Requisitos

*   Python 3.12.10
*   torch 2.5.1
*   torchaudio 2.5.1
*   torchvision 0.20.1
*   numpy
*   matplotlib
*   soundfile
*   scipy
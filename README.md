# Superresolución de Audio con UNet 2D

Este proyecto implementa un modelo de Deep Learning basado en una arquitectura UNet 2D que opera sobre representaciones STFT para realizar Superresolución de Audio. El objetivo es reconstruir el contenido de alta frecuencia a partir de entradas de audio de baja resolución, mejorando cualquier archivo de audio de baja resolución a 44.1kHz.

## Características

*   **Arquitectura UNet 2D**: Aplicada al procesamiento de la magnitud real e imaginaria del STFT incorporando Attention Gates.
*   **Superresolución**: Escala el audio desde frecuencias de muestreo más bajas a un objetivo de 44.1kHz.
*   **Pérdida Multi-objetivo**: Utiliza `CombinedLoss` (STFTMagnitudeLoss + MelSpectrogramLoss) para optimizar tanto la fidelidad espectral como la percepción auditiva mediante mel-spectrogramas.
*   **Aprendizaje Residual**: El modelo aprende a predecir el contenido faltante (residuo) y sumandolo a la entrada de baja resolución.
*   **Inferencia y Visualización**:
    - Genera archivos de audio super-resueltos.
    - Produce gráficos comparativos de forma de onda (Entrada vs. Salida).
    - Produce gráficos comparativos de espectrograma para visualizar la reconstrucción de frecuencias.

## Dataset

El modelo entrenado se ha realizado con el dataset **[MUSDB18-HQ](https://zenodo.org/records/3338373)**

## Estructura del Proyecto

```
.
├── data/
│   ├── train/              # Dataset de entrenamiento (HR/LR)
│   ├── test/               # Dataset de validación (HR/LR)
│   └── inference/          # Archivos .wav para procesar con el modelo
├── results/                # Salida de la inferencia (audio + gráficos)
├── src/
│   ├── dataset.py          # Clase Dataset: Carga audio y lo convierte a STFT
│   ├── model.py            # Definición de UNetAudio2D + AttentionGate
│   ├── loss.py             # CombinedLoss (STFT + Mel-Spectrogram) con énfasis frecuencial
│   └── downgrade.py        # Herramienta para generar pares LR desde HR
├── train.py                # Script de entrenamiento con Scheduler y Early Stopping
├── inference.py            # Script para ejecución y visualización de resultados
├── requirements.txt        # Dependencias del proyecto
└── unet2D_superres.pth     # Checkpoint del mejor modelo guardado
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
3. Colocar los archivos de validación en `./data/test/HR/` y `./data/test/LR/` de igual forma.
4.  Ejecutar el script de entrenamiento:

    ```bash
    python train.py
    ```

El script entrenará el modelo y guardará el mejor checkpoint en `unet2D_superres.pth` basado en la pérdida de validación. Utiliza un sistema de *Early Stopping* si la pérdida no mejora durante varias épocas.

### 2. Inferencia

Para probar el modelo en nuevos archivos de audio:

1.  Colocar los archivos de entrada `.wav` en `./data/inference/`.
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
*   torch 2.4.1
*   torchaudio 2.4.1
*   torchvision 0.19.1
*   numpy
*   matplotlib
*   soundfile
*   scipy
# Superresolución de Audio con UNet

Este proyecto implementa un modelo de Deep Learning basado en una arquitectura UNet 1D para realizar Superresolución de Audio. El objetivo es reconstruir el contenido de alta frecuencia a partir de entradas de audio de baja resolución, mejorando efectivamente la conversión de un muestreo bajo a un objetivo de 44.1kHz.

## Características

*   **Arquitectura UNet 1D**: Personalizada para el procesamiento eficiente de formas de onda de audio.
*   **Superresolución**: Escala el audio desde frecuencias de muestreo más bajas a un objetivo de 44.1kHz.
*   **Manejo de Datos**: Gestiona longitudes de audio arbitrarias mediante relleno o recorte aleatorio durante el entrenamiento.
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
│   ├── dataset.py          # Clase Dataset personalizada para cargar pares de audio
│   ├── model.py            # Definición del modelo UNetAudio
│   └── ...
├── inference.py            # Script para ejecutar inferencia en datos de prueba
├── train.py                # Script para entrenar el modelo
├── requirements.txt        # Dependencias de Python
└── unet_superres.pth       # Checkpoint del modelo entrenado
```

## Instalación

1.  Clona el repositorio.
2.  Instala las dependencias requeridas:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

### 1. Entrenamiento

Para entrenar el modelo, necesitas un dataset de pares de archivos de audio de Alta Resolución (HR) y Baja Resolución (LR).

1.  Coloca los archivos wav de **Alta Resolución** en `./data/train/HR/`.
2.  Coloca los archivos wav correspondientes de **Baja Resolución** en `./data/train/LR/`.
    *   *Nota: Los nombres de archivo deben coincidir exactamente entre las carpetas HR y LR.*
3.  Ejecuta el script de entrenamiento:

    ```bash
    python train.py
    ```

El script entrenará el modelo por un número especificado de épocas y guardará el mejor modelo (basado en la pérdida) en `unet_superres.pth`.

### 2. Inferencia

Para probar el modelo en nuevos archivos de audio:

1.  Coloca tus archivos de entrada `.wav` en `./data/test/`.
2.  Ejecuta el script de inferencia:

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
*   torch 2.9.1
*   torchaudio 2.9.1
*   numpy
*   matplotlib
*   soundfile

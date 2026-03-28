# SuperResolución de Audio con Attention Res-UNet 2D

Este proyecto implementa un modelo de Deep Learning basado en una arquitectura UNet 2D que opera sobre representaciones STFT para realizar Superresolución de Audio. El objetivo es reconstruir el contenido de alta frecuencia a partir de entradas de audio de baja resolución, mejorando cualquier archivo de audio de baja resolución a una frecuencia de muestreo de 44.1kHz.

## Características

*   **Arquitectura Attention Res-UNet 2D**: Aplicada mediante procesamiento Complex as Channels (CaC) para separar magnitud y fase, se implementan `Attention Gates` y `Residual Blocks` sobre la U-Net para mejorar la calidad de la reconstrucción.
*   **Superresolución**: Escala el audio desde frecuencias de muestreo más bajas a un objetivo de 44.1kHz.
*   **Arquitectura GAN**: El proyecto utiliza una arquitectura GAN inspirada en HiFi-GAN y AERO, mediante discriminadores MPD y MSD.
*   **Métrica de Pérdida**: El entrenamiento utiliza una combinación de pérdidas L1, MR-STFT y GAN.
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
│   ├── dataset.py          # Clase Dataset: Carga audio y lo convierte a pares STFT
│   ├── model.py            # Definición de UNetAudio2D (Attention Res-UNet 2D)
│   ├── discriminator.py    # Definición del Discriminador MSD y MPD
│   ├── loss.py             # CombinedLoss (L1, MR-STFT) y DiscriminatorLoss
│   └── downgrade.py        # Herramienta para generar pares LR desde HR
├── jupyter/
│   ├── inference.ipynb     # Inferencia adaptada a cuaderno para ejecutar en GPU
│   └── train.ipynb         # Entrenamiento adaptado a cuaderno para ejecutar en GPU
├── train.py                # Script de entrenamiento del modelo
├── inference.py            # Script para inferencia y visualización de resultados
├── requirements.txt        # Dependencias del proyecto
└── unet2D_superres.pt      # Checkpoint del mejor modelo guardado
```

## Modelos entrenados

*   [Repositorio modelo pre-entrenado](https://drive.google.com/file/d/1dxCMkGfHNDsXdcmxfNl1TRpjgO7KC5Gf/view?usp=sharing)

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

El script entrenará el modelo y guardará el mejor checkpoint en `unet2D_superres.pt` basado en la pérdida de validación. Utiliza un sistema de *Early Stopping* si la pérdida no mejora durante varias épocas.

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

### Uso Jupyter

Los cuadernos jupyter son versiones adaptadas para computación en gpu directamente en jupyter notebook:
* `train.ipynb`: Entrenamiento del modelo.
* `inference.ipynb`: Inferencia del modelo.

## Papers y proyectos de referencia

*   [Audio Super Resolution using Neural Networks](https://arxiv.org/abs/1708.00853) ([GitHub](https://github.com/kuleshov/audio-super-res))
*   [AERO: Audio Super Resolution in the Spectral Domain](https://arxiv.org/abs/2211.12232) ([GitHub](https://github.com/slp-rl/aero))
*   [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) ([GitHub](https://github.com/jik876/hifi-gan))
*   [Versatile_Audio_Super_Resolution](https://github.com/haoheliu/versatile_audio_super_resolution)


## Desarrollado con los siguientes paquetes

*   Python 3.12.10
*   torch 2.4.1
*   torchaudio 2.4.1
*   torchvision 0.19.1
*   torchmetrics 1.9.0
*   pystoi
*   numpy
*   matplotlib
*   soundfile
*   scipy
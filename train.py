# Script para entrenar la red neuronal
# Este script guardara el mejor modelo en un archivo .pt

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.model import UNetAudio2D
from src.dataset import AudioSuperResDataset
from src.discriminator import CombinedDiscriminator
from src.loss import CombinedLoss, DiscriminatorLoss, LossMetrics

TRAIN_HR_DIR = './data/train/HR'    # Archivos de alta resolución (ground truth)
TRAIN_LR_DIR = './data/train/LR'    # Archivos de baja resolución (input)
VAL_HR_DIR = './data/test/HR'       # Archivos de alta resolución para validación
VAL_LR_DIR = './data/test/LR'       # Archivos de baja resolución para validación

BATCH_SIZE = 4                      # Tamaño de lote
EPOCHS = 500                        # Épocas
LEARNING_RATE_G = 2e-4              # LR del generador
LEARNING_RATE_D = 1e-4              # LR del discriminador

try:
    import torch_directml
    has_dml = torch_directml.is_available()
except ImportError:
    has_dml = False

if has_dml:
    DEVICE = torch_directml.device()
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def evaluate(model_g, dataloader, criterion):
    """Evalúa el modelo en el conjunto de validación y devuelve la pérdida promedio, SI-SDR, STOI y LSD."""
    model_g.eval()
    total_loss = 0.0
    total_sisdr = 0.0
    total_stoi = 0.0
    total_lsd = 0.0
    with torch.no_grad():
        # Validar el modelo sin gradientes
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model_g(inputs)   # Predicción del modelo
            loss = criterion(outputs, targets)  # Pérdida del modelo
            total_loss += loss.item()
            pred_wav, target_wav = criterion.get_waveforms(outputs, targets) # Obtener waveforms
            total_sisdr += LossMetrics.sisdr_loss(pred_wav, target_wav) # SI-SDR
            total_stoi += LossMetrics.stoi_loss(pred_wav, target_wav) # STOI
            total_lsd += LossMetrics.lsd_loss(pred_wav, target_wav) # LSD
    return total_loss / len(dataloader), total_sisdr / len(dataloader), total_stoi / len(dataloader), total_lsd / len(dataloader)

def train():
    """Entrenar el modelo"""
    # Verificar que los directorios existen
    dirs_to_check = [TRAIN_HR_DIR, TRAIN_LR_DIR, VAL_HR_DIR, VAL_LR_DIR]
    for d in dirs_to_check:
        if not os.path.exists(d):
            print(f"Error: directorio no encontrado: {d}")
            return

    # Cargar datasets
    train_dataset = AudioSuperResDataset(TRAIN_HR_DIR, TRAIN_LR_DIR)
    val_dataset = AudioSuperResDataset(VAL_HR_DIR, VAL_LR_DIR)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: El dataset está vacío")
        return

    num_workers = max(0, os.cpu_count()-1)
    # len(dataset) -> número de fragmentos del conjunto
    # Shape: (2, F, T) -> 2=componentes real/imag, F=frecuencia, T=tiempo
    print(f"Cargados {len(train_dataset)} elementos de entrenamiento con shapes de entrada y salida: {train_dataset[0][0].shape} y {train_dataset[0][1].shape}")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Cargados {len(val_dataset)} elementos de validación con shapes de entrada y salida: {val_dataset[0][0].shape} y {val_dataset[0][1].shape}")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Inicializar Modelos
    model_g = UNetAudio2D().to(DEVICE)
    model_d = CombinedDiscriminator().to(DEVICE)

    # Inicializar Pérdidas
    criterion_g = CombinedLoss(lambda_l1=1.0, lambda_mrstft=1.0).to(DEVICE)
    criterion_d = DiscriminatorLoss(lambda_adv=1.0, lambda_fm=2.0, lambda_mel=45.0).to(DEVICE)

    # Inicializar Optimizadores
    optimizer_g = optim.AdamW(model_g.parameters(), lr=LEARNING_RATE_G, betas=(0.8, 0.99))
    optimizer_d = optim.AdamW(model_d.parameters(), lr=LEARNING_RATE_D, betas=(0.8, 0.99))

    # Schedulers
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=10)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=10)

    # Inicializar TensorBoard
    writer = SummaryWriter(log_dir='./runs')

    print(f"Iniciando entrenamiento en {DEVICE}...")

    # Early stopping
    best_val_loss = float('inf')
    patience_earlystop = 50
    epochs_no_improve = 0

    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        # Entrenar modelos
        model_g.train()
        model_d.train()

        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Entrenar generador
            pred = model_g(inputs)

            # Obtener waveforms de LR y HR
            pred_wav, target_wav = criterion_g.get_waveforms(pred, targets)

            # Entrenar discriminador
            optimizer_d.zero_grad(set_to_none=True)
            y_d_rs, y_d_gs, _, _ = model_d(target_wav.detach(), pred_wav.detach())
            loss_d = criterion_d.discriminator_loss(y_d_rs, y_d_gs)

            # Backward
            loss_d.backward()

            # Gradient clipping para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), max_norm=1.0)

            # Actualizar los pesos del modelo
            optimizer_d.step()

            # Acumular pérdidas
            running_loss_d += loss_d.item()

            optimizer_g.zero_grad(set_to_none=True)
            
            # Perdidas del generador
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = model_d(target_wav.detach(), pred_wav)
            loss_adv = criterion_d.generator_loss(y_d_gs)
            loss_fm = criterion_d.feature_matching_loss(fmap_rs, fmap_gs)
            loss_mel = criterion_d.mel_spectrogram_loss(pred_wav, target_wav.detach())
            loss_g = loss_adv + loss_fm + loss_mel
            
            # Backward
            loss_g.backward()
            
            # Gradient clipping para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
            
            # Actualizar los pesos del modelo
            optimizer_g.step()

            # Acumular pérdidas
            running_loss_g += loss_g.item()
        
        # Métricas
        train_loss_g = running_loss_g / len(train_dataloader)
        train_loss_d = running_loss_d / len(train_dataloader)

        # Evaluar generador en el conjunto de validación
        val_loss, val_sisdr, val_stoi, val_lsd = evaluate(model_g, val_dataloader, criterion_g)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss G: {train_loss_g:.6f} | Loss D: {train_loss_d:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer_g.param_groups[0]['lr']:.8f}")
        print(f"Métricas de calidad: SI-SDR: {val_sisdr:.6f} | STOI: {val_stoi:.6f} | LSD: {val_lsd:.6f}")

        # Registrar métricas en TensorBoard
        writer.add_scalars('Loss/train', {'generator': train_loss_g, 'discriminator': train_loss_d}, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/SI-SDR', val_sisdr, epoch)
        writer.add_scalar('Metrics/STOI', val_stoi, epoch)
        writer.add_scalar('Metrics/LSD', val_lsd, epoch)
        writer.add_scalar('LearningRate/generator', optimizer_g.param_groups[0]['lr'], epoch)
        writer.add_scalar('LearningRate/discriminator', optimizer_d.param_groups[0]['lr'], epoch)

        # Actualizar schedulers
        scheduler_g.step(val_loss)
        scheduler_d.step(train_loss_d)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model_g.state_dict(), 'unet2D_superres_best.pt')
            print(f"Mejor modelo guardado con loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_earlystop:
                print(f"Early stopping en epoch {epoch+1}")
                break
        
        torch.save(model_g.state_dict(), 'unet2D_superres.pt')

    writer.close()
    print("Entrenamiento completado")
    print(f"Mejor loss alcanzado: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
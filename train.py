# Script para entrenar la red neuronal
# Este script guardara el mejor modelo en un archivo .pth

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import AudioSuperResDataset
from src.model import UNetAudio2D
from src.loss import STFTMagnitudeLoss

TRAIN_HR_DIR = 'D:/Audio/train/HR'  # Archivos de alta resolución (output de la red)
TRAIN_LR_DIR = 'D:/Audio/train/LR'  # Archivos de baja resolución (input de la red)
VAL_HR_DIR = 'D:/Audio/test/HR'     # Archivos de alta resolución para validación
VAL_LR_DIR = 'D:/Audio/test/LR'     # Archivos de baja resolución para validación
BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 2e-4

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

def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo en el conjunto de validación y devuelve la pérdida promedio."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train():
    
    if not os.path.exists(TRAIN_HR_DIR) or not os.path.exists(TRAIN_LR_DIR):
        print(f"Error: Por favor, proporcione rutas válidas para {TRAIN_HR_DIR} y {TRAIN_LR_DIR}")
        return

    # Cargar dataset
    train_dataset = AudioSuperResDataset(TRAIN_HR_DIR, TRAIN_LR_DIR)
    val_dataset = AudioSuperResDataset(VAL_HR_DIR, VAL_LR_DIR)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: El dataset está vacío")
        return

    print(f"Dataset de entrenamiento cargado: {len(train_dataset)} archivos")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
    print(f"Dataset de validación cargado: {len(val_dataset)} archivos")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

    # Inicializar modelo
    model = UNetAudio2D().to(DEVICE)

    # Inicializar Loss y Optimizer
    # STFTMagnitudeLoss (Convergencia espectral + Log-Magnitude L1 + MSE complejo)
    # Optimizer Adam
    criterion = STFTMagnitudeLoss(alpha=1.0, beta=1.0, gamma=0.5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999)) 

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    print(f"Iniciando entrenamiento en {DEVICE}...")

    best_val_loss = float('inf')
    patience_earlystop = 50
    epochs_no_improve = 0

    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Actualizar los pesos del modelo
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)

        # Evaluar en el conjunto de validación
        val_loss = evaluate(model, val_dataloader, criterion, DEVICE)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Scheduler basado en val_loss
        scheduler.step(val_loss)

        # Guardar mejor modelo según val_loss
        if epoch >= 5:  # Solo considerar guardar el modelo después de algunas épocas para evitar guardar modelos muy malos al inicio
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'unet2D_superres.pth')
                print(f"Mejor modelo guardado con loss: {best_val_loss:.6f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience_earlystop:
                    print(f"Early stopping en epoch {epoch+1}")
                    break

    print("Entrenamiento completado")
    print(f"Mejor loss alcanzado: {best_val_loss:.6f}")

if __name__ == "__main__":
    train()
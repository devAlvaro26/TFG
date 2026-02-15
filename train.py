# Script para entrenar la red neuronal
# Este script guardara el mejor modelo en un archivo .pth

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import AudioSuperResDataset
from src.model import UNetAudio

HR_DIR = './data/train/HR'
LR_DIR = './data/train/LR'
BATCH_SIZE = 2  
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    # Preparar datos
    if not os.path.exists(HR_DIR) or not os.path.exists(LR_DIR):
        print(f"Error: Por favor, proporcione rutas válidas para {HR_DIR} y {LR_DIR}")
        return

    dataset = AudioSuperResDataset(HR_DIR, LR_DIR)
    if len(dataset) == 0:
        print("Error: El dataset está vacío")
        return
    
    print(f"Dataset cargado: {len(dataset)} archivos")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Inicializar modelo
    model = UNetAudio().to(DEVICE)
    
    # Inicializar Loss y Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    print(f"Iniciando entrenamiento en {DEVICE}...")

    best_loss = float('inf')
    
    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para evitar gradientes muy grandes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Actualizar scheduler
        scheduler.step(epoch_loss)
        
        # Guardar mejor modelo
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'unet_superres.pth')
            print(f"Mejor modelo guardado con loss: {best_loss:.6f}")

    print("Entrenamiento completado")
    print(f"Mejor loss alcanzado: {best_loss:.6f}")

if __name__ == "__main__":
    train()
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import AudioSuperResDataset
from src.model import UNetAudio

HR_DIR = 'D:/PruebasTFG/data/train/HR'
LR_DIR = 'D:/PruebasTFG/data/train/LR'
BATCH_SIZE = 2  
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    # Prepare Data
    if not os.path.exists(HR_DIR) or not os.path.exists(LR_DIR):
        print(f"Error: Please provide valid paths for {HR_DIR} and {LR_DIR}")
        return

    dataset = AudioSuperResDataset(HR_DIR, LR_DIR)
    if len(dataset) == 0:
        print("Error: The dataset is empty")
        return
    
    print(f"Dataset loaded: {len(dataset)} files")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = UNetAudio().to(DEVICE)
    
    # Initialize Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    print(f"Starting training on {DEVICE}...")

    best_loss = float('inf')
    
    # Training Loop
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
            
            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update scheduler
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'unet_superres.pth')
            print(f"Best model saved with loss: {best_loss:.6f}")

    print("Training completed!")
    print(f"Best loss reached: {best_loss:.6f}")

if __name__ == "__main__":
    train()
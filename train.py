import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import VideoDeepfakeCNN, AudioDeepfakeCNN, MultimodalDeepfakeModel

def train_model(model_type, metadata_csv, epochs=10, batch_size=32, lr=0.001, device='cpu'):
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # 1. Initialize DataLoader
    print(f"Loading data from {metadata_csv}...")
    try:
        train_loader, val_loader = get_dataloaders(metadata_csv, batch_size=batch_size)
    except Exception as e:
        print(f"Error loading dataloaders: {e}")
        print("Please ensure run_preprocessing.py has run successfully and generated the CSV.")
        return
        
    # 2. Select Model
    print(f"Initializing {model_type} model...")
    if model_type == "video":
        model = VideoDeepfakeCNN(pretrained=True, feature_extraction=False)
    elif model_type == "audio":
        model = AudioDeepfakeCNN(feature_extraction=False)
    elif model_type == "multimodal":
        model = MultimodalDeepfakeModel()
    else:
        raise ValueError("Invalid model type. Choose from: video, audio, multimodal")
        
    model.to(device)
    
    # 3. Loss & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train_samples = 0
        
        for faces, audios, labels in train_loader:
            faces, audios, labels = faces.to(device), audios.to(device), labels.to(device)
            labels = labels.unsqueeze(1) # shape: (batch_size, 1)
            
            optimizer.zero_grad()
            
            # Forward pass depending on model type
            if model_type == "video":
                outputs = model(faces)
            elif model_type == "audio":
                outputs = model(audios)
            elif model_type == "multimodal":
                outputs = model(faces, audios)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * faces.size(0)
            
            # Accuracy calculation
            preds = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train_samples += faces.size(0)
            
        epoch_train_loss = train_loss / total_train_samples
        epoch_train_acc = train_correct / total_train_samples
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for faces, audios, labels in val_loader:
                faces, audios, labels = faces.to(device), audios.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                if model_type == "video":
                    outputs = model(faces)
                elif model_type == "audio":
                    outputs = model(audios)
                elif model_type == "multimodal":
                    outputs = model(faces, audios)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * faces.size(0)
                
                preds = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val_samples += faces.size(0)
                
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = val_correct / total_val_samples
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")
              
        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            save_path = f"models/{model_type}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint to {save_path}")
            
    print("\nTraining completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Models")
    parser.add_argument("--model", type=str, default="multimodal", choices=["video", "audio", "multimodal"],
                        help="Model architecture to train")
    parser.add_argument("--csv", type=str, default="data/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(
        model_type=args.model,
        metadata_csv=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for Multimodal Deepfake Detection.
    Pairs video face crops with corresponding audio Mel Spectrograms.
    """
    def __init__(self, metadata_csv, transform=None, max_audio_len=300):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file containing file paths and labels.
                                Columns expected: ['face_path', 'audio_mel_path', 'label']
            transform (callable, optional): Optional transform to be applied on a face crop Image.
            max_audio_len (int): Maximum length of audio spectrogram time steps to pad/truncate to.
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        self.max_audio_len = max_audio_len
        
        # Standard transforms for ResNet face crops if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load Face Crop Image
        face_path = self.metadata.iloc[idx]['face_path']
        if not os.path.exists(face_path):
            raise FileNotFoundError(f"Face image not found: {face_path}")
        
        face_img = Image.open(face_path).convert('RGB')
        face_tensor = self.transform(face_img)

        # 2. Load Audio Mel Spectrogram
        mel_path = self.metadata.iloc[idx]['audio_mel_path']
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Mel spectrogram file not found: {mel_path}")
        
        # Load pre-saved numpy array
        mel_spec = np.load(mel_path) # Expected shape: (n_mels, time_steps)
        
        # Add channel dimension: (1, n_mels, time_steps)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        # 3. Handle Variable Length Audio (Pad or Truncate)
        channels, n_mels, time_steps = mel_spec.shape
        if time_steps < self.max_audio_len:
            # Pad with zeros
            pad_width = self.max_audio_len - time_steps
            mel_spec = np.pad(mel_spec, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate/Crop the middle segment
            start = (time_steps - self.max_audio_len) // 2
            mel_spec = mel_spec[:, :, start:start + self.max_audio_len]
            
        audio_tensor = torch.tensor(mel_spec, dtype=torch.float32)

        # 4. Get Label
        label = self.metadata.iloc[idx]['label']
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return face_tensor, audio_tensor, label_tensor

def get_dataloaders(metadata_csv, batch_size=32, shuffle=True, num_workers=0, max_audio_len=300, train_ratio=0.8):
    """
    Splits the dataset and returns Train and Validation DataLoaders.
    """
    # Define augmentations for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define standard validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read metadata to split
    df = pd.read_csv(metadata_csv)
    
    # Simple train-validation split (grouped by video if needed, but here simple split)
    # Shuffling first
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Save temporary CSVs for dataset initialization
    train_csv_path = metadata_csv.replace('.csv', '_train.csv')
    val_csv_path = metadata_csv.replace('.csv', '_val.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    # Create Datasets
    train_dataset = DeepfakeDataset(train_csv_path, transform=train_transform, max_audio_len=max_audio_len)
    val_dataset = DeepfakeDataset(val_csv_path, transform=val_transform, max_audio_len=max_audio_len)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoaders created. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader

if __name__ == "__main__":
    print("Dataset module initialized.")
    # Example usage:
    # try:
    #     train_loader, val_loader = get_dataloaders("data/metadata.csv", batch_size=16)
    #     for faces, audios, labels in train_loader:
    #         print("Faces batch shape:", faces.shape)
    #         print("Audios batch shape:", audios.shape)
    #         print("Labels batch shape:", labels.shape)
    #         break
    # except Exception as e:
    #     print("Could not test loader:", e)

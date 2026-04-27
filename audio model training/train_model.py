import os
import glob
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(r"c:\Users\94771\Desktop\ChildSafety-Backend-API\audio model training")
TRAINING_DIR = BASE_DIR / "Training_Data"
SAFE_DIR = TRAINING_DIR / "Class_0_Safe"
THREAT_DIR = TRAINING_DIR / "Class_1_Threat"
MODEL_SAVE_PATH = BASE_DIR / "audio_threat_model.pth"

# Hyperparameters
MAX_LEN = 150  # Fixed sequence length for MFCC
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 10     # Keep it small for prototyping
LEARNING_RATE = 0.001

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio and extract MFCC
        try:
            # Load with standard sample rate
            y, sr = librosa.load(file_path, sr=22050, duration=3.0) 
            
            # Extract MFCC
            # mfcc shape: (n_mfcc, time_steps)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            
            # Pad or truncate to MAX_LEN
            if mfcc.shape[1] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :MAX_LEN]
                
            # Convert to PyTorch tensor. 
            # We want shape: (Channels, Sequence Length) for Conv1d
            # Let Channels = n_mfcc, Length = MAX_LEN
            # Actually standard Conv1d expects (batch_size, in_channels, seq_len)
            tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32)
            tensor_label = torch.tensor(label, dtype=torch.long)
            
            return tensor_mfcc, tensor_label
        except Exception as e:
            # In case of broken file, return zeros
            logging.error(f"Error loading {file_path}: {e}")
            return torch.zeros((N_MFCC, MAX_LEN), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class AudioThreatNet(nn.Module):
    def __init__(self):
        super(AudioThreatNet, self).__init__()
        # Input shape: (Batch, N_MFCC, MAX_LEN)
        
        # 1D CNN block
        self.conv1 = nn.Conv1d(in_channels=N_MFCC, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # LSTM block
        # The output of conv2 after pooling will be (Batch, 128, MAX_LEN/2)
        # We need to reshape for LSTM which expects (Batch, Seq_Len, Features) if batch_first=True
        # So we permute to (Batch, MAX_LEN/2, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully connected block
        self.fc = nn.Linear(64, 2) # 2 classes: 0=Safe, 1=Threat
        
    def forward(self, x):
        # x: (B, N_MFCC, MAX_LEN)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Reshape for LSTM: (B, Channels, Seq_Len) -> (B, Seq_Len, Channels)
        x = x.permute(0, 2, 1)
        
        # LSTM forward
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        # Fully connected
        out = self.fc(out)
        return out

def prepare_data():
    safe_files = list(SAFE_DIR.glob("*.wav"))
    threat_files = list(THREAT_DIR.glob("*.wav"))
    
    # Optional: Limiting samples to ensure balanced dataset or faster prototyping
    # np.random.shuffle(safe_files)
    # np.random.shuffle(threat_files)
    # min_len = min(len(safe_files), len(threat_files))
    # safe_files = safe_files[:min_len]
    # threat_files = threat_files[:min_len]

    files = safe_files + threat_files
    labels = [0] * len(safe_files) + [1] * len(threat_files)
    
    return train_test_split(files, labels, test_size=0.2, random_state=42)

def train_model():
    train_files, val_files, train_labels, val_labels = prepare_data()
    logging.info(f"Training on {len(train_files)} files, Validating on {len(val_files)} files")
    
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)
    
    # Use multiple workers for faster data loading if possible, but 0 is safer for stability
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = AudioThreatNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | "
                     f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                     
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved new best model to {MODEL_SAVE_PATH}")
            
    logging.info("Training complete.")

if __name__ == "__main__":
    train_model()

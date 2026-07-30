import os
import glob
import shutil
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

# Use a dynamic path to the current directory
BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "Training_Data"
SAFE_DIR = TRAINING_DIR / "Class_0_Safe"
THREAT_DIR = TRAINING_DIR / "Class_1_Threat"
RAW_DATASETS_DIR = BASE_DIR.parent / "datasets for acustic model training"
MODEL_SAVE_PATH = BASE_DIR / "audio_threat_model.pth"

# Hyperparameters
MAX_LEN = 150      # Fixed sequence length for features
N_MFCC = 40        # Base MFCCs
N_CHANNELS = 120   # 40 MFCC + 40 Delta + 40 Delta-Delta
BATCH_SIZE = 32
EPOCHS = 25        # Training epochs
LEARNING_RATE = 0.001

def extract_features(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extracts a 120-channel feature matrix:
    - 40 Base MFCCs
    - 40 First-order Delta velocity coefficients
    - 40 Second-order Delta-Delta acceleration coefficients
    Performs per-sample Z-score standardization and pads/truncates to MAX_LEN=150.
    Returns shape: (120, MAX_LEN).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Stack 40 MFCC + 40 Delta + 40 Delta2 -> 120 channels
    features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    # Per-sample Z-score standardization
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)

    # Pad or truncate to MAX_LEN (150)
    if features.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :MAX_LEN]

    return features.astype(np.float32)


def apply_spec_augment(features: np.ndarray, f_max: int = 10, t_max: int = 15) -> np.ndarray:
    """
    Applies SpecAugment data augmentation:
    - Frequency Masking: Zero out random contiguous block of 0 to f_max channels (0 to 10).
    - Time Masking: Zero out random contiguous block of 0 to t_max time steps (0 to 15).
    """
    augmented = features.copy()
    num_channels, num_frames = augmented.shape

    # Frequency Masking
    f = np.random.randint(0, f_max + 1)
    if f > 0 and num_channels > f:
        f0 = np.random.randint(0, num_channels - f)
        augmented[f0 : f0 + f, :] = 0.0

    # Time Masking
    t = np.random.randint(0, t_max + 1)
    if t > 0 and num_frames > t:
        t0 = np.random.randint(0, num_frames - t)
        augmented[:, t0 : t0 + t] = 0.0

    return augmented


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, is_train: bool = True):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load audio at standard 22050 Hz for 3.0 seconds
            y, sr = librosa.load(file_path, sr=22050, duration=3.0)

            # Extract 120-channel features
            features = extract_features(y, sr)

            # Apply SpecAugment if in training mode
            if self.is_train:
                features = apply_spec_augment(features, f_max=10, t_max=15)

            tensor_features = torch.tensor(features, dtype=torch.float32)
            tensor_label = torch.tensor(label, dtype=torch.long)

            return tensor_features, tensor_label
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return torch.zeros((N_CHANNELS, MAX_LEN), dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class AudioThreatNet(nn.Module):
    def __init__(self, in_channels: int = 120):
        super(AudioThreatNet, self).__init__()
        # Input shape: (Batch, 120, MAX_LEN)
        
        # 1D CNN block 1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 1D CNN block 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # LSTM block
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully connected output block (2 classes: 0=Safe, 1=Threat)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # x: (B, 120, MAX_LEN)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)       # (B, 64, MAX_LEN/2)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)       # (B, 128, MAX_LEN/4)
        
        # Reshape for LSTM: (B, Channels, Seq_Len) -> (B, Seq_Len, Channels)
        x = x.permute(0, 2, 1)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x)
        
        # Take the output at the last sequence time step
        out = out[:, -1, :] 
        
        # Fully connected linear classification
        out = self.fc(out)
        return out


def prepare_data():
    SAFE_DIR.mkdir(parents=True, exist_ok=True)
    THREAT_DIR.mkdir(parents=True, exist_ok=True)

    safe_files = list(SAFE_DIR.glob("*.wav"))
    threat_files = list(THREAT_DIR.glob("*.wav"))

    if len(safe_files) == 0 or len(threat_files) == 0:
        logging.info("Training_Data directories empty. Populating from raw datasets...")
        ravdess_files = list(RAW_DATASETS_DIR.rglob("*.wav")) if RAW_DATASETS_DIR.exists() else []

        if ravdess_files:
            logging.info(f"Found {len(ravdess_files)} dataset WAV files. Processing emotion classes...")
            for audio_file in ravdess_files:
                parts = audio_file.name.split('-')
                if len(parts) == 7:
                    emotion = parts[2]
                    # Emotion 05 = Angry, 06 = Fearful -> THREAT (Class 1)
                    if emotion in ["05", "06"]:
                        dest = THREAT_DIR / f"ravdess_{audio_file.name}"
                        if not dest.exists():
                            shutil.copy2(audio_file, dest)
                    # Emotion 01 = Neutral, 02 = Calm, 03 = Happy -> SAFE (Class 0)
                    elif emotion in ["01", "02", "03"]:
                        dest = SAFE_DIR / f"ravdess_{audio_file.name}"
                        if not dest.exists():
                            shutil.copy2(audio_file, dest)

            safe_files = list(SAFE_DIR.glob("*.wav"))
            threat_files = list(THREAT_DIR.glob("*.wav"))

    if len(safe_files) == 0 or len(threat_files) == 0:
        raise RuntimeError("No training files found in SAFE_DIR or THREAT_DIR.")

    # Ensure balanced dataset to prevent model bias
    np.random.shuffle(safe_files)
    np.random.shuffle(threat_files)
    min_len = min(len(safe_files), len(threat_files))
    safe_files = safe_files[:min_len]
    threat_files = threat_files[:min_len]

    files = safe_files + threat_files
    labels = [0] * len(safe_files) + [1] * len(threat_files)
    
    return train_test_split(files, labels, test_size=0.2, random_state=42)


def train_model():
    train_files, val_files, train_labels, val_labels = prepare_data()
    logging.info(f"Training on {len(train_files)} files, Validating on {len(val_files)} files")
    
    train_dataset = AudioDataset(train_files, train_labels, is_train=True)
    val_dataset = AudioDataset(val_files, val_labels, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = AudioThreatNet(in_channels=N_CHANNELS).to(device)
    
    # Cost-Sensitive Loss Weighting (Class 0: Safe = 1.0, Class 1: Threat = 1.2)
    class_weights = torch.tensor([1.0, 1.2], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

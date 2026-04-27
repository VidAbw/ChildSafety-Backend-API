import io
import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Constants (Must match training script)
N_MFCC = 40
MAX_LEN = 150
MODEL_PATH = Path(r"c:\Users\94771\Desktop\ChildSafety-Backend-API\audio model training\audio_threat_model.pth")

class AudioThreatNet(nn.Module):
    def __init__(self):
        super(AudioThreatNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=N_MFCC, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class ThreatPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioThreatNet().to(self.device)
        self.is_loaded = False
        
        self.load_model()

    def load_model(self):
        if MODEL_PATH.exists():
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.model.eval()
                self.is_loaded = True
                logger.info("ThreatPredictor model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load ThreatPredictor model: {e}")
        else:
            logger.warning("ThreatPredictor model file not found. Inference will return Safe by default.")

    def predict_from_wav_bytes(self, wav_bytes: bytes) -> tuple[int, float]:
        """
        Takes raw WAV bytes, extracts MFCCs, and runs inference.
        Returns (Class_ID, Probability)
        Class_ID: 0 = Safe, 1 = Threat
        """
        if not self.is_loaded:
            return 0, 0.0

        try:
            # librosa.load can read from file-like object using soundfile
            # but soundfile requires a seekable stream. BytesIO provides this.
            wav_io = io.BytesIO(wav_bytes)
            y, sr = librosa.load(wav_io, sr=22050, duration=3.0)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            
            if mfcc.shape[1] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :MAX_LEN]
                
            tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_mfcc)
                probabilities = torch.softmax(outputs, dim=1)
                
                prob_threat = probabilities[0][1].item()
                predicted_class = 1 if prob_threat > 0.5 else 0
                
            return predicted_class, prob_threat
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0, 0.0

    def verify_parent(self, wav_bytes: bytes, parent_profile_path: Path) -> bool:
        """
        Uses simple MFCC distance (Dynamic Time Warping) to check if the voice 
        matches the parent's registered profile.
        Returns True if it's the parent, False otherwise.
        """
        if not parent_profile_path.exists():
            return False
            
        try:
            # Load incoming chunk
            wav_io = io.BytesIO(wav_bytes)
            y_in, sr_in = librosa.load(wav_io, sr=22050)
            mfcc_in = librosa.feature.mfcc(y=y_in, sr=sr_in, n_mfcc=20)
            
            # Load parent profile
            y_parent, sr_parent = librosa.load(parent_profile_path, sr=22050)
            mfcc_parent = librosa.feature.mfcc(y=y_parent, sr=sr_parent, n_mfcc=20)
            
            # Compute DTW distance
            D, wp = librosa.sequence.dtw(X=mfcc_in, Y=mfcc_parent, metric='cosine')
            dist = D[-1, -1] / len(wp) # Normalize by path length
            
            # Threshold for Cosine distance DTW
            # Lower distance means more similar. Adjust threshold as needed.
            THRESHOLD = 0.15 
            
            if dist < THRESHOLD:
                logger.info(f"Parent voice verified. Distance: {dist:.4f}")
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error during parent verification: {e}")
            return False

predictor = ThreatPredictor()

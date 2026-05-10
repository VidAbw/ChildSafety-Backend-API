import io
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Constants (Must match training script)
N_MFCC = 40
MAX_LEN = 150
# Use a relative path so it works on any computer
MODEL_PATH = Path(__file__).parent.parent / "audio model training" / "audio_threat_model.pth"
# Local fallback WAV (kept for backward compatibility)
LOCAL_PROFILE_PATH = Path("parent_profile.wav")

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

    def extract_mfcc_matrix(self, wav_bytes: bytes, n_mfcc: int = 20) -> Optional[np.ndarray]:
        """
        Extracts an MFCC matrix from raw audio bytes.
        Supports WAV and WebM (from browser) via librosa + audioread fallback.
        Returns a 2D numpy array of shape (n_mfcc, T) or None on failure.
        """
        logger.info(f"extract_mfcc_matrix: received {len(wav_bytes)} bytes")
        try:
            wav_io = io.BytesIO(wav_bytes)
            try:
                y, sr = librosa.load(wav_io, sr=22050)
            except Exception as e1:
                logger.warning(f"soundfile decode failed ({e1}), trying audioread fallback...")
                wav_io.seek(0)
                y, sr = librosa.load(wav_io, sr=22050, res_type='kaiser_fast')

            logger.info(f"extract_mfcc_matrix: decoded {len(y)} samples at {sr}Hz")

            if len(y) < 100:  # ~4ms at 22kHz — practically silent/empty
                logger.error(f"Audio too short after decode: {len(y)} samples. File may be corrupt or silent.")
                return None

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            logger.info(f"extract_mfcc_matrix: MFCC shape {mfcc.shape}")
            return mfcc
        except Exception as e:
            logger.error(f"MFCC extraction failed: {type(e).__name__}: {e}")
            return None

    def predict_from_wav_bytes(self, wav_bytes: bytes) -> tuple[int, float]:
        """
        Takes raw WAV bytes, extracts MFCCs, and runs inference.
        Returns (Class_ID, Probability)
        Class_ID: 0 = Safe, 1 = Threat
        """
        if not self.is_loaded:
            return 0, 0.0

        try:
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

    def verify_parent_from_matrix(self, wav_bytes: bytes, stored_mfcc_list: list) -> bool:
        """
        Verifies if the incoming audio matches ANY of the stored MFCC profile matrices.
        This is the primary verification path — reads from Supabase registered_voice_profiles.
        
        stored_mfcc_list: A list of 2D MFCC arrays (deserialized from Supabase jsonb).
        Returns True if the voice matches any registered profile.
        """
        THRESHOLD = 0.18  # Cosine DTW distance — lower = more similar. Tunable.
        
        mfcc_in = self.extract_mfcc_matrix(wav_bytes, n_mfcc=20)
        if mfcc_in is None:
            return False

        for stored_matrix in stored_mfcc_list:
            try:
                mfcc_stored = np.array(stored_matrix, dtype=np.float32)
                D, wp = librosa.sequence.dtw(X=mfcc_in, Y=mfcc_stored, metric='cosine')
                dist = float(D[-1, -1]) / len(wp)
                logger.info(f"DTW distance to stored profile: {dist:.4f} (threshold: {THRESHOLD})")
                if dist < THRESHOLD:
                    return True
            except Exception as e:
                logger.warning(f"Comparison against one profile failed: {e}")
                continue

        return False

    def verify_parent(self, wav_bytes: bytes, parent_profile_path: Path) -> bool:
        """
        Fallback verification using a local WAV file.
        Used when no Supabase profiles exist.
        """
        if not parent_profile_path.exists():
            return False

        # Guard: if the file is too small it is likely corrupt/empty
        if parent_profile_path.stat().st_size < 10_000:
            logger.warning(
                f"Local profile WAV too small ({parent_profile_path.stat().st_size} bytes). "
                "Please re-register a voice profile via the dashboard."
            )
            return False

        try:
            wav_io = io.BytesIO(wav_bytes)
            y_in, sr_in = librosa.load(wav_io, sr=22050)
            mfcc_in = librosa.feature.mfcc(y=y_in, sr=sr_in, n_mfcc=20)

            y_parent, sr_parent = librosa.load(parent_profile_path, sr=22050)
            mfcc_parent = librosa.feature.mfcc(y=y_parent, sr=sr_parent, n_mfcc=20)

            D, wp = librosa.sequence.dtw(X=mfcc_in, Y=mfcc_parent, metric='cosine')
            dist = D[-1, -1] / len(wp)

            THRESHOLD = 0.18
            if dist < THRESHOLD:
                logger.info(f"Parent verified (local fallback). Distance: {dist:.4f}")
                return True

            return False
        except Exception as e:
            logger.error(f"Error during local parent verification: {type(e).__name__}: {e}")
            return False

predictor = ThreatPredictor()

import io
import os
import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Constants (Must match training script)
N_MFCC = 40
N_CHANNELS = 120   # 40 MFCC + 40 Delta + 40 Delta2
MAX_LEN = 150
# Use a relative path so it works on any computer
MODEL_PATH = Path(__file__).parent.parent / "audio model training" / "audio_threat_model.pth"
# Local fallback WAV (kept for backward compatibility)
LOCAL_PROFILE_PATH = Path("parent_profile.wav")

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


class AudioThreatNet(nn.Module):
    def __init__(self, in_channels: int = N_CHANNELS):
        super(AudioThreatNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
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
        self.model = AudioThreatNet(in_channels=N_CHANNELS).to(self.device)
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

    def decode_audio(self, audio_bytes: bytes, target_sr: int = 22050, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Decodes audio bytes from ANY container (WAV, WebM, M4A, OGG, MP3, AAC, FLAC)
        into a mono float32 numpy array at target_sr.
        Supports in-memory decoding with PyAV (libav) and soundfile/librosa fallback.
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return None

        # 1. Try PyAV (fastest universal in-memory decoder for WebM, M4A, Opus, AAC, MP3, WAV)
        try:
            import av
            container = av.open(io.BytesIO(audio_bytes))
            if container.streams.audio:
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
                frames_list = []
                for frame in container.decode(stream):
                    for resampled in resampler.resample(frame):
                        frames_list.append(resampled.to_ndarray())
                if frames_list:
                    y = np.concatenate(frames_list, axis=1).squeeze().astype(np.float32)
                    if duration is not None:
                        max_len = int(target_sr * duration)
                        y = y[:max_len]
                    if len(y) > 100:
                        return y
        except Exception as e_av:
            logger.debug(f"PyAV stream decode skipped: {e_av}")

        # 2. Try librosa / soundfile direct in-memory
        try:
            wav_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(wav_io, sr=target_sr, duration=duration)
            if y is not None and len(y) > 100:
                return y.astype(np.float32)
        except Exception:
            pass

        # 3. Temp file fallback based on magic bytes
        suffix = ".wav"
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
            suffix = ".webm"
        elif audio_bytes.startswith(b'OggS'):
            suffix = ".ogg"
        elif b'ftyp' in audio_bytes[:32]:
            suffix = ".m4a"
        elif audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb'):
            suffix = ".mp3"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=target_sr, duration=duration)
            if y is not None and len(y) > 100:
                return y.astype(np.float32)
        except Exception as e_tmp:
            logger.warning(f"Temp file decode error ({suffix}): {e_tmp}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return None

    def extract_mfcc_matrix(self, wav_bytes: bytes, n_mfcc: int = 20) -> Optional[np.ndarray]:
        """
        Extracts an MFCC matrix from raw audio bytes for DTW voice matching.
        Supports WAV, WebM (browser), M4A (mobile) via universal decoding.
        Returns a 2D numpy array of shape (n_mfcc, T) or None on failure.
        """
        logger.info(f"extract_mfcc_matrix: received {len(wav_bytes)} bytes")
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 100:
            logger.error("Audio too short or could not be decoded.")
            return None

        logger.info(f"extract_mfcc_matrix: decoded {len(y)} samples at 22050Hz (~{len(y)/22050:.2f}s)")
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=n_mfcc)
        logger.info(f"extract_mfcc_matrix: MFCC shape {mfcc.shape}")
        return mfcc

    def predict_from_wav_bytes(self, wav_bytes: bytes) -> tuple[int, float]:
        """
        Takes raw audio bytes, extracts 120-channel features, and runs inference.
        Returns (Class_ID, Probability)
        Class_ID: 0 = Safe, 1 = Threat
        """
        if not self.is_loaded:
            return 0, 0.0

        try:
            y = self.decode_audio(wav_bytes, target_sr=22050, duration=3.0)
            if y is None:
                return 0, 0.0

            features = extract_features(y, sr=22050)
            tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_features)
                probabilities = torch.softmax(outputs, dim=1)
                
                prob_threat = probabilities[0][1].item()
                
            return (1 if prob_threat >= 0.5 else 0, prob_threat)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0, 0.0

    def verify_parent_from_matrix(self, wav_bytes: bytes, stored_mfcc_list: list) -> tuple[bool, Optional[dict], float]:
        """
        Verifies if the incoming audio matches ANY of the stored MFCC profile matrices.
        Supports both raw 2D matrices and profile dicts: {'matrix': [...], 'person_name': '...', 'role': '...', 'id': '...'}.
        Returns (is_match: bool, matched_profile: Optional[dict], min_distance: float).
        """
        THRESHOLD = 0.18  # Cosine DTW distance — lower = more similar. Tunable.
        
        mfcc_in = self.extract_mfcc_matrix(wav_bytes, n_mfcc=20)
        if mfcc_in is None:
            return False, None, 1.0

        min_dist = float('inf')
        matched_profile = None

        for item in stored_mfcc_list:
            try:
                if isinstance(item, dict):
                    stored_matrix = item.get("matrix")
                    name = item.get("person_name")
                    role = item.get("role", "Parent")
                    prof_id = item.get("id")
                else:
                    stored_matrix = item
                    name = None
                    role = "Parent"
                    prof_id = None

                if stored_matrix is None:
                    continue

                mfcc_stored = np.array(stored_matrix, dtype=np.float32)
                D, wp = librosa.sequence.dtw(X=mfcc_in, Y=mfcc_stored, metric='cosine')
                dist = float(D[-1, -1]) / len(wp)
                
                if dist < min_dist:
                    min_dist = dist
                    matched_profile = {"name": name, "role": role, "id": prof_id}

                logger.info(f"DTW distance to '{name or 'stored profile'}' ({role}): {dist:.4f} (threshold: {THRESHOLD})")
                if dist < THRESHOLD:
                    return True, {"name": name, "role": role, "id": prof_id}, dist
            except Exception as e:
                logger.warning(f"Comparison against profile failed: {e}")
                continue

        return False, matched_profile, (min_dist if min_dist != float('inf') else 1.0)

    def verify_parent(self, wav_bytes: bytes, parent_profile_path: Path) -> bool:
        """
        Fallback verification using a local WAV file.
        Used when no Supabase profiles exist.
        """
        if not parent_profile_path.exists():
            return False

        # Guard: if the file is too small it is likely corrupt/empty
        if parent_profile_path.stat().st_size < 1000:
            logger.warning(
                f"Local profile WAV too small ({parent_profile_path.stat().st_size} bytes). "
                "Please re-register a voice profile via the dashboard."
            )
            return False

        try:
            y_in = self.decode_audio(wav_bytes, target_sr=22050)
            if y_in is None:
                return False
            mfcc_in = librosa.feature.mfcc(y=y_in, sr=22050, n_mfcc=20)

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

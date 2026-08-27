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

    def validate_and_extract_voice(self, y: np.ndarray, sr: int = 22050) -> tuple[bool, Optional[np.ndarray], str]:
        """
        Performs multi-stage Voice Activity Detection (VAD) and speech extraction:
        1. Checks overall energy & decibel level against noise floor.
        2. Performs frame-by-frame RMS and spectral centroid analysis to detect human vocal presence.
        3. Validates minimum voiced speech duration (at least 1.0s of actual speech).
        4. Verifies harmonic energy via Harmonic-Percussive Source Separation (HPSS).
        5. Extracts and concatenates isolated voiced audio segments.
        Returns: (is_valid: bool, voiced_audio: Optional[np.ndarray], message: str)
        """
        if y is None or len(y) < int(sr * 1.5):
            return False, None, "Recording is too short. Please speak for at least 3 to 5 seconds."

        # 1. Overall volume / RMS check
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)

        if db < 40.0 or rms < 0.003:
            logger.warning(f"Voice validation failed: Audio is silent/too quiet (RMS={rms:.5f}, dB={db:.1f}).")
            return False, None, "No sound detected (Audio was silent or too quiet). Please speak clearly into the microphone."

        # 2. Short-time frame analysis (approx 46ms frame, 23ms hop)
        frame_length = 1024
        hop_length = 512
        try:
            frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            frame_zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            frame_sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        except Exception as e_feat:
            logger.error(f"Error computing frame acoustic features: {e_feat}")
            return False, None, "Could not process audio features."

        # Thresholds tailored for conversational voice vs room noise
        max_frame_rms = float(np.max(frame_rms)) if len(frame_rms) > 0 else 0.0
        thresh_rms = max(0.010, max_frame_rms * 0.20)

        # Voiced frames: sufficient energy, spectral centroid within vocal range (150 - 4500 Hz), reasonable ZCR
        voiced_mask = (
            (frame_rms >= thresh_rms) &
            (frame_sc >= 150.0) &
            (frame_sc <= 4500.0) &
            (frame_zcr <= 0.40)
        )

        num_voiced_frames = int(np.sum(voiced_mask))
        voiced_duration = (num_voiced_frames * hop_length) / sr

        logger.info(f"VAD Analysis: Peak RMS={max_frame_rms:.4f}, Total Voiced Duration={voiced_duration:.2f}s ({num_voiced_frames} frames)")

        if voiced_duration < 1.0:
            logger.warning(f"Voice validation failed: Voiced speech duration ({voiced_duration:.2f}s) is under the 1.0s minimum threshold.")
            return False, None, f"No voice detected or sample was too short (Only {voiced_duration:.1f}s of speech). Please speak clearly for 3 to 5 seconds."

        # 3. Voiced Segment Extraction with temporal padding
        # Dilate mask by 2 frames (approx 50ms) on each side to preserve phoneme onsets/offsets
        dilated_mask = voiced_mask.copy()
        for i in range(len(dilated_mask)):
            if voiced_mask[i]:
                dilated_mask[max(0, i - 2): min(len(dilated_mask), i + 3)] = True

        sample_mask = np.zeros(len(y), dtype=bool)
        for i, is_voiced in enumerate(dilated_mask):
            if is_voiced:
                start_sample = i * hop_length
                end_sample = min(len(y), start_sample + frame_length)
                sample_mask[start_sample:end_sample] = True

        y_voiced = y[sample_mask]
        if len(y_voiced) < int(sr * 0.8):
            return False, None, "Extracted speech segment is too short. Please speak clearly for 3 to 5 seconds."

        # 4. Harmonic Formant Confirmation (Differentiates human voice from stationary noise/whistle)
        try:
            y_harm, _ = librosa.effects.hpss(y_voiced)
            harm_rms = float(np.sqrt(np.mean(y_harm**2)))
            if harm_rms < 0.004:
                logger.warning(f"Harmonic verification failed: harm_rms={harm_rms:.5f}")
                return False, None, "Audio lacks human voice characteristics. Please speak clearly into the microphone."
        except Exception:
            pass

        return True, y_voiced, f"Voice successfully extracted ({voiced_duration:.1f}s voiced speech, {db:.1f} dB)"

    def extract_mfcc_matrix(self, wav_bytes: bytes, n_mfcc: int = 20, require_vad: bool = True) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Extracts a Cepstral Mean and Variance Normalized (CMVN) MFCC matrix from raw audio bytes.
        If require_vad is True (used during registration), validates voice activity and extracts pure speech.
        Returns (mfcc_matrix, error_message_if_any).
        """
        logger.info(f"extract_mfcc_matrix: received {len(wav_bytes)} bytes (require_vad={require_vad})")
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 100:
            logger.error("Audio too short or could not be decoded.")
            return None, "Audio could not be decoded or was empty."

        if require_vad:
            is_valid, y_speech, err_msg = self.validate_and_extract_voice(y, sr=22050)
            if not is_valid or y_speech is None:
                return None, err_msg
            target_y = y_speech
        else:
            # For live verification chunks, trim silent margins
            try:
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                target_y = y_trimmed if len(y_trimmed) > 1000 else y
            except Exception:
                target_y = y

        logger.info(f"extract_mfcc_matrix: computing MFCC on {len(target_y)} samples (~{len(target_y)/22050:.2f}s)")
        mfcc = librosa.feature.mfcc(y=target_y, sr=22050, n_mfcc=n_mfcc)

        # Cepstral Mean Normalization (CMN) across frames
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        # Unit Variance Normalization
        std = np.std(mfcc, axis=1, keepdims=True)
        std[std == 0] = 1e-6
        mfcc = mfcc / std

        logger.info(f"extract_mfcc_matrix: CMVN normalized MFCC shape {mfcc.shape}")
        return mfcc.astype(np.float32), None

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
        THRESHOLD = 0.35  # Cosine DTW distance for CMVN-normalized matrices (allows cross-mic matching while rejecting strangers).
        
        # Check volume before running DTW
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return False, None, 1.0

        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)
        if db < 40.0:
            # Sound is below speech volume — skip DTW to prevent false matches on room silence
            return False, None, 1.0

        mfcc_in, err = self.extract_mfcc_matrix(wav_bytes, n_mfcc=20, require_vad=False)
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
                # Apply CMVN to stored matrix if raw
                mfcc_stored = mfcc_stored - np.mean(mfcc_stored, axis=1, keepdims=True)
                std_stored = np.std(mfcc_stored, axis=1, keepdims=True)
                std_stored[std_stored == 0] = 1e-6
                mfcc_stored = mfcc_stored / std_stored

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

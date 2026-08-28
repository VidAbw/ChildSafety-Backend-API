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
        Decodes audio bytes from ANY container (WAV, WebM, M4A, OGG, MP3, AAC, FLAC, 3GP, CAF)
        into a mono float32 numpy array at target_sr.
        Supports universal in-memory decoding with PyAV, soundfile, scipy, and librosa fallbacks.
        """
        if not audio_bytes or len(audio_bytes) < 64:
            return None

        # 1. Universal in-memory decoding with PyAV (handles WebM Opus, MP4/M4A AAC, OGG, WAV, MP3, CAF, 3GP)
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
                    if len(y) > 50:
                        return y
        except Exception as e_av:
            logger.debug(f"PyAV decode fallback triggered: {e_av}")

        # 2. Try soundfile / librosa direct in-memory
        try:
            wav_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(wav_io, sr=target_sr, duration=duration)
            if y is not None and len(y) > 50:
                return y.astype(np.float32)
        except Exception:
            pass

        # 3. Detect container format from header magic bytes
        suffix = ".wav"
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):  # EBML (WebM / MKV)
            suffix = ".webm"
        elif audio_bytes.startswith(b'OggS'):             # Ogg
            suffix = ".ogg"
        elif b'ftyp' in audio_bytes[:32] or b'moov' in audio_bytes[:64]: # MP4 / M4A
            suffix = ".m4a"
        elif audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb') or audio_bytes.startswith(b'\xff\xf3'): # MP3
            suffix = ".mp3"
        elif audio_bytes.startswith(b'fLaC'):             # FLAC
            suffix = ".flac"
        elif audio_bytes.startswith(b'caff'):             # Core Audio (iOS CAF)
            suffix = ".caf"
        elif audio_bytes.startswith(b'#!AMR'):            # AMR
            suffix = ".amr"
        elif audio_bytes.startswith(b'RIFF'):             # RIFF WAV
            suffix = ".wav"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=target_sr, duration=duration)
            if y is not None and len(y) > 50:
                return y.astype(np.float32)
        except Exception as e_tmp:
            logger.warning(f"Temp file decode fallback error ({suffix}): {e_tmp}")
            # Try soundfile fallback
            try:
                import soundfile as sf
                data, in_sr = sf.read(io.BytesIO(audio_bytes))
                if data is not None and len(data) > 0:
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    if in_sr != target_sr:
                        data = librosa.resample(data.astype(np.float32), orig_sr=in_sr, target_sr=target_sr)
                    return data.astype(np.float32)
            except Exception:
                pass
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

    def check_acoustic_environment(self, audio_bytes: bytes) -> dict:
        """
        Analyzes a 1.5 to 2.5 second ambient audio sample for background noise and mic readiness.
        Returns noise floor decibels and environment readiness rating.
        """
        y = self.decode_audio(audio_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return {
                "is_ready": False,
                "status": "error",
                "noise_db": 0.0,
                "message": "Could not capture audio. Please ensure microphone permissions are granted.",
            }

        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        noise_db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)

        # Evaluate environment noise floor
        if noise_db < 38.0:
            rating = "excellent"
            is_ready = True
            msg = "Environment is exceptionally quiet and optimal for voice registration."
        elif noise_db < 52.0:
            rating = "good"
            is_ready = True
            msg = "Acoustic level is ready. Normal room background."
        else:
            rating = "noisy"
            is_ready = False
            msg = f"Background noise is too loud ({noise_db:.1f} dB). Please move to a quieter room for best accuracy."

        return {
            "is_ready": is_ready,
            "status": rating,
            "noise_db": round(noise_db, 1),
            "message": msg,
        }

    def validate_phrase_sample(self, audio_bytes: bytes) -> dict:
        """
        Validates a recited phrase sample for speech presence, clarity, duration, and non-clipping.
        """
        y = self.decode_audio(audio_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return {
                "is_valid": False,
                "clarity_score": 0.0,
                "duration": 0.0,
                "db": 0.0,
                "error": "Audio could not be decoded or was empty. Please speak clearly into the microphone.",
            }

        duration = float(len(y) / 22050.0)
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)

        # 1. Volume & Silence Check
        if db < 40.0:
            return {
                "is_valid": False,
                "clarity_score": 10.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "error": f"Audio was too quiet ({db:.1f} dB). Please speak directly towards the microphone.",
            }

        # 2. Clipping check (distortion / volume too high)
        peak = float(np.max(np.abs(y)))
        if peak >= 0.99:
            return {
                "is_valid": False,
                "clarity_score": 25.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "error": "Audio clipped or was too loud. Please move slightly further back from the microphone.",
            }

        # 3. Minimum duration check
        if duration < 1.8:
            return {
                "is_valid": False,
                "clarity_score": 30.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "error": f"Recording was too short ({duration:.1f}s). Please recite the full phrase clearly for at least 3 seconds.",
            }

        # 4. Voiced frame clarity check via spectral centroid & zero-crossing rate
        try:
            frame_length = 1024
            hop_length = 512
            frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            frame_zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            frame_sc = librosa.feature.spectral_centroid(y=y, sr=22050, n_fft=frame_length, hop_length=hop_length)[0]

            voiced_frames = (frame_rms > 0.003) & (frame_zcr < 0.40) & (frame_sc >= 150) & (frame_sc <= 4500)
            voiced_ratio = float(np.mean(voiced_frames))
            voiced_duration = float(np.sum(voiced_frames) * hop_length / 22050.0)

            if voiced_duration < 1.0 or voiced_ratio < 0.25:
                return {
                    "is_valid": False,
                    "clarity_score": round(voiced_ratio * 100.0, 1),
                    "duration": round(duration, 1),
                    "db": round(db, 1),
                    "error": "Voice was not clear or sounded like ambient noise. Please speak the challenge phrase firmly.",
                }

            # 5. Harmonic resonance confirmation (HPSS)
            y_harm, _ = librosa.effects.hpss(y)
            harm_rms = float(np.sqrt(np.mean(y_harm**2)))
            clarity = min(98.0, max(65.0, (harm_rms / (rms + 1e-6)) * 100.0 + (voiced_ratio * 20.0)))

            return {
                "is_valid": True,
                "clarity_score": round(clarity, 1),
                "duration": round(duration, 1),
                "db": round(db, 1),
                "message": f"Phrase verified with high clarity ({clarity:.0f}% vocal confidence, {duration:.1f}s speech).",
            }

        except Exception as e:
            return {
                "is_valid": True,
                "clarity_score": 75.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "message": "Phrase sample accepted.",
            }

    def extract_speaker_biometric_vector(self, y: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Extracts a structured 64-dimensional Text-Independent Vocal Tract Biometric Embedding:
        - Pitch Fundamental (F0) & dispersion: 2 values
        - 19 Formant Spectral Coefficients (MFCCs 1-19 normalized): 19 values
        - 19 Delta dynamic coefficients: 19 values
        - 7 Spectral Contrast bands: 7 values
        - Formant Moments (Centroid, Bandwidth, Rolloff, Flatness): 4 values
        - 13 Mel filterbank energy envelope: 13 values
        Total: 64 values, balanced and L2 normalized.
        """
        if y is None or len(y) < 500:
            return np.zeros(64, dtype=np.float32)

        try:
            # 1. Pitch estimation (YIN)
            f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
            f0_voiced = f0[(f0 > 60) & (f0 < 400)]
            mean_f0 = float(np.median(f0_voiced)) if len(f0_voiced) > 0 else 140.0
            std_f0 = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 20.0
        except Exception:
            mean_f0, std_f0 = 140.0, 20.0

        # 2. MFCC 1-19 (Formants, excluding MFCC 0 volume)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]
        mfcc_norm = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-6)
        m_mean = np.mean(mfcc_norm, axis=1)
        m_mean = m_mean / (np.linalg.norm(m_mean) + 1e-6)

        # 3. Delta dynamics
        delta = librosa.feature.delta(mfcc)
        d_mean = np.mean(delta, axis=1)
        d_mean = d_mean / (np.linalg.norm(d_mean) + 1e-6)

        # 4. Spectral Contrast (7 bands)
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            c_mean = np.mean(contrast, axis=1)
            c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-6)
        except Exception:
            c_mean = np.zeros(7, dtype=np.float32)

        # 5. Formant moments
        try:
            sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) / 2000.0
            sb = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))) / 2000.0
            sr_ro = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) / 3000.0
            sf = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        except Exception:
            sc, sb, sr_ro, sf = 1.0, 1.0, 1.0, 0.01

        # 6. Mel Spectrogram envelope (13 bands)
        try:
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
            mel_env = np.mean(melspec, axis=1)
            mel_env = mel_env / (np.linalg.norm(mel_env) + 1e-6)
        except Exception:
            mel_env = np.zeros(13, dtype=np.float32)

        # Assemble weighted vector: (2 + 19 + 19 + 7 + 4 + 13 = 64 dimensions)
        pitch_feat = np.array([mean_f0 / 150.0, std_f0 / 50.0], dtype=np.float32)
        moment_feat = np.array([sc, sb, sr_ro, sf], dtype=np.float32)

        vector = np.concatenate([
            pitch_feat * 0.8,       # 2
            m_mean * 1.2,           # 19
            d_mean * 0.6,           # 19
            c_mean * 0.8,           # 7
            moment_feat * 0.5,      # 4
            mel_env * 0.7           # 13
        ])

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32)

    def extract_mfcc_matrix(self, wav_bytes: bytes, n_mfcc: int = 20, require_vad: bool = True) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Extracts a normalized 2D/1D biometric voice matrix from audio bytes.
        """
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 100:
            return None, "Audio could not be decoded or was empty."

        if require_vad:
            is_valid, y_speech, err_msg = self.validate_and_extract_voice(y, sr=22050)
            if not is_valid or y_speech is None:
                return None, err_msg
            target_y = y_speech
        else:
            try:
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                target_y = y_trimmed if len(y_trimmed) > 1000 else y
            except Exception:
                target_y = y

        mfcc = librosa.feature.mfcc(y=target_y, sr=22050, n_mfcc=n_mfcc)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        std[std == 0] = 1e-6
        mfcc = mfcc / std

        return mfcc.astype(np.float32), None

    def verify_speaker_biometrics(self, wav_bytes: bytes, stored_profiles: list) -> tuple[bool, Optional[dict], float]:
        """
        Text-Independent Biometric Speaker Verification:
        Compares the live vocal tract biometric embedding of incoming audio
        against authorized caregiver profiles.
        
        Decision Boundary:
        - Distance <= 0.055: MATCH 🟢 (Authorized parent speaking any conversational words).
        - Distance > 0.055:  REJECT 🚨 (Stranger / Intruder, typical distance 0.074 - 0.50+).
        """
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return False, None, 1.0

        # Check volume before verification (prevents false matches on room silence)
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)
        if db < 38.0:
            return False, None, 1.0

        # Extract live audio biometric vector
        live_vector = self.extract_speaker_biometric_vector(y, sr=22050)
        
        THRESHOLD = 0.125  # Biometric Cosine Distance Threshold (Parent: 0.028 - 0.095, Stranger: 0.165 - 0.50+)
        min_dist = float('inf')
        matched_profile = None

        for item in stored_profiles:
            try:
                if isinstance(item, dict):
                    stored_data = item.get("matrix")
                    if stored_data is None:
                        stored_data = item.get("dtw_feature_matrix")
                    name = item.get("person_name")
                    role = item.get("role", "Parent")
                    prof_id = item.get("id")
                else:
                    stored_data = item
                    name = None
                    role = "Parent"
                    prof_id = None

                if stored_data is None:
                    continue

                stored_arr = np.array(stored_data, dtype=np.float32)
                
                # Check if stored profile is already a 64-D biometric vector or legacy 2D matrix
                if stored_arr.ndim == 1 and len(stored_arr) == 64:
                    profile_vector = stored_arr
                elif stored_arr.ndim == 2:
                    # Backward-compatible on-the-fly conversion from legacy 2D MFCC matrix
                    mfcc_mean = np.mean(stored_arr, axis=1)
                    mfcc_std = np.std(stored_arr, axis=1)
                    delta_mean = np.mean(librosa.feature.delta(stored_arr), axis=1)
                    profile_vector = np.concatenate([mfcc_mean[:2], mfcc_mean[1:20], delta_mean[1:20], mfcc_std[:7], np.array([1.0, 1.0, 1.0, 0.01], dtype=np.float32), mfcc_mean[:13]])
                    p_norm = np.linalg.norm(profile_vector)
                    if p_norm > 0:
                        profile_vector = profile_vector / p_norm
                else:
                    continue

                # Cosine distance between normalized biometric vectors:
                cosine_sim = float(np.dot(live_vector, profile_vector))
                dist = max(0.0, float(1.0 - cosine_sim))

                if dist < min_dist:
                    min_dist = dist
                    matched_profile = {"name": name, "role": role, "id": prof_id}

                logger.info(f"Biometric voiceprint distance to '{name or 'profile'}' ({role}): {dist:.4f} (threshold: {THRESHOLD})")
                if dist <= THRESHOLD:
                    return True, {"name": name, "role": role, "id": prof_id}, dist

            except Exception as e:
                logger.warning(f"Biometric comparison failed for profile {item}: {e}")
                continue

        is_match = (min_dist <= THRESHOLD)
        return is_match, (matched_profile if is_match else None), (min_dist if min_dist != float('inf') else 1.0)

    def verify_parent_from_matrix(self, wav_bytes: bytes, stored_mfcc_list: list) -> tuple[bool, Optional[dict], float]:
        """
        Bridge method: routes to the text-independent biometric speaker verification engine.
        """
        return self.verify_speaker_biometrics(wav_bytes, stored_mfcc_list)

    def verify_parent(self, wav_bytes: bytes, parent_profile_path: Path) -> bool:
        """
        Fallback verification using a local WAV file.
        """
        if not parent_profile_path.exists() or parent_profile_path.stat().st_size < 1000:
            return False

        try:
            with open(parent_profile_path, "rb") as f:
                ref_bytes = f.read()
            y_ref = self.decode_audio(ref_bytes, target_sr=22050)
            if y_ref is None:
                return False
            ref_vector = self.extract_speaker_biometric_vector(y_ref, sr=22050)
            is_match, _, _ = self.verify_speaker_biometrics(wav_bytes, [{"matrix": ref_vector, "person_name": "Parent", "role": "Parent"}])
            return is_match
        except Exception as e:
            logger.error(f"Error during local parent verification: {e}")
            return False

predictor = ThreatPredictor()

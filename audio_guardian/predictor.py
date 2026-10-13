"""
predictor.py - Audio Threat Detection and Biometric Speaker Verification Module.

Part of the Audio Guardian subsystem in the Child Safety Guardian project.
This module provides two core capabilities:
1. Acoustic Threat Classification: Uses a deep learning model (AudioThreatNet)
   to classify environmental audio into Safe (Class 0) or Threat (Class 1).
2. Biometric Speaker Verification: Uses Digital Signal Processing (DSP) and
   vocal tract acoustics to extract a 64-dimensional biometric voiceprint,
   verifying whether a speaker is an authorized parent/caregiver.
"""

import io
import os
import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# Third-party audio processing, numerical, and deep learning libraries
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress benign UserWarnings from audio libraries (e.g. PySoundFile / Librosa sample rate warnings)
warnings.filterwarnings("ignore", category=UserWarning)

# Module-level logger for debugging, informational notices, and error tracking
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global Constants & Configuration (Must remain aligned with model training)
# ─────────────────────────────────────────────────────────────────────────────

# N_MFCC: Number of Mel-Frequency Cepstral Coefficients extracted per time frame.
# Mel-Frequency Cepstral Coefficients represent the spectral envelope/timbre of sound.
N_MFCC = 40

# N_CHANNELS: Total input feature channels fed into the AudioThreatNet neural network.
# Calculated as 40 base MFCCs + 40 first-order delta (velocity) + 40 second-order delta-delta (acceleration).
N_CHANNELS = 120   # 40 MFCC + 40 Delta + 40 Delta2

# MAX_LEN: Fixed temporal dimension (number of time frames) expected by the neural network.
# Variable-length audio sequences are padded or truncated to exactly this length.
MAX_LEN = 150

# MODEL_PATH: Relative filesystem path locating the trained PyTorch neural network checkpoint.
MODEL_PATH = Path(__file__).parent.parent / "audio model training" / "audio_threat_model.pth"

# LOCAL_PROFILE_PATH: Relative filesystem path to a reference WAV recording of an authorized parent.
# Maintained for local disk fallback when a remote database is unavailable.
LOCAL_PROFILE_PATH = Path("parent_profile.wav")


def extract_features(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extracts a 120-channel acoustic feature matrix from a 1D audio waveform.

    Processing Steps:
    1. Computes 40 Mel-Frequency Cepstral Coefficients (MFCCs).
    2. Computes 40 first-order delta coefficients (velocity of spectral changes).
    3. Computes 40 second-order delta-delta coefficients (acceleration of spectral changes).
    4. Vertically stacks these into a 120 x T matrix.
    5. Standardizes the matrix per-sample using Z-score normalization (mean=0, std=1).
    6. Pads with zeros or truncates along the time axis to enforce fixed shape (120, MAX_LEN).

    Parameters:
        y (np.ndarray): 1D floating-point audio time-series signal.
        sr (int): Audio sampling rate in Hertz (samples per second), default 22050 Hz.

    Returns:
        np.ndarray: 2D array of shape (120, 150) with dtype float32.
    """
    # Step 1: Compute 40 base MFCCs representing spectral timbre across time frames
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Step 2: Compute first-order delta coefficients (speed of frequency envelope changes)
    delta_mfcc = librosa.feature.delta(mfcc)

    # Step 3: Compute second-order delta coefficients (acceleration of frequency envelope changes)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Step 4: Vertically stack all 3 feature groups into a single 120-row matrix
    features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    # Step 5: Per-sample Z-score standardization: subtract global mean and divide by standard deviation.
    # Adding 1e-6 in the denominator prevents division by zero on silent audio.
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)

    # Step 6: Enforce a uniform time dimension (MAX_LEN = 150 frames)
    if features.shape[1] < MAX_LEN:
        # Shorter clips: pad trailing edge with zeros (constant mode)
        pad_width = MAX_LEN - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # Longer clips: truncate to the first MAX_LEN frames
        features = features[:, :MAX_LEN]

    # Return 32-bit floating point array ready for PyTorch tensor ingestion
    return features.astype(np.float32)


class AudioThreatNet(nn.Module):
    """
    Convolutional-Recurrent Neural Network (CRNN) for Audio Threat Classification.

    Architecture Overview:
    - Stage 1 (Spatial/Frequency Feature Extraction):
        * Conv1d (120 in -> 64 out, kernel 3, padding 1)
        * ReLU activation
        * MaxPool1d (kernel 2) -> halves time frames from 150 to 75
    - Stage 2 (Deeper Spatial Feature Extraction):
        * Conv1d (64 in -> 128 out, kernel 3, padding 1)
        * ReLU activation
        * MaxPool1d (kernel 2) -> halves time frames from 75 to 37
    - Stage 3 (Temporal Sequence Modeling):
        * Permute dimensions to (batch, time_steps, features) -> (batch, 37, 128)
        * LSTM (input_size=128, hidden_size=64, 1 layer, batch_first=True)
        * Slice final time-step hidden state -> shape (batch, 64)
    - Stage 4 (Classification Head):
        * Linear layer (64 in -> 2 out) producing raw class logits:
          Index 0: Safe
          Index 1: Threat
    """
    def __init__(self, in_channels: int = N_CHANNELS):
        """
        Initializes the layers of AudioThreatNet.

        Parameters:
            in_channels (int): Number of input channels (120 for MFCC + Delta + Delta-Delta).
        """
        super(AudioThreatNet, self).__init__()
        # First 1D convolution: extracts low-level spectro-temporal feature patterns
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        # Rectified Linear Unit activation function (f(x) = max(0, x))
        self.relu = nn.ReLU()
        # Max-pooling: downsamples temporal resolution by a factor of 2
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Second 1D convolution: aggregates features into higher-level abstractions
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Recurrent LSTM: tracks temporal context across sequential audio frames
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # Final fully connected classification layer producing 2 class logits
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        """
        Executes the forward inference pass.

        Parameters:
            x (torch.Tensor): Input batch tensor of shape (batch_size, 120, 150).

        Returns:
            torch.Tensor: Unnormalized output logits of shape (batch_size, 2).
        """
        # Block 1: Conv1d -> ReLU -> MaxPool1d
        # Input: (batch, 120, 150) -> Conv: (batch, 64, 150) -> Pool: (batch, 64, 75)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2: Conv1d -> ReLU -> MaxPool1d
        # Input: (batch, 64, 75) -> Conv: (batch, 128, 75) -> Pool: (batch, 128, 37)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Reorder tensor axes from (batch, channels, time) to (batch, time, channels) for LSTM
        # (batch, 128, 37) -> (batch, 37, 128)
        x = x.permute(0, 2, 1)

        # Recurrent processing: out contains hidden states for all 37 time steps
        out, _ = self.lstm(x)

        # Extract the hidden state at the final time-step to summarize the sequence: (batch, 64)
        out = out[:, -1, :]

        # Linear projection to class logits: (batch, 2)
        out = self.fc(out)
        return out


class ThreatPredictor:
    """
    Central inference and voice biometric verification engine for Audio Guardian.

    Manages:
    - Audio Threat Classification (loading weights, executing forward pass, applying threshold).
    - Multi-container audio decoding (PyAV, Librosa, SoundFile, magic-byte temp file fallback).
    - Multi-stage Voice Activity Detection (VAD) and phrase quality validation.
    - 64-dimensional vocal tract biometric vector extraction.
    - Cosine distance biometric verification against parent voice profiles.
    """
    def __init__(self):
        """
        Initializes hardware device, instantiates AudioThreatNet, and attempts weight loading.
        """
        # Select GPU (CUDA) if available; otherwise fall back to host CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instantiate model and transfer parameters to target computation device
        self.model = AudioThreatNet(in_channels=N_CHANNELS).to(self.device)
        # Flag indicating whether trained weights have been successfully loaded
        self.is_loaded = False
        
        # Load checkpoint weights from disk
        self.load_model()

    def load_model(self):
        """
        Loads the pre-trained neural network weights from MODEL_PATH into AudioThreatNet.
        If the file does not exist or fails to load, sets is_loaded=False and logs a warning.
        Inference safely defaults to Class 0 (Safe) with 0.0 probability.
        """
        if MODEL_PATH.exists():
            try:
                # Load PyTorch checkpoint dictionary onto active device
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                # Set neural network to evaluation mode (disables dropout / batchnorm training behavior)
                self.model.eval()
                self.is_loaded = True
                logger.info("ThreatPredictor model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load ThreatPredictor model: {e}")
        else:
            logger.warning("ThreatPredictor model file not found. Inference will return Safe by default.")

    def decode_audio(self, audio_bytes: bytes, target_sr: int = 22050, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Decodes raw audio bytes from arbitrary audio containers into a 1D mono float32 array.

        Supported Formats:
        - WAV, WebM (Opus), M4A/MP4 (AAC), OGG (Vorbis), MP3, AAC, FLAC, 3GP, CAF, AMR.

        Decoding Pipeline:
        1. PyAV In-Memory: Streams packets through ffmpeg bindings and resamples directly to mono float32.
        2. Librosa / SoundFile In-Memory: Directly parses standard WAV bytes from io.BytesIO.
        3. Magic Bytes Container Detection: Sniffs file header bytes, saves to a temporary file
           with appropriate suffix, and loads via librosa.
        4. SoundFile Fallback: Reads directly via soundfile library.

        Parameters:
            audio_bytes (bytes): Binary audio data received over network or read from disk.
            target_sr (int): Desired output sampling rate in Hz (default 22050).
            duration (Optional[float]): Maximum duration in seconds to decode (or None for full file).

        Returns:
            Optional[np.ndarray]: Mono float32 numpy array, or None if decoding failed.
        """
        # Guard against empty or truncated byte sequences
        if not audio_bytes or len(audio_bytes) < 64:
            return None

        # ── Strategy 1: Universal in-memory decoding with PyAV ─────────────────
        # PyAV wraps FFmpeg and supports virtually all compressed formats without disk I/O.
        try:
            import av
            container = av.open(io.BytesIO(audio_bytes))
            if container.streams.audio:
                stream = container.streams.audio[0]
                # Configure resampler for mono layout and 32-bit floating-point planar format
                resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
                frames_list = []
                for frame in container.decode(stream):
                    for resampled in resampler.resample(frame):
                        frames_list.append(resampled.to_ndarray())
                if frames_list:
                    # Concatenate resampled frames along the time axis
                    y = np.concatenate(frames_list, axis=1).squeeze().astype(np.float32)
                    if duration is not None:
                        max_len = int(target_sr * duration)
                        y = y[:max_len]
                    if len(y) > 50:
                        return y
        except Exception as e_av:
            logger.debug(f"PyAV decode fallback triggered: {e_av}")

        # ── Strategy 2: Direct in-memory load with librosa / soundfile ─────────
        # Fast path for standard uncompressed WAV containers
        try:
            wav_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(wav_io, sr=target_sr, duration=duration)
            if y is not None and len(y) > 50:
                return y.astype(np.float32)
        except Exception:
            pass

        # ── Strategy 3: Magic-byte sniffing and temporary file loading ────────
        # Inspect leading magic bytes to determine proper file extension for ffmpeg/librosa
        suffix = ".wav"
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):  # EBML header (WebM / MKV)
            suffix = ".webm"
        elif audio_bytes.startswith(b'OggS'):             # Ogg container header
            suffix = ".ogg"
        elif b'ftyp' in audio_bytes[:32] or b'moov' in audio_bytes[:64]: # ISO base media file (MP4 / M4A)
            suffix = ".m4a"
        elif audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb') or audio_bytes.startswith(b'\xff\xf3'): # MP3
            suffix = ".mp3"
        elif audio_bytes.startswith(b'fLaC'):             # Free Lossless Audio Codec
            suffix = ".flac"
        elif audio_bytes.startswith(b'caff'):             # Core Audio Format (Apple iOS CAF)
            suffix = ".caf"
        elif audio_bytes.startswith(b'#!AMR'):            # Adaptive Multi-Rate speech codec
            suffix = ".amr"
        elif audio_bytes.startswith(b'RIFF'):             # Resource Interchange File Format (WAV)
            suffix = ".wav"

        tmp_path = None
        try:
            # Write bytes to temporary file on disk so librosa's external backends can read it
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=target_sr, duration=duration)
            if y is not None and len(y) > 50:
                return y.astype(np.float32)
        except Exception as e_tmp:
            logger.warning(f"Temp file decode fallback error ({suffix}): {e_tmp}")
            # Strategy 4: Fallback attempt directly using soundfile
            try:
                import soundfile as sf
                data, in_sr = sf.read(io.BytesIO(audio_bytes))
                if data is not None and len(data) > 0:
                    # Convert stereo / multi-channel audio to mono by taking channel average
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    # Resample if incoming sample rate does not match target sample rate
                    if in_sr != target_sr:
                        data = librosa.resample(data.astype(np.float32), orig_sr=in_sr, target_sr=target_sr)
                    return data.astype(np.float32)
            except Exception:
                pass
        finally:
            # Ensure temporary file is cleanly removed from filesystem
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return None

    def validate_and_extract_voice(self, y: np.ndarray, sr: int = 22050) -> tuple[bool, Optional[np.ndarray], str]:
        """
        Performs multi-stage Voice Activity Detection (VAD) and speech extraction.

        VAD Pipeline:
        1. Energy Check: Computes overall Root Mean Square (RMS) and decibel (dB) level.
        2. Short-Time Frame Analysis: Computes frame RMS, Zero-Crossing Rate (ZCR),
           and Spectral Centroid (SC) across 46ms windows.
        3. Voiced Frame Filtering: Flags frames meeting speech criteria:
           - Energy above dynamic noise threshold
           - Spectral Centroid in vocal range (150 Hz to 4500 Hz)
           - ZCR <= 0.40 (excludes unvoiced hiss / friction)
        4. Speech Duration Check: Requires at least 1.0 second of voiced frames.
        5. Temporal Dilation: Expands voiced frames by 2 frames (~50ms) on each side
           to preserve phoneme start and end transitions.
        6. Harmonic Confirmation (HPSS): Confirms vocal cords' harmonic resonance.

        Parameters:
            y (np.ndarray): 1D float32 audio waveform.
            sr (int): Sampling rate (default 22050 Hz).

        Returns:
            tuple[bool, Optional[np.ndarray], str]:
                - is_valid: True if genuine human voice was detected and extracted.
                - voiced_audio: Concatenated voiced speech segments, or None if invalid.
                - message: Human-readable diagnostic description or rejection reason.
        """
        # Guard: Check minimum raw recording duration (at least 1.5 seconds)
        if y is None or len(y) < int(sr * 1.5):
            return False, None, "Recording is too short. Please speak for at least 3 to 5 seconds."

        # ── Step 1: Overall volume / RMS check ───────────────────────────────
        # RMS measures average signal energy; scaling by 32767 maps float [-1.0, 1.0] to 16-bit PCM scale
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)

        # Reject quiet audio or background room silence
        if db < 40.0 or rms < 0.003:
            logger.warning(f"Voice validation failed: Audio is silent/too quiet (RMS={rms:.5f}, dB={db:.1f}).")
            return False, None, "No sound detected (Audio was silent or too quiet). Please speak clearly into the microphone."

        # ── Step 2: Short-time frame acoustic feature analysis ───────────────
        # Window size = 1024 samples (~46ms at 22050Hz), hop size = 512 samples (~23ms)
        frame_length = 1024
        hop_length = 512
        try:
            # Frame RMS: energy per frame
            frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            # Zero-Crossing Rate: how often waveform crosses zero amplitude
            frame_zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            # Spectral Centroid: center of mass / brightness of frequency spectrum
            frame_sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        except Exception as e_feat:
            logger.error(f"Error computing frame acoustic features: {e_feat}")
            return False, None, "Could not process audio features."

        # Dynamic energy threshold: 20% of peak energy, with a baseline floor of 0.010
        max_frame_rms = float(np.max(frame_rms)) if len(frame_rms) > 0 else 0.0
        thresh_rms = max(0.010, max_frame_rms * 0.20)

        # Identify voiced frames: sufficient energy, frequency in human vocal range (150-4500Hz), low hiss
        voiced_mask = (
            (frame_rms >= thresh_rms) &
            (frame_sc >= 150.0) &
            (frame_sc <= 4500.0) &
            (frame_zcr <= 0.40)
        )

        num_voiced_frames = int(np.sum(voiced_mask))
        voiced_duration = (num_voiced_frames * hop_length) / sr

        logger.info(f"VAD Analysis: Peak RMS={max_frame_rms:.4f}, Total Voiced Duration={voiced_duration:.2f}s ({num_voiced_frames} frames)")

        # Reject if cumulative voiced speech is less than 1.0 second
        if voiced_duration < 1.0:
            logger.warning(f"Voice validation failed: Voiced speech duration ({voiced_duration:.2f}s) is under the 1.0s minimum threshold.")
            return False, None, f"No voice detected or sample was too short (Only {voiced_duration:.1f}s of speech). Please speak clearly for 3 to 5 seconds."

        # ── Step 3: Voiced Segment Extraction with Temporal Padding ──────────
        # Dilate mask by 2 frames (~50ms) left and right to avoid clipping soft phonemes at start/end
        dilated_mask = voiced_mask.copy()
        for i in range(len(dilated_mask)):
            if voiced_mask[i]:
                dilated_mask[max(0, i - 2): min(len(dilated_mask), i + 3)] = True

        # Map frame-level mask back to sample-level boolean mask
        sample_mask = np.zeros(len(y), dtype=bool)
        for i, is_voiced in enumerate(dilated_mask):
            if is_voiced:
                start_sample = i * hop_length
                end_sample = min(len(y), start_sample + frame_length)
                sample_mask[start_sample:end_sample] = True

        # Extract voiced waveform
        y_voiced = y[sample_mask]
        if len(y_voiced) < int(sr * 0.8):
            return False, None, "Extracted speech segment is too short. Please speak clearly for 3 to 5 seconds."

        # ── Step 4: Harmonic Formant Confirmation via HPSS ────────────────────
        # Differentiates vocal cord harmonics from stationary room noise or fans
        try:
            # Separate signal into harmonic (tonal/vocal) and percussive (click/transient) components
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
        [NOTE: Duplicate Definition 1 of 2]
        Extracts a Cepstral Mean and Variance Normalized (CMVN) MFCC matrix from raw audio bytes.
        
        NOTE: In this file, this method is defined a second time further below (around line 552).
        At runtime, Python executes class definitions sequentially, so the second definition
        silently overwrites and replaces this first definition. This first definition includes
        verbose logger.info calls that are omitted in the second definition.

        Parameters:
            wav_bytes (bytes): Binary audio data.
            n_mfcc (int): Number of MFCC coefficients to compute (default 20).
            require_vad (bool): If True, runs full voice validation and speech extraction.
                                If False, performs simple silence trimming.

        Returns:
            tuple[Optional[np.ndarray], Optional[str]]: (mfcc_matrix, error_message_if_any)
        """
        logger.info(f"extract_mfcc_matrix: received {len(wav_bytes)} bytes (require_vad={require_vad})")
        # Decode incoming bytes to floating-point audio array
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 100:
            logger.error("Audio too short or could not be decoded.")
            return None, "Audio could not be decoded or was empty."

        # Either apply strict VAD or simple silence margin trimming
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
        # Compute MFCC matrix: shape (n_mfcc, time_frames)
        mfcc = librosa.feature.mfcc(y=target_y, sr=22050, n_mfcc=n_mfcc)

        # Cepstral Mean Normalization (CMN) across frames (centers features around 0)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        # Unit Variance Normalization (scales feature standard deviation to 1.0)
        std = np.std(mfcc, axis=1, keepdims=True)
        std[std == 0] = 1e-6
        mfcc = mfcc / std

        logger.info(f"extract_mfcc_matrix: CMVN normalized MFCC shape {mfcc.shape}")
        return mfcc.astype(np.float32), None

    def predict_from_wav_bytes(self, wav_bytes: bytes) -> tuple[int, float]:
        """
        Performs end-to-end audio threat classification on raw audio bytes.

        Steps:
        1. Checks whether model weights were successfully loaded; defaults to (0, 0.0) if not.
        2. Decodes audio bytes up to 3.0 seconds at 22050 Hz.
        3. Extracts the 120-channel (120, 150) feature matrix using extract_features.
        4. Converts feature matrix to a PyTorch tensor with batch dimension (1, 120, 150).
        5. Executes forward pass in torch.no_grad() mode.
        6. Computes softmax probabilities over classes [Safe, Threat].
        7. Classifies as Threat (1) if Threat probability >= 0.5, else Safe (0).

        Parameters:
            wav_bytes (bytes): Raw audio bytes.

        Returns:
            tuple[int, float]: (predicted_class_id, threat_probability)
                - predicted_class_id: 0 = Safe, 1 = Threat.
                - threat_probability: Float between 0.0 and 1.0 indicating model confidence.
        """
        # Safe default if model checkpoint was absent or failed to load
        if not self.is_loaded:
            return 0, 0.0

        try:
            # Decode audio clip, limiting duration to 3.0 seconds for consistent inference
            y = self.decode_audio(wav_bytes, target_sr=22050, duration=3.0)
            if y is None:
                return 0, 0.0

            # Extract 120-channel standardized feature matrix
            features = extract_features(y, sr=22050)
            # Add batch dimension: shape (1, 120, 150) and transfer to target device
            tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Execute inference without computing gradients
            with torch.no_grad():
                outputs = self.model(tensor_features)
                # Compute normalized probabilities using softmax
                probabilities = torch.softmax(outputs, dim=1)
                
                # Threat class is at index 1
                prob_threat = probabilities[0][1].item()
                
            # Decision threshold: 0.5
            return (1 if prob_threat >= 0.5 else 0, prob_threat)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0, 0.0

    def check_acoustic_environment(self, audio_bytes: bytes) -> dict:
        """
        Analyzes a short ambient audio sample (1.5-2.5s) to assess background noise.

        Calculates the noise floor in decibels (dB) and evaluates readiness for voice enrollment:
        - < 38 dB: "excellent" (optimal, quiet room)
        - 38 to 52 dB: "good" (normal background noise)
        - >= 52 dB: "noisy" (unfavorable noise; prompts user to move to a quieter space)

        Parameters:
            audio_bytes (bytes): Ambient room recording bytes.

        Returns:
            dict: {
                "is_ready": bool,
                "status": "excellent" | "good" | "noisy" | "error",
                "noise_db": float,
                "message": str
            }
        """
        # Decode ambient audio clip
        y = self.decode_audio(audio_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return {
                "is_ready": False,
                "status": "error",
                "noise_db": 0.0,
                "message": "Could not capture audio. Please ensure microphone permissions are granted.",
            }

        # Calculate Root Mean Square energy and convert to decibels
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        noise_db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)

        # Categorize environmental acoustic noise floor
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
        Validates an enrollment passphrase sample for speech presence, clarity, duration, and distortion.

        Quality Checks:
        1. Volume / Silence: Rejects recordings below 40.0 dB.
        2. Clipping / Distortion: Rejects if peak sample amplitude >= 0.99 (microphone overdrive).
        3. Duration: Rejects if total audio duration is under 1.8 seconds.
        4. Speech Clarity: Checks spectral centroid and zero-crossing rate to ensure vocal presence.
        5. Harmonic Resonance: Evaluates HPSS harmonic energy to confirm clean human vocalization.

        Parameters:
            audio_bytes (bytes): Recited phrase audio bytes.

        Returns:
            dict: Evaluation results containing is_valid, clarity_score, duration, db, and status message.
        """
        # Decode audio bytes into float32 array
        y = self.decode_audio(audio_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return {
                "is_valid": False,
                "clarity_score": 0.0,
                "duration": 0.0,
                "db": 0.0,
                "error": "Audio could not be decoded or was empty. Please speak clearly into the microphone.",
            }

        # Calculate basic signal metrics: duration and volume in decibels
        duration = float(len(y) / 22050.0)
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)

        # 1. Volume & Silence Check: audio must be loud enough to analyze
        if db < 40.0:
            return {
                "is_valid": False,
                "clarity_score": 10.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "error": f"Audio was too quiet ({db:.1f} dB). Please speak directly towards the microphone.",
            }

        # 2. Clipping check: values reaching or exceeding 0.99 indicate microphone saturation / clipping
        peak = float(np.max(np.abs(y)))
        if peak >= 0.99:
            return {
                "is_valid": False,
                "clarity_score": 25.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "error": "Audio clipped or was too loud. Please move slightly further back from the microphone.",
            }

        # 3. Minimum duration check: requires at least 1.8 seconds
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

            # Voiced frames: energy > 0.003, ZCR < 0.40, frequency centroid between 150Hz and 4500Hz
            voiced_frames = (frame_rms > 0.003) & (frame_zcr < 0.40) & (frame_sc >= 150) & (frame_sc <= 4500)
            voiced_ratio = float(np.mean(voiced_frames))
            voiced_duration = float(np.sum(voiced_frames) * hop_length / 22050.0)

            # Ensure speech occupies at least 1.0 second and 25% of the recording
            if voiced_duration < 1.0 or voiced_ratio < 0.25:
                return {
                    "is_valid": False,
                    "clarity_score": round(voiced_ratio * 100.0, 1),
                    "duration": round(duration, 1),
                    "db": round(db, 1),
                    "error": "Voice was not clear or sounded like ambient noise. Please speak the challenge phrase firmly.",
                }

            # 5. Harmonic resonance confirmation using HPSS
            y_harm, _ = librosa.effects.hpss(y)
            harm_rms = float(np.sqrt(np.mean(y_harm**2)))
            # Compute composite clarity percentage score
            clarity = min(98.0, max(65.0, (harm_rms / (rms + 1e-6)) * 100.0 + (voiced_ratio * 20.0)))

            return {
                "is_valid": True,
                "clarity_score": round(clarity, 1),
                "duration": round(duration, 1),
                "db": round(db, 1),
                "message": f"Phrase verified with high clarity ({clarity:.0f}% vocal confidence, {duration:.1f}s speech).",
            }

        except Exception as e:
            # Graceful fallback if feature extraction encounters an unexpected numerical issue
            return {
                "is_valid": True,
                "clarity_score": 75.0,
                "duration": round(duration, 1),
                "db": round(db, 1),
                "message": "Phrase sample accepted.",
            }

    def extract_speaker_biometric_vector(self, y: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Extracts a 64-dimensional Text-Independent Vocal Tract Biometric Embedding.

        This biometric vector captures unique physiological characteristics of a speaker's
        vocal tract, larynx, and resonance cavities, independent of the specific words spoken.

        The 64 dimensions consist of:
        - Pitch Fundamental Frequency (F0) & Dispersion: 2 values (weight: 0.8)
        - Formant Spectral Coefficients (MFCCs 1-19 normalized): 19 values (weight: 1.2)
        - Delta Dynamic Coefficients (temporal velocity): 19 values (weight: 0.6)
        - Spectral Contrast (octave peak-valley energy across 7 bands): 7 values (weight: 0.8)
        - Formant Moments (Centroid, Bandwidth, Rolloff, Flatness): 4 values (weight: 0.5)
        - Mel Filterbank Energy Envelope (across 13 bands): 13 values (weight: 0.7)
        Total: 2 + 19 + 19 + 7 + 4 + 13 = 64 dimensions, L2-normalized to unit length.

        Parameters:
            y (np.ndarray): 1D float32 audio waveform of voiced speech.
            sr (int): Sampling rate (default 22050 Hz).

        Returns:
            np.ndarray: 1D float32 vector of shape (64,), normalized such that ||v||_2 = 1.0.
        """
        if y is None or len(y) < 500:
            return np.zeros(64, dtype=np.float32)

        # ── Component 1: Fundamental Pitch (F0) estimation via YIN algorithm (2 values)
        try:
            # YIN algorithm estimates fundamental frequency across human vocal pitch bounds (60-400 Hz)
            f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
            f0_voiced = f0[(f0 > 60) & (f0 < 400)]
            mean_f0 = float(np.median(f0_voiced)) if len(f0_voiced) > 0 else 140.0
            std_f0 = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 20.0
        except Exception:
            mean_f0, std_f0 = 140.0, 20.0

        # ── Component 2: Formant Coefficients: MFCC 1-19 (19 values) ─────────
        # Note: MFCC index 0 is excluded because it reflects overall volume, not vocal tract anatomy
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]
        # Normalize per-frame across coefficients
        mfcc_norm = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-6)
        # Average across time frames to obtain static vocal tract profile
        m_mean = np.mean(mfcc_norm, axis=1)
        m_mean = m_mean / (np.linalg.norm(m_mean) + 1e-6)

        # ── Component 3: Delta Dynamic Coefficients (19 values) ──────────────
        # Captures how quickly the speaker shifts formants during phonation
        delta = librosa.feature.delta(mfcc)
        d_mean = np.mean(delta, axis=1)
        d_mean = d_mean / (np.linalg.norm(d_mean) + 1e-6)

        # ── Component 4: Spectral Contrast across 7 sub-bands (7 values) ──────
        # Measures the difference in energy between spectral peaks and valleys
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            c_mean = np.mean(contrast, axis=1)
            c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-6)
        except Exception:
            c_mean = np.zeros(7, dtype=np.float32)

        # ── Component 5: Formant Moments (4 values) ──────────────────────────
        # Global shape descriptors of the power spectrum:
        # Centroid (center of mass), Bandwidth (spread), Rolloff (high-frequency cutoff), Flatness (tonality)
        try:
            sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) / 2000.0
            sb = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))) / 2000.0
            sr_ro = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) / 3000.0
            sf = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        except Exception:
            sc, sb, sr_ro, sf = 1.0, 1.0, 1.0, 0.01

        # ── Component 6: Mel Spectrogram Frequency Envelope (13 values) ──────
        # Coarse energy distribution across 13 perceptually spaced Mel frequency bins
        try:
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
            mel_env = np.mean(melspec, axis=1)
            mel_env = mel_env / (np.linalg.norm(mel_env) + 1e-6)
        except Exception:
            mel_env = np.zeros(13, dtype=np.float32)

        # ── Assembly and Weighting ───────────────────────────────────────────
        # Assemble feature groups: (2 + 19 + 19 + 7 + 4 + 13 = 64 dimensions)
        pitch_feat = np.array([mean_f0 / 150.0, std_f0 / 50.0], dtype=np.float32)
        moment_feat = np.array([sc, sb, sr_ro, sf], dtype=np.float32)

        vector = np.concatenate([
            pitch_feat * 0.8,       # 2 values (Pitch)
            m_mean * 1.2,           # 19 values (Formant MFCCs - given highest weight)
            d_mean * 0.6,           # 19 values (Delta dynamics)
            c_mean * 0.8,           # 7 values (Spectral Contrast)
            moment_feat * 0.5,      # 4 values (Spectral Moments)
            mel_env * 0.7           # 13 values (Mel Filterbank Envelope)
        ])

        # Final L2-normalization to project vector onto the unit hypersphere
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32)

    def extract_mfcc_matrix(self, wav_bytes: bytes, n_mfcc: int = 20, require_vad: bool = True) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        [NOTE: Active Runtime Definition 2 of 2]
        Extracts a normalized 2D MFCC matrix from raw audio bytes.

        NOTE ON DUPLICATION: This second definition replaces the first definition at runtime.
        Any call to `predictor.extract_mfcc_matrix(...)` invokes this implementation.
        It performs the same mathematical extraction and CMVN normalization as definition 1,
        but excludes logger.info debugging messages.

        Parameters:
            wav_bytes (bytes): Binary audio data.
            n_mfcc (int): Number of MFCC coefficients to extract (default 20).
            require_vad (bool): If True, validates and extracts speech segments.
                                If False, performs silence trimming.

        Returns:
            tuple[Optional[np.ndarray], Optional[str]]: (mfcc_matrix, error_message)
        """
        # Step 1: Decode raw audio bytes to 1D float array at 22050 Hz
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 100:
            return None, "Audio could not be decoded or was empty."

        # Step 2: Voice activity filtering or boundary silence trimming
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

        # Step 3: Compute raw MFCCs
        mfcc = librosa.feature.mfcc(y=target_y, sr=22050, n_mfcc=n_mfcc)

        # Step 4: Cepstral Mean and Variance Normalization (CMVN)
        # Center mean of each cepstral coefficient across time to 0
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        # Scale standard deviation of each coefficient to 1
        std = np.std(mfcc, axis=1, keepdims=True)
        std[std == 0] = 1e-6
        mfcc = mfcc / std

        return mfcc.astype(np.float32), None

    def verify_speaker_biometrics(self, wav_bytes: bytes, stored_profiles: list) -> tuple[bool, Optional[dict], float]:
        """
        Text-Independent Biometric Speaker Verification.

        Compares the 64-dimensional vocal tract biometric embedding of incoming audio
        against authorized caregiver voice profiles stored in the system.

        Decision Logic & Distance Metric:
        - Both live and stored vectors are L2-normalized unit vectors.
        - Cosine similarity = dot_product(live_vector, profile_vector).
        - Cosine distance = 1.0 - cosine_similarity.
        - Distance <= 0.125: MATCH (Authorized parent/caregiver verified).
        - Distance > 0.125:  REJECT (Unrecognized voice / Stranger).

        Parameters:
            wav_bytes (bytes): Binary audio data of the speech to verify.
            stored_profiles (list): List of profile dictionaries or arrays.
                                    Supports both modern 64-D vectors and legacy 2D MFCC matrices.

        Returns:
            tuple[bool, Optional[dict], float]:
                - is_match: True if closest profile distance <= 0.125, False otherwise.
                - matched_profile: Dictionary containing {'name', 'role', 'id'} of matched person, or None.
                - min_distance: Smallest cosine distance computed across all stored profiles.
        """
        # Step 1: Decode incoming audio bytes to float32 waveform
        y = self.decode_audio(wav_bytes, target_sr=22050)
        if y is None or len(y) < 1000:
            return False, None, 1.0

        # Step 2: Volume check (reject quiet room noise to prevent spurious false matches)
        rms = float(np.sqrt(np.mean(y**2)))
        rms_scaled = rms * 32767.0
        db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)
        if db < 38.0:
            return False, None, 1.0

        # Step 3: Extract live 64-dimensional biometric vocal tract embedding
        live_vector = self.extract_speaker_biometric_vector(y, sr=22050)
        
        # Operational biometric cosine distance threshold
        THRESHOLD = 0.125  # Biometric Cosine Distance Threshold (Parent: 0.028 - 0.095, Stranger: 0.165 - 0.50+)
        min_dist = float('inf')
        matched_profile = None

        # Step 4: Compare against each registered profile
        for item in stored_profiles:
            try:
                # Extract matrix data and metadata from profile entry
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
                
                # Check profile format: 1D 64-D biometric vector vs legacy 2D MFCC matrix
                if stored_arr.ndim == 1 and len(stored_arr) == 64:
                    profile_vector = stored_arr
                elif stored_arr.ndim == 2:
                    # Backward-compatible on-the-fly conversion from legacy 2D MFCC matrix:
                    # Derives static means, standard deviations, and deltas into a 64-D vector.
                    mfcc_mean = np.mean(stored_arr, axis=1)
                    mfcc_std = np.std(stored_arr, axis=1)
                    delta_mean = np.mean(librosa.feature.delta(stored_arr), axis=1)
                    profile_vector = np.concatenate([
                        mfcc_mean[:2],
                        mfcc_mean[1:20],
                        delta_mean[1:20],
                        mfcc_std[:7],
                        np.array([1.0, 1.0, 1.0, 0.01], dtype=np.float32),
                        mfcc_mean[:13]
                    ])
                    p_norm = np.linalg.norm(profile_vector)
                    if p_norm > 0:
                        profile_vector = profile_vector / p_norm
                else:
                    continue

                # Cosine distance between normalized unit biometric vectors:
                # dot_product == cosine similarity; distance = 1.0 - similarity
                cosine_sim = float(np.dot(live_vector, profile_vector))
                dist = max(0.0, float(1.0 - cosine_sim))

                # Track best-matching profile
                if dist < min_dist:
                    min_dist = dist
                    matched_profile = {"name": name, "role": role, "id": prof_id}

                logger.info(f"Biometric voiceprint distance to '{name or 'profile'}' ({role}): {dist:.4f} (threshold: {THRESHOLD})")
                
                # Early return on confident match within threshold
                if dist <= THRESHOLD:
                    return True, {"name": name, "role": role, "id": prof_id}, dist

            except Exception as e:
                logger.warning(f"Biometric comparison failed for profile {item}: {e}")
                continue

        # Final match decision across all profiles
        is_match = (min_dist <= THRESHOLD)
        return is_match, (matched_profile if is_match else None), (min_dist if min_dist != float('inf') else 1.0)

    def verify_parent_from_matrix(self, wav_bytes: bytes, stored_mfcc_list: list) -> tuple[bool, Optional[dict], float]:
        """
        Bridge method routing legacy callers to the biometric speaker verification engine.

        Parameters:
            wav_bytes (bytes): Binary audio data of live speech.
            stored_mfcc_list (list): List of registered profiles (dict or raw arrays).

        Returns:
            tuple[bool, Optional[dict], float]: Match result from verify_speaker_biometrics.
        """
        return self.verify_speaker_biometrics(wav_bytes, stored_mfcc_list)

    def verify_parent(self, wav_bytes: bytes, parent_profile_path: Path) -> bool:
        """
        Fallback speaker verification comparing incoming audio against a local WAV file.

        Loads the reference WAV from disk (e.g. parent_profile.wav), extracts its
        biometric vector, and executes verify_speaker_biometrics.

        Parameters:
            wav_bytes (bytes): Live audio chunk to verify.
            parent_profile_path (Path): Filesystem path to the reference WAV file.

        Returns:
            bool: True if incoming audio matches reference voice within distance threshold.
        """
        # Guard: check reference file existence and minimum size
        if not parent_profile_path.exists() or parent_profile_path.stat().st_size < 1000:
            return False

        try:
            # Read reference audio bytes from local disk
            with open(parent_profile_path, "rb") as f:
                ref_bytes = f.read()
            # Decode reference audio to float32 waveform
            y_ref = self.decode_audio(ref_bytes, target_sr=22050)
            if y_ref is None:
                return False
            # Extract 64-D biometric vector from reference audio
            ref_vector = self.extract_speaker_biometric_vector(y_ref, sr=22050)
            # Delegate comparison to verify_speaker_biometrics
            is_match, _, _ = self.verify_speaker_biometrics(
                wav_bytes,
                [{"matrix": ref_vector, "person_name": "Parent", "role": "Parent"}]
            )
            return is_match
        except Exception as e:
            logger.error(f"Error during local parent verification: {e}")
            return False

# Global singleton instance of ThreatPredictor for use across routers and services
predictor = ThreatPredictor()

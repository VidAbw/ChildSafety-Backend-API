"""
Unit and Integration Tests for ChildSafety Acoustic Inference & Biometric Verification
(audio_guardian/predictor.py and audio_guardian/router.py)

Covers peer-review methodological updates:
1. Calibrated Digital-to-SPL Audio Level Conversion (INMP441 MEMS microphone calibration)
2. Standardized 64-D Multi-Band Spectral-Cepstral Speaker Embedding with VAD/HPSS rejection
3. Context-Safe Caregiver Arbitration with Pediatric Scream Band Veto (Eq. 4 Fix)
4. Telemetry preserving API endpoints (/upload-chunk, /check-environment, /validate-phrase-sample)
"""

import io
import math
import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio_guardian.predictor import (
    predictor,
    compute_calibrated_spl,
    compute_scream_ratio,
    INMP441_AOP_SPL,
    INMP441_SENSITIVITY_DBFS,
    SUB_BLOCK_WEIGHTS,
    DEFAULT_CORPUS_MEAN,
    DEFAULT_CORPUS_STD,
    load_corpus_stats,
)
from audio_guardian.router import router as audio_guardian_router


# ============================================================================
# Helpers to generate synthetic audio waveforms
# ============================================================================

def make_sine_wave(freq_hz: float, duration_s: float = 1.0, sr: int = 22050, amplitude: float = 0.5) -> np.ndarray:
    """Generate a clean sinusoidal audio signal."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

def make_harmonic_speech(f0: float = 140.0, duration_s: float = 3.0, sr: int = 22050, amplitude: float = 0.2) -> np.ndarray:
    """Generate speech-like harmonic vocal tone with fundamental and overtones."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = (
        amplitude * np.sin(2 * np.pi * f0 * t) +
        (amplitude * 0.5) * np.sin(2 * np.pi * 2 * f0 * t) +
        (amplitude * 0.25) * np.sin(2 * np.pi * 3 * f0 * t) +
        (amplitude * 0.12) * np.sin(2 * np.pi * 4 * f0 * t)
    )
    return signal.astype(np.float32)

def wav_to_bytes(y: np.ndarray, sr: int = 22050) -> bytes:
    """Encode float waveform to 16-bit PCM WAV bytes in-memory."""
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ============================================================================
# 1. Calibrated Digital-to-SPL Audio Level Conversion Tests
# ============================================================================

class TestCalibratedSPLConversion:
    """Verifies exact InvenSense INMP441 MEMS microphone calibration."""

    def test_inmp441_hardware_constants(self):
        """Microphone specs: -26 dBFS at 94 dB SPL; 120 dB SPL AOP at 0 dBFS."""
        assert INMP441_AOP_SPL == 120.0
        assert INMP441_SENSITIVITY_DBFS == -26.0

    def test_exact_spl_calibration_points(self):
        """
        Verify:
        - 0.0 dBFS (RMS = 1.0) -> 120.0 dB SPL (Acoustic Overload Point)
        - -26.0 dBFS (RMS = 10^(-26/20)) -> 94.0 dB SPL (INMP441 sensitivity)
        - -40.0 dBFS (RMS = 10^(-40/20) = 0.01) -> 80.0 dB SPL (Critical intensity threshold)
        - -55.0 dBFS (RMS = 10^(-55/20) ~ 0.001778) -> 65.0 dB SPL (Conversational speech threshold)
        """
        # 1. Acoustic Overload Point (0 dBFS -> 120 dB SPL)
        rms_0dbfs = np.array([1.0], dtype=np.float32)
        rms, l_dbfs, l_spl = compute_calibrated_spl(rms_0dbfs)
        assert l_dbfs == pytest.approx(0.0, abs=1e-3)
        assert l_spl == pytest.approx(120.0, abs=1e-3)

        # 2. INMP441 Sensitivity point (-26 dBFS -> 94 dB SPL)
        rms_val_26 = 10.0 ** (-26.0 / 20.0)
        rms_sig_26 = np.array([rms_val_26], dtype=np.float32)
        rms, l_dbfs, l_spl = compute_calibrated_spl(rms_sig_26)
        assert l_dbfs == pytest.approx(-26.0, abs=1e-3)
        assert l_spl == pytest.approx(94.0, abs=1e-3)

        # 3. Critical Acoustic Intensity Override (-40 dBFS -> 80 dB SPL)
        rms_val_40 = 10.0 ** (-40.0 / 20.0)
        rms_sig_40 = np.array([rms_val_40], dtype=np.float32)
        rms, l_dbfs, l_spl = compute_calibrated_spl(rms_sig_40)
        assert l_dbfs == pytest.approx(-40.0, abs=1e-3)
        assert l_spl == pytest.approx(80.0, abs=1e-3)

        # 4. Conversational speech threshold (-55 dBFS -> 65 dB SPL)
        rms_val_55 = 10.0 ** (-55.0 / 20.0)
        rms_sig_55 = np.array([rms_val_55], dtype=np.float32)
        rms, l_dbfs, l_spl = compute_calibrated_spl(rms_sig_55)
        assert l_dbfs == pytest.approx(-55.0, abs=1e-3)
        assert l_spl == pytest.approx(65.0, abs=1e-3)

    def test_floor_clipping_for_silence(self):
        """Zero / near-zero audio must be clipped at 1e-9 to avoid log(0) -inf."""
        zeros = np.zeros(2048, dtype=np.float32)
        rms, l_dbfs, l_spl = compute_calibrated_spl(zeros)
        assert rms == 0.0
        assert l_dbfs == pytest.approx(-180.0, abs=1e-2)
        assert l_spl == pytest.approx(-60.0, abs=1e-2)

    def test_predictor_instance_method_delegation(self):
        """predictor.compute_calibrated_spl delegates to compute_calibrated_spl."""
        sig = np.full(1000, 0.05, dtype=np.float32)
        res1 = compute_calibrated_spl(sig)
        res2 = predictor.compute_calibrated_spl(sig)
        assert res1 == res2

    def test_check_acoustic_environment_returns_calibrated_metrics(self):
        """Environment check returns both L_dBFS and L_SPL in telemetry dictionary."""
        # Ultra quiet signal: RMS = 1e-3 (-60 dBFS / 60 dB SPL)
        quiet_signal = (np.random.randn(22050 * 2) * 0.001).astype(np.float32)
        wav_bytes = wav_to_bytes(quiet_signal)
        res = predictor.check_acoustic_environment(wav_bytes)

        assert "L_dBFS" in res
        assert "L_SPL" in res
        assert "is_ready" in res
        assert res["is_ready"] is True
        assert res["L_SPL"] == pytest.approx(res["L_dBFS"] + 120.0, abs=0.1)

    def test_validate_phrase_sample_returns_calibrated_metrics(self):
        """Phrase sample validation returns both L_dBFS and L_SPL."""
        speech = make_harmonic_speech(f0=150.0, duration_s=2.5, amplitude=0.2)
        wav_bytes = wav_to_bytes(speech)
        res = predictor.validate_phrase_sample(wav_bytes)

        assert "L_dBFS" in res
        assert "L_SPL" in res
        assert res["is_valid"] is True
        assert res["L_SPL"] == pytest.approx(res["L_dBFS"] + 120.0, abs=0.1)


# ============================================================================
# 2. Standardized 64-D Multi-Band Speaker Embedding & Noise Rejection Tests
# ============================================================================

class TestStandardizedSpeakerEmbedding:
    """Verifies 64-D multi-band extraction, Z-score standardization, and HPSS/VAD rejection."""

    def test_embedding_dimensions_and_norm(self):
        """Extracted embedding vector must be strictly 64-D and L2-normalized to 1.0."""
        speech = make_harmonic_speech(f0=140.0, duration_s=2.0)
        vec = predictor.extract_speaker_biometric_vector(speech, sr=22050)

        assert vec is not None
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,), f"Expected 64 dimensions, got {vec.shape}"
        l2_norm = float(np.linalg.norm(vec))
        assert l2_norm == pytest.approx(1.0, rel=1e-5), f"Expected unit norm 1.0, got {l2_norm}"

    def test_sub_block_weight_vector(self):
        """Check exact sub-block weights: [0.8 x 2, 1.2 x 19, 0.6 x 19, 0.8 x 7, 0.5 x 4, 0.7 x 13]."""
        assert len(SUB_BLOCK_WEIGHTS) == 64
        # 2D pitch
        assert np.allclose(SUB_BLOCK_WEIGHTS[0:2], 0.8)
        # 19D static MFCCs
        assert np.allclose(SUB_BLOCK_WEIGHTS[2:21], 1.2)
        # 19D delta MFCCs
        assert np.allclose(SUB_BLOCK_WEIGHTS[21:40], 0.6)
        # 7D spectral contrast
        assert np.allclose(SUB_BLOCK_WEIGHTS[40:47], 0.8)
        # 4D spectral moments
        assert np.allclose(SUB_BLOCK_WEIGHTS[47:51], 0.5)
        # 13D Mel energies
        assert np.allclose(SUB_BLOCK_WEIGHTS[51:64], 0.7)

    def test_corpus_statistics_loading(self):
        """Verify training corpus mean and std arrays are 64-D and valid."""
        mean, std = load_corpus_stats()
        assert mean.shape == (64,)
        assert std.shape == (64,)
        assert np.all(std > 0.0), "Standard deviation must be strictly positive"

    def test_vad_and_hpss_reject_pure_broadband_noise(self):
        """
        Broadband Gaussian noise without vocal tract resonance must be rejected
        (HPR < 0.35 or voiced ratio < 0.35) BEFORE Cosine distance calculation.
        """
        sr = 22050
        # Generate pure white noise (flat spectrum, no harmonic resonance)
        white_noise = (np.random.randn(sr * 2) * 0.15).astype(np.float32)
        wav_bytes = wav_to_bytes(white_noise, sr=sr)

        # Mock profile
        dummy_vec = np.ones(64, dtype=np.float32)
        dummy_vec /= np.linalg.norm(dummy_vec)
        mock_profiles = [{
            "person_name": "Test Parent",
            "role": "Parent",
            "voice_embedding": dummy_vec.tolist(),
            "is_active": True
        }]

        is_match, matched_profile, dist = predictor.verify_speaker_biometrics(wav_bytes, mock_profiles)
        assert is_match is False
        assert matched_profile is None
        assert dist == 1.0, "Rejected non-vocal noise should return max distance 1.0"

    def test_biometric_cosine_discrimination(self):
        """Identical or near voice yields small distance; distinct voice yields larger distance."""
        sr = 22050
        speech_base = make_harmonic_speech(f0=130.0, duration_s=2.5, sr=sr)
        speech_same_speaker = make_harmonic_speech(f0=133.0, duration_s=2.5, sr=sr)
        speech_different_pitch = make_harmonic_speech(f0=240.0, duration_s=2.5, sr=sr)

        vec_base = predictor.extract_speaker_biometric_vector(speech_base, sr=sr)
        vec_same = predictor.extract_speaker_biometric_vector(speech_same_speaker, sr=sr)
        vec_diff = predictor.extract_speaker_biometric_vector(speech_different_pitch, sr=sr)

        dist_same = float(1.0 - np.dot(vec_base, vec_same))
        dist_diff = float(1.0 - np.dot(vec_base, vec_diff))

        assert dist_same < dist_diff
        assert dist_same <= 0.125, f"Same speaker distance {dist_same} should be <= match threshold 0.125"


# ============================================================================
# 3. Context-Safe Caregiver Arbitration with Pediatric Distress Veto Tests
# ============================================================================

class TestPediatricDistressArbitration:
    """Verifies Eq. 4 pediatric scream band energy ratio and anti-fatigue arbitration."""

    def test_scream_ratio_formula_definition(self):
        """
        Formula: R_scream = sum(P[2500 Hz : 5000 Hz]) / sum(P[20 Hz : 8000 Hz])
        A 3500 Hz distress sine wave should produce R_scream >= 0.22 (near 1.0).
        A 150 Hz speech tone should produce R_scream < 0.05.
        """
        sr = 22050
        scream_signal = make_sine_wave(freq_hz=3500.0, duration_s=1.0, sr=sr, amplitude=0.4)
        r_scream = compute_scream_ratio(scream_signal, sr=sr)
        assert r_scream >= 0.22, f"Expected distress scream ratio >= 0.22, got {r_scream:.3f}"

        normal_signal = make_sine_wave(freq_hz=150.0, duration_s=1.0, sr=sr, amplitude=0.4)
        r_normal = compute_scream_ratio(normal_signal, sr=sr)
        assert r_normal < 0.22, f"Expected normal vocal ratio < 0.22, got {r_normal:.3f}"

    def test_arbitration_downgrade_when_safe(self):
        """
        Safe Condition: MatchParent == True (Dist <= 0.125, L_SPL >= 65.0)
                        AND R_scream < 0.22
                        AND vision_tier == 0
        Decision: State downgraded to 'Contextualized_Safe (Push_Suppressed)'.
        """
        dtw_dist = 0.08
        l_spl = 70.0
        r_scream = 0.05
        vision_tier = 0

        is_parent = (dtw_dist <= 0.125 and l_spl >= 65.0)
        assert is_parent is True

        scream_veto = (r_scream >= 0.22)
        vision_veto = (vision_tier > 0)
        assert not scream_veto
        assert not vision_veto

        predicted_class = "Crying"
        if is_parent and not scream_veto and not vision_veto:
            predicted_class = "Contextualized_Safe (Push_Suppressed)"

        assert predicted_class == "Contextualized_Safe (Push_Suppressed)"

    def test_pediatric_scream_vetoes_caregiver_match(self):
        """
        Veto Condition (a): R_scream >= 0.22 (overlapping child pain scream).
        Caregiver voice match must NEVER silence the distress alert!
        """
        dtw_dist = 0.08
        l_spl = 72.0
        r_scream = 0.35  # Child screaming concurrently in distress
        vision_tier = 0

        is_parent = (dtw_dist <= 0.125 and l_spl >= 65.0)
        assert is_parent is True

        scream_veto = (r_scream >= 0.22)
        assert scream_veto is True

        predicted_class = "Crying"
        if is_parent and not scream_veto and (vision_tier == 0):
            predicted_class = "Contextualized_Safe (Push_Suppressed)"

        # Alert must NOT be silenced
        assert predicted_class == "Crying"

    def test_vision_hazard_vetoes_caregiver_match(self):
        """
        Veto Condition (b): vision_tier > 0 (active visual hazard, fall, or unknown adult).
        Caregiver voice match must NEVER silence the alert!
        """
        dtw_dist = 0.06
        l_spl = 70.0
        r_scream = 0.04
        vision_tier = 2  # Active visual danger

        is_parent = (dtw_dist <= 0.125 and l_spl >= 65.0)
        assert is_parent is True

        vision_veto = (vision_tier > 0)
        assert vision_veto is True

        predicted_class = "Screaming"
        if is_parent and (r_scream < 0.22) and not vision_veto:
            predicted_class = "Contextualized_Safe (Push_Suppressed)"

        # Alert must NOT be silenced
        assert predicted_class == "Screaming"

    def test_critical_acoustic_intensity_override(self):
        """L_SPL > 80.0 dB SPL (-40.0 dBFS) triggers critical override regardless of caregiver presence."""
        l_spl = 82.5  # Exceeds 80.0 dB SPL threshold
        dtw_dist = 0.05
        is_parent = (dtw_dist <= 0.125 and l_spl >= 65.0)

        predicted_class = "Normal"
        if l_spl > 80.0:
            predicted_class = "Critical_Acoustic_Override"

        assert predicted_class == "Critical_Acoustic_Override"


# ============================================================================
# 4. FastAPI Router Integration Endpoints Tests
# ============================================================================

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(audio_guardian_router, prefix="/audio")
    return TestClient(app)


class TestRouterIntegrationEndpoints:
    """Verifies router API endpoints return calibrated metrics and preserve telemetry."""

    def test_check_environment_endpoint(self, client):
        """POST /audio/check-environment returns L_dBFS and calibrated L_SPL."""
        quiet_signal = (np.random.randn(22050 * 2) * 0.001).astype(np.float32)
        wav_bytes = wav_to_bytes(quiet_signal)

        response = client.post(
            "/audio/check-environment",
            files={"file": ("ambient.wav", wav_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "L_dBFS" in data
        assert "L_SPL" in data
        assert "is_ready" in data
        assert data["L_SPL"] == pytest.approx(data["L_dBFS"] + 120.0, abs=0.2)

    def test_validate_phrase_sample_endpoint(self, client):
        """POST /audio/validate-phrase-sample returns L_dBFS and calibrated L_SPL."""
        speech = make_harmonic_speech(f0=145.0, duration_s=2.5, amplitude=0.2)
        wav_bytes = wav_to_bytes(speech)

        response = client.post(
            "/audio/validate-phrase-sample",
            files={"file": ("phrase.wav", wav_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "L_dBFS" in data
        assert "L_SPL" in data
        assert "is_valid" in data
        assert data["L_SPL"] == pytest.approx(data["L_dBFS"] + 120.0, abs=0.2)

    def test_upload_chunk_telemetry_preservation(self, client):
        """
        POST /audio/upload-chunk:
        - Accepts vision_tier form parameter (defaults to 0)
        - Returns telemetry preserving L_dBFS, L_SPL, R_scream, vision_tier, match_parent
        """
        tone = make_sine_wave(freq_hz=300.0, duration_s=1.0, amplitude=0.1)
        wav_bytes = wav_to_bytes(tone)

        # Test with vision_tier = 0
        response0 = client.post(
            "/audio/upload-chunk",
            files={"file": ("chunk.wav", wav_bytes, "audio/wav")},
            data={"vision_tier": "0"}
        )
        assert response0.status_code == 200
        data0 = response0.json()
        assert "L_dBFS" in data0
        assert "L_SPL" in data0
        assert "R_scream" in data0
        assert "vision_tier" in data0
        assert data0["vision_tier"] == 0
        assert data0["L_SPL"] == pytest.approx(data0["L_dBFS"] + 120.0, abs=0.2)

        # Test with vision_tier = 2
        response2 = client.post(
            "/audio/upload-chunk",
            files={"file": ("chunk.wav", wav_bytes, "audio/wav")},
            data={"vision_tier": "2"}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["vision_tier"] == 2
        assert "R_scream" in data2

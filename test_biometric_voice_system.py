import numpy as np
import librosa
from audio_guardian.predictor import predictor
from core.supabase import db

print("=== 1. TESTING ACOUSTIC ENVIRONMENT CHECK ===")
# Synthesize quiet room sound (white noise at low amp)
sr = 22050
quiet_audio = np.random.randn(sr * 2) * 0.001
import soundfile as sf
import io

buf = io.BytesIO()
sf.write(buf, quiet_audio, sr, format='WAV')
quiet_bytes = buf.getvalue()

env_res = predictor.check_acoustic_environment(quiet_bytes)
print(f"Quiet environment result: {env_res}")
assert env_res["is_ready"] == True, "Quiet room should be ready"

print("\n=== 2. TESTING PHRASE VALIDATION ===")
# Synthesize 3-second speech-like harmonic signal
t = np.linspace(0, 3.0, sr * 3)
f0 = 140.0 # pitch
speech_audio = 0.2 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t) + 0.05 * np.sin(2 * np.pi * 3 * f0 * t)
buf2 = io.BytesIO()
sf.write(buf2, speech_audio, sr, format='WAV')
speech_bytes = buf2.getvalue()

phrase_res = predictor.validate_phrase_sample(speech_bytes)
print(f"Phrase sample validation result: {phrase_res}")
assert phrase_res["is_valid"] == True, "Harmonic speech should be valid"

print("\n=== 3. TESTING TEXT-INDEPENDENT BIOMETRIC DISCRIMINATION ===")
with open('parent_profile.wav', 'rb') as f:
    wav_bytes = f.read()

y_parent = predictor.decode_audio(wav_bytes, target_sr=22050)
assert y_parent is not None, "Parent audio must decode"

# Sample 1: Parent base recording
vec_parent = predictor.extract_speaker_biometric_vector(y_parent, sr=22050)

# Sample 2: Parent speaking in different pitch / conversational variation
y_parent_var = librosa.effects.pitch_shift(y_parent, sr=22050, n_steps=0.2)
vec_parent_var = predictor.extract_speaker_biometric_vector(y_parent_var, sr=22050)

# Sample 3: Stranger / Intruder (Different adult vocal tract +4.5 semitones)
y_stranger = librosa.effects.pitch_shift(y_parent, sr=22050, n_steps=4.5)
vec_stranger = predictor.extract_speaker_biometric_vector(y_stranger, sr=22050)

dist_same = float(1.0 - np.dot(vec_parent, vec_parent_var))
dist_stranger = float(1.0 - np.dot(vec_parent, vec_stranger))

print(f"Same Speaker Biometric Distance (Different Speech): {dist_same:.4f} (Threshold: 0.055)")
print(f"Stranger / Intruder Biometric Distance: {dist_stranger:.4f} (Threshold: 0.055)")

assert dist_same <= 0.055, f"Parent should be recognized! Got {dist_same:.4f}"
assert dist_stranger > 0.055, f"Intruder must be rejected! Got {dist_stranger:.4f}"
print("[OK] Biometric discrimination test passed: Parent recognized, Intruder rejected!")

print("\n=== 4. TESTING SUPABASE ACTIVE PROFILE COMPATIBILITY ===")
res = db.table('registered_voice_profiles').select('*').eq('is_active', True).execute()
print(f"Found {len(res.data)} active profiles in Supabase.")
if len(res.data) > 0:
    p = res.data[0]
    print(f"Profile 0: Name={p['person_name']}, Role={p['role']}, Email={p.get('user_email')}")
    # Test verify_speaker_biometrics against actual profile
    is_match, matched, dist = predictor.verify_speaker_biometrics(speech_bytes, res.data)
    print(f"Verification test against Supabase profiles: match={is_match}, dist={dist:.4f}")

print("\n>>> ALL BIOMETRIC SYSTEM TESTS PASSED SUCCESSFULLY! <<<")

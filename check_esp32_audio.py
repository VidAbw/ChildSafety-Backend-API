from audio_guardian.router import get_registered_mfcc_profiles
from audio_guardian.predictor import predictor
import numpy as np

profiles = get_registered_mfcc_profiles()
print(f"Total loaded active profiles: {len(profiles)}")
for p in profiles:
    mat = np.array(p['matrix'])
    print(f"ID: {p['id']} | Name: {p['person_name']} | Role: {p['role']} | Shape: {mat.shape} | Dims: {mat.ndim}")

# Let's check how predictor.verify_speaker_biometrics behaves with sample audio
import io, soundfile as sf
# Read parent_profile.wav
try:
    with open('parent_profile.wav', 'rb') as f:
        parent_bytes = f.read()
    
    is_match, matched, dist = predictor.verify_speaker_biometrics(parent_bytes, profiles)
    print(f"Testing parent_profile.wav against Supabase profiles:")
    print(f"Match: {is_match} | Matched profile: {matched} | Distance: {dist:.4f}")
except Exception as e:
    print(f"Error testing parent_profile.wav: {e}")

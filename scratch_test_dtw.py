from core.supabase import db
from audio_guardian.predictor import predictor
from pathlib import Path
import numpy as np

# Load profiles from Supabase
res = db.table('registered_voice_profiles').select('id, person_name, role, is_active, dtw_feature_matrix').eq('is_active', True).execute()
stored_profiles = [{'id': r['id'], 'person_name': r['person_name'], 'role': r['role'], 'matrix': r['dtw_feature_matrix']} for r in res.data if r.get('dtw_feature_matrix')]

print(f"Loaded {len(stored_profiles)} profiles:")
for p in stored_profiles:
    print(f" - {p['person_name']} ({p['role']}) -> Matrix shape: {np.array(p['matrix']).shape}")

# Check local parent_profile.wav if exists
p_path = Path("parent_profile.wav")
if p_path.exists():
    with open(p_path, "rb") as f:
        wav_bytes = f.read()
    print(f"\nTesting parent_profile.wav ({len(wav_bytes)} bytes) against stored profiles:")
    is_match, matched, dist = predictor.verify_parent_from_matrix(wav_bytes, stored_profiles)
    print(f"Result: is_match={is_match}, matched={matched}, dist={dist:.4f}")

from core.supabase import db
from audio_guardian.predictor import predictor
import numpy as np
import librosa

res = db.table('registered_voice_profiles').select('*').eq('is_active', True).execute()
print(f"Total active profiles in Supabase: {len(res.data)}")
for i, r in enumerate(res.data):
    mat = np.array(r['dtw_feature_matrix'], dtype=np.float32)
    print(f"[{i}] Name: {r['person_name']} | Role: {r['role']} | Email: {r.get('user_email')} | Matrix shape: {mat.shape} | Mean: {np.mean(mat):.3f} | Std: {np.std(mat):.3f}")

if len(res.data) >= 2:
    for i in range(len(res.data)):
        for j in range(i + 1, len(res.data)):
            m1 = np.array(res.data[i]['dtw_feature_matrix'], dtype=np.float32)
            m2 = np.array(res.data[j]['dtw_feature_matrix'], dtype=np.float32)
            
            m1_norm = (m1 - np.mean(m1, axis=1, keepdims=True)) / np.maximum(np.std(m1, axis=1, keepdims=True), 1e-6)
            m2_norm = (m2 - np.mean(m2, axis=1, keepdims=True)) / np.maximum(np.std(m2, axis=1, keepdims=True), 1e-6)
            
            D, wp = librosa.sequence.dtw(X=m1_norm, Y=m2_norm, metric='cosine')
            dist = float(D[-1, -1]) / len(wp)
            print(f"DTW distance between [{res.data[i]['person_name']}] and [{res.data[j]['person_name']}]: {dist:.4f}")

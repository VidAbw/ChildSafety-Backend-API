from core.supabase import db
import numpy as np

res = db.table('registered_voice_profiles').select('id, person_name, role, is_active, created_at, dtw_feature_matrix').eq('is_active', True).execute()
for r in res.data:
    mat = r.get('dtw_feature_matrix')
    if mat is not None:
        arr = np.array(mat, dtype=np.float32)
        has_nan = np.isnan(arr).any()
        all_zero = np.all(arr == 0)
        print(f"ID: {r['id'][:8]} | Name: {r['person_name']} | Role: {r['role']} | Shape: {arr.shape} | Has NaN: {has_nan} | All Zeros: {all_zero} | Std: {np.std(arr):.4f} | Created: {r['created_at']}")
    else:
        print(f"ID: {r['id'][:8]} | Name: {r['person_name']} | No matrix")

import io
import time
import sounddevice as sd
import numpy as np
import librosa
from pathlib import Path
from audio_guardian.predictor import predictor

SAMPLE_RATE = 22050
DURATION = 3.0
PARENT_PROFILE = Path("parent_profile.wav")
THRESHOLD = 0.18

def record_audio(duration: float = 3.0, sr: int = 22050) -> np.ndarray:
    print(f"\n🎙️  Recording {duration:.1f} seconds from microphone... SPEAK NOW!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording finished.")
    return np.squeeze(audio)

def numpy_to_wav_bytes(y: np.ndarray, sr: int = 22050) -> bytes:
    import wave
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def run_dtw_interactive_test():
    print("=" * 70)
    print(" 🔬 DYNAMIC TIME WARPING (DTW) VOICE VERIFICATION INTERACTIVE TEST ")
    print("=" * 70)
    print("How DTW works:")
    print(" 1. Extracts 20-dimensional MFCC acoustic vectors over time.")
    print(" 2. Uses Cosine Distance DTW to align your voice against reference voice.")
    print(" 3. Distance < 0.18 => ✅ MATCH (Authorized Parent / Safe Override)")
    print("    Distance >= 0.18 => ❌ NO MATCH (Unrecognized speaker / Potential Threat)")
    print("=" * 70)

    # Step 1: Check or record Reference Parent Voice
    if not PARENT_PROFILE.exists() or PARENT_PROFILE.stat().st_size < 1000:
        print("\n[Step 1/2] No reference voice profile found. Let's record your PARENT profile first.")
        input("Press [ENTER] to record 4 seconds of your reference voice...")
        parent_audio = record_audio(duration=4.0, sr=SAMPLE_RATE)
        wav_bytes = numpy_to_wav_bytes(parent_audio, sr=SAMPLE_RATE)
        with open(PARENT_PROFILE, "wb") as f:
            f.write(wav_bytes)
        print(f"🎉 Saved reference voice to '{PARENT_PROFILE}'.")
    else:
        print(f"\n[Step 1/2] Found existing parent reference profile: '{PARENT_PROFILE}'")

    # Extract reference MFCC
    y_parent, sr_parent = librosa.load(PARENT_PROFILE, sr=SAMPLE_RATE)
    mfcc_parent = librosa.feature.mfcc(y=y_parent, sr=sr_parent, n_mfcc=20)
    print(f"Reference voice MFCC matrix shape: {mfcc_parent.shape} (20 Mel bands x {mfcc_parent.shape[1]} frames)")

    while True:
        print("\n" + "-" * 70)
        print("[Step 2/2] Test Voice Verification")
        print("Choose test mode:")
        print("  1: Speak as YOURSELF (Authorized Parent) -> Expect MATCH")
        print("  2: Have someone else speak or speak in a disguised voice -> Expect NO MATCH")
        print("  3: Re-record reference parent profile")
        print("  q: Quit")
        
        choice = input("\nEnter choice (1/2/3/q): ").strip().lower()
        if choice == 'q':
            print("Exiting test. Goodbye!")
            break
        elif choice == '3':
            input("Press [ENTER] to re-record 4 seconds of parent reference voice...")
            parent_audio = record_audio(duration=4.0, sr=SAMPLE_RATE)
            wav_bytes = numpy_to_wav_bytes(parent_audio, sr=SAMPLE_RATE)
            with open(PARENT_PROFILE, "wb") as f:
                f.write(wav_bytes)
            y_parent, sr_parent = librosa.load(PARENT_PROFILE, sr=SAMPLE_RATE)
            mfcc_parent = librosa.feature.mfcc(y=y_parent, sr=sr_parent, n_mfcc=20)
            print(f"🎉 Updated '{PARENT_PROFILE}'.")
            continue

        input("\nPress [ENTER] and speak for 3 seconds...")
        test_audio = record_audio(duration=3.0, sr=SAMPLE_RATE)
        mfcc_test = librosa.feature.mfcc(y=test_audio, sr=SAMPLE_RATE, n_mfcc=20)

        # Compute DTW alignment
        D, wp = librosa.sequence.dtw(X=mfcc_test, Y=mfcc_parent, metric='cosine')
        dtw_distance = float(D[-1, -1]) / len(wp)

        is_match = dtw_distance < THRESHOLD
        similarity_pct = max(0.0, min(100.0, (1.0 - (dtw_distance / 0.35)) * 100.0))

        print("\n" + "=" * 50)
        print(" 📊 DTW VERIFICATION RESULTS")
        print("=" * 50)
        print(f" DTW Alignment Path Length : {len(wp)} warping steps")
        print(f" Normalized Cosine Distance: {dtw_distance:.4f}  (Threshold: {THRESHOLD})")
        print(f" Voice Similarity Score    : {similarity_pct:.1f}%")
        print("-" * 50)
        if is_match:
            print(" 🟢 VERIFICATION STATUS : ✅ MATCH (AUTHORIZED PARENT)")
            print(" 🛡️  GUARDIAN ACTION     : Anti-Fatigue Suppressor ACTIVE -> Threat suppressed to SAFE.")
        else:
            print(" 🔴 VERIFICATION STATUS : ❌ NO MATCH (UNAUTHORIZED / OTHER SPEAKER)")
            print(" ⚠️  GUARDIAN ACTION     : Threat alerts remain UNBLOCKED if loudness or screaming is detected.")
        print("=" * 50)

if __name__ == "__main__":
    run_dtw_interactive_test()

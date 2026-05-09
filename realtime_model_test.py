import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import wave
import sys
import warnings
warnings.filterwarnings('ignore')

# Import the actual AI model from the backend!
from audio_guardian.predictor import predictor

DURATION = 3.0
SAMPLE_RATE = 16000

def record_audio():
    print("\n🎙️ Recording 3 seconds of audio...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete!")
    return audio_data.flatten()

def create_wav_bytes(audio_data):
    # Create an in-memory WAV file to feed the model exactly like the API does
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_data.tobytes())
    return wav_io.getvalue()

def run_demo():
    print("="*50)
    print("REAL-TIME AUDIO GUARDIAN AI DEMO")
    print("="*50)
    input("Press Enter to start recording (Play a radio/TV first)...")
    
    audio_data = record_audio()
    
    # Calculate raw volume
    rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    amplitude_db = 20 * np.log10(rms) if rms > 0 else 0.0
    
    wav_bytes = create_wav_bytes(audio_data)
    
    # Run ACTUAL Deep Learning Model
    print("🧠 Analyzing with 1D-CNN + LSTM...")
    class_id, probability = predictor.predict_from_wav_bytes(wav_bytes)
    
    # Always verify speaker if it's loud enough
    is_vidusha = False
    if amplitude_db > 45.0:
        from pathlib import Path
        PARENT_PROFILE_PATH = Path("parent_profile.wav")
        is_vidusha = predictor.verify_parent(wav_bytes, PARENT_PROFILE_PATH)
        
    if class_id == 1:
        if is_vidusha:
            status = "✅ SAFE (Vidusha is speaking - Threat Override!)"
            class_id = 0
        else:
            status = "🚨 THREAT DETECTED (Unknown Speaker)"
    else:
        status = "✅ SAFE (Vidusha is speaking)" if is_vidusha else "✅ SAFE (No Vocal Aggression)"
    color = 'red' if class_id == 1 else 'green'
    
    print(f"\nResult: {status}")
    print(f"Raw Volume: {amplitude_db:.1f} dB")
    print(f"Threat Probability: {probability:.2%}")
    
    # Compute MFCC for visualization
    # Convert int16 to float32 for librosa
    float_audio = audio_data.astype(np.float32) / 32768.0
    mfccs = librosa.feature.mfcc(y=float_audio, sr=SAMPLE_RATE, n_mfcc=20)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Audio Guardian Analysis\n{status} ({probability:.1%}) | Volume: {amplitude_db:.1f}dB", 
                 fontsize=14, fontweight='bold', color=color)
    
    # Waveform
    librosa.display.waveshow(float_audio, sr=SAMPLE_RATE, ax=axes[0], color='blue' if class_id == 0 else 'red')
    axes[0].set_title('Raw Audio Waveform')
    axes[0].set_ylim([-1, 1])
    
    # MFCC
    img = librosa.display.specshow(mfccs, x_axis='time', ax=axes[1])
    axes[1].set_title('MFCC Spectral Features (What the AI "sees")')
    fig.colorbar(img, ax=axes[1], format='%+2.0f')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    while True:
        run_demo()
        print("\nClose the graph window to continue...")
        retry = input("\nTest again with a different sound? (y/n): ")
        if retry.lower() != 'y':
            break

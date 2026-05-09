import sounddevice as sd
import wave
import io
import numpy as np
from pathlib import Path

DURATION = 5.0
SAMPLE_RATE = 22050
OUTPUT_FILE = "parent_profile.wav"

def record_and_save():
    print("="*50)
    print("🔊 PARENT VOICE REGISTRATION")
    print("="*50)
    print("When you are ready, press Enter and speak normally for 5 seconds.")
    print("Just introduce yourself (e.g., 'Hi, I am Vidusha, this is my voice profile.')")
    input("\nPress Enter to start recording...")
    
    print("\n🎙️ Recording 5 seconds of audio... SPEAK NOW!")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete!")
    
    # Save to WAV file
    with wave.open(OUTPUT_FILE, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_data.tobytes())
        
    print(f"\n🎉 Success! Your voice profile has been saved as '{OUTPUT_FILE}' in the backend folder.")
    print("The AI will now recognize you as Vidusha when you test the Guardian system!")

if __name__ == "__main__":
    record_and_save()

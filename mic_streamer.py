import pyaudio
import wave
import requests
import time
import io

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 3

# Backend endpoint
API_URL = "http://localhost:8000/api/audio/upload-chunk"

def record_and_send():
    audio = pyaudio.PyAudio()

    print("Starting audio stream to backend... Press Ctrl+C to stop.")
    try:
        while True:
            # Start recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            frames = []
            
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            # Save to an in-memory byte buffer
            wav_io = io.BytesIO()
            wf = wave.open(wav_io, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            wav_bytes = wav_io.getvalue()
            
            # Send to backend
            try:
                files = {'file': ('chunk.wav', wav_bytes, 'audio/wav')}
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "Unknown")
                    prob = data.get("probability", "0%")
                    
                    if data.get("class_id") == 1:
                        print(f"🚨 THREAT DETECTED! (Confidence: {prob})")
                    else:
                        print(f"✅ Safe ({prob} threat)")
                else:
                    print(f"Backend error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print("Could not connect to backend. Is it running?")
                time.sleep(2)

    except KeyboardInterrupt:
        print("Stopping microphone stream.")
    finally:
        audio.terminate()

if __name__ == "__main__":
    record_and_send()

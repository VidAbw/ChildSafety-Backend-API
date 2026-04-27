import os
import shutil
import urllib.request
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(r"c:\Users\94771\Desktop\ChildSafety-Backend-API\audio model training")
TRAINING_DIR = BASE_DIR / "Training_Data"
SAFE_DIR = TRAINING_DIR / "Class_0_Safe"
THREAT_DIR = TRAINING_DIR / "Class_1_Threat"

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_ZIP_PATH = BASE_DIR / "ESC-50-master.zip"

def create_directories():
    SAFE_DIR.mkdir(parents=True, exist_ok=True)
    THREAT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directories:\n- {SAFE_DIR}\n- {THREAT_DIR}")

def process_ravdess():
    """
    RAVDESS filename format: 
    Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
    e.g., 03-01-05-01-01-01-01.wav
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    count_safe = 0
    count_threat = 0
    
    # Iterate through Actor_* directories
    for actor_dir in BASE_DIR.glob("Actor_*"):
        if not actor_dir.is_dir():
            continue
        
        for audio_file in actor_dir.glob("*.wav"):
            filename = audio_file.name
            parts = filename.split('-')
            if len(parts) != 7:
                continue
            
            emotion = parts[2]
            
            if emotion in ["05", "06"]: # Angry or Fearful
                dest = THREAT_DIR / f"ravdess_{filename}"
                shutil.copy2(audio_file, dest)
                count_threat += 1
            elif emotion in ["01", "02", "03"]: # Neutral, Calm, Happy
                dest = SAFE_DIR / f"ravdess_{filename}"
                shutil.copy2(audio_file, dest)
                count_safe += 1

    logging.info(f"Processed RAVDESS: {count_safe} Safe files, {count_threat} Threat files.")

def process_tess():
    """
    TESS folders usually named like: OAF_angry, YAF_fear
    or files named like: OAF_back_angry.wav
    """
    count_safe = 0
    count_threat = 0
    
    tess_dir = BASE_DIR / "TESS Toronto emotional speech set data"
    if not tess_dir.exists():
        logging.info("TESS directory not found, skipping.")
        return

    for emotion_dir in tess_dir.iterdir():
        if not emotion_dir.is_dir():
            continue
            
        dir_name = emotion_dir.name.lower()
        is_threat = "angry" in dir_name or "fear" in dir_name
        is_safe = "neutral" in dir_name or "happy" in dir_name or "pleasant_surprise" in dir_name
        
        for audio_file in emotion_dir.glob("*.wav"):
            if is_threat:
                dest = THREAT_DIR / f"tess_{audio_file.name}"
                shutil.copy2(audio_file, dest)
                count_threat += 1
            elif is_safe:
                dest = SAFE_DIR / f"tess_{audio_file.name}"
                shutil.copy2(audio_file, dest)
                count_safe += 1

    logging.info(f"Processed TESS: {count_safe} Safe files, {count_threat} Threat files.")

def download_and_extract_esc50():
    """
    Downloads ESC-50 dataset and extracts audio files to Class_0_Safe.
    """
    if not ESC50_ZIP_PATH.exists():
        logging.info("Downloading ESC-50 dataset... This might take a few minutes (approx. 600MB).")
        try:
            urllib.request.urlretrieve(ESC50_URL, ESC50_ZIP_PATH)
            logging.info("ESC-50 download complete.")
        except Exception as e:
            logging.error(f"Failed to download ESC-50: {e}")
            return
    else:
        logging.info("ESC-50 zip file already exists, skipping download.")

    esc50_audio_dir = BASE_DIR / "ESC-50-master" / "audio"
    if not esc50_audio_dir.exists():
        logging.info("Extracting ESC-50 dataset...")
        try:
            with zipfile.ZipFile(ESC50_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
        except Exception as e:
            logging.error(f"Failed to extract ESC-50: {e}")
            return
            
    # Now copy all ESC-50 audio to Safe directory
    count = 0
    for audio_file in esc50_audio_dir.glob("*.wav"):
        dest = SAFE_DIR / f"esc50_{audio_file.name}"
        if not dest.exists():
            shutil.copy2(audio_file, dest)
            count += 1
            
    logging.info(f"Copied {count} ESC-50 files to Safe directory.")

def main():
    logging.info("Starting data preparation...")
    create_directories()
    process_ravdess()
    process_tess()
    download_and_extract_esc50()
    logging.info("Data preparation completed successfully.")

if __name__ == "__main__":
    main()

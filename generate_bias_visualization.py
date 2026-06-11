import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Parameters
sr = 16000
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 1. Loud Non-Threat Noise (e.g., slamming a door or dropping a heavy object)
# High amplitude, low frequency burst
noise_signal = np.sin(2 * np.pi * 50 * t) * np.exp(-3 * t)
noise_signal += 0.2 * np.random.randn(len(t)) * np.exp(-3 * t)
noise_signal = librosa.util.normalize(noise_signal) * 0.9  # Very Loud

# 2. Quiet Scream (e.g., child crying or screaming further away)
# Lower amplitude, high frequency with harmonics and distress pitch
scream_base_freq = 800
scream_signal = np.sin(2 * np.pi * scream_base_freq * t)
# Add harmonics
for h in range(2, 6):
    scream_signal += (1.0 / h) * np.sin(2 * np.pi * scream_base_freq * h * t)
# Add frequency modulation (vibrato/distress)
scream_signal *= (1 + 0.3 * np.sin(2 * np.pi * 15 * t))
# Lower amplitude than the noise
scream_signal = librosa.util.normalize(scream_signal) * 0.4

# Compute MFCCs
mfcc_noise = librosa.feature.mfcc(y=noise_signal, sr=sr, n_mfcc=20)
mfcc_scream = librosa.feature.mfcc(y=scream_signal, sr=sr, n_mfcc=20)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Solving Intensity Bias: Raw Volume vs. AI Spectral Analysis (MFCC)', fontsize=16, fontweight='bold')

# --- Row 1: Waveforms (What old systems see) ---
librosa.display.waveshow(noise_signal, sr=sr, ax=axes[0,0], color='red')
axes[0,0].set_title('Loud Object Dropped (90dB)\nLegacy System: TRIGGERS FALSE ALARM', color='darkred')
axes[0,0].set_ylim([-1, 1])
axes[0,0].set_ylabel('Amplitude')

librosa.display.waveshow(scream_signal, sr=sr, ax=axes[0,1], color='orange')
axes[0,1].set_title('Distant Child Scream (60dB)\nLegacy System: IGNORES (Too Quiet)', color='darkorange')
axes[0,1].set_ylim([-1, 1])

# --- Row 2: MFCCs (What our 1D-CNN sees) ---
img1 = librosa.display.specshow(mfcc_noise, x_axis='time', ax=axes[1,0])
axes[1,0].set_title('MFCC of Loud Object\nAI System: Identifies as SAFE')
fig.colorbar(img1, ax=axes[1,0], format='%+2.0f')

img2 = librosa.display.specshow(mfcc_scream, x_axis='time', ax=axes[1,1])
axes[1,1].set_title('MFCC of Scream\nAI System: Identifies as THREAT (Alerts!)')
fig.colorbar(img2, ax=axes[1,1], format='%+2.0f')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(r'C:\Users\User\.gemini\antigravity\brain\8bd6fa10-aebc-4542-b0e9-94a095d7f790\intensity_bias_solution.png', dpi=150)
print("Saved visualization to intensity_bias_solution.png")

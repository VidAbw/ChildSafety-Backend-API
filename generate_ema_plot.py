import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Parameters from project implementation (tracker.py & thesis section 3.4)
EMA_ALPHA = 0.65
N_FRAMES = 50
FPS = 30.0

# 1. Simulate a realistic physical trajectory (smooth underlying arm/wrist motion)
# e.g., an arm reaching out or moving across the field of view with an acceleration-deceleration profile
time = np.linspace(0, (N_FRAMES - 1) / FPS, N_FRAMES)
frames = np.arange(1, N_FRAMES + 1)

# Base ground-truth physical trajectory (smooth sigmoidal / polynomial transition)
true_trajectory = 220.0 + 160.0 / (1.0 + np.exp(-0.25 * (frames - 25)))

# Add high-frequency Gaussian pixel jitter typical of MediaPipe landmark localization
jitter_noise = np.random.normal(loc=0.0, scale=8.5, size=N_FRAMES)
raw_coordinates = true_trajectory + jitter_noise

# 2. Apply the exact EMA logic from VelocityTracker in tracker.py:
# S_t = alpha * X_t + (1 - alpha) * S_{t-1} with S_0 = X_0
ema_smoothed = np.zeros(N_FRAMES)
smoothed_val = None

for i in range(N_FRAMES):
    px = raw_coordinates[i]
    if smoothed_val is None:
        smoothed_val = px
    else:
        smoothed_val = EMA_ALPHA * px + (1.0 - EMA_ALPHA) * smoothed_val
    ema_smoothed[i] = smoothed_val

# 3. Academic styling with Seaborn & Matplotlib
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "font.family": "serif"
})

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

# Plot raw coordinates: light red, semi-transparent, markers to indicate frame samples
ax.plot(
    frames,
    raw_coordinates,
    color="#E74C3C",
    alpha=0.45,
    linestyle="-",
    linewidth=1.8,
    marker="o",
    markersize=4.5,
    label="Raw Coordinate ($X_t$, Jittered Observation)"
)

# Plot EMA smoothed coordinates: bold dark blue
ax.plot(
    frames,
    ema_smoothed,
    color="#1B365D",
    alpha=0.95,
    linestyle="-",
    linewidth=2.5,
    label=f"EMA Smoothed Coordinate ($S_t$, $\\alpha = {EMA_ALPHA}$)"
)

# Annotate key characteristics
ax.set_title("Kinematic Landmark Trajectory Filtering via Exponential Moving Average (EMA)", pad=14, weight="bold")
ax.set_xlabel("Frame Index ($t$)", labelpad=8)
ax.set_ylabel("Wrist Coordinate [Pixels]", labelpad=8)

# Subtle grid styling
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
ax.set_xlim(1, N_FRAMES)

# Academic legend placement with clean box
ax.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", framealpha=0.95, loc="upper left")

plt.tight_layout()

# Save as high-resolution PNG
output_filename = "ema_smoothing.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight")
print(f"[+] Plot successfully saved to {output_filename}")

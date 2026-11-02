import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Academic styling configuration
sns.set_theme(style="white", font="serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "font.family": "serif"
})

color_map = {
    "YOLOv8s Detection": "#2B5C8F",              # Steel blue
    "MediaPipe Pose": "#2A9D8F",                # Teal
    "DeepFace SFace": "#E76F51",                # Terracotta / Coral
    "Temporal Tracking & Rules": "#6C757D",      # Cool slate gray
}

# Operating modes: Steady-State on top (y=0), Periodic Evaluation below (y=1)
conditions = [
    (
        "Steady-State Frame\n(Standard Operation, 29/30)",
        [
            ("YOLOv8s Detection", 5.8),
            ("MediaPipe Pose", 32.4),
            ("Temporal Tracking & Rules", 1.8),
        ],
    ),
    (
        "Periodic Evaluation Frame\n(Face Verification Active, 1/30)",
        [
            ("YOLOv8s Detection", 5.8),
            ("MediaPipe Pose", 32.4),
            ("DeepFace SFace", 8.4),
            ("Temporal Tracking & Rules", 1.8),
        ],
    ),
]

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=300)

bar_height = 0.40
y_positions = np.array([0, 1])

legend_handles = {}

for y_idx, (cond_name, stages) in enumerate(conditions):
    left_offset = 0.0
    total_latency = sum(lat for _, lat in stages)
    
    for stage_name, latency in stages:
        color = color_map[stage_name]
        bar = ax.barh(
            y_idx,
            latency,
            left=left_offset,
            height=bar_height,
            color=color,
            edgecolor="white",
            linewidth=1.2,
        )
        
        if stage_name not in legend_handles:
            legend_handles[stage_name] = bar[0]
            
        mid_x = left_offset + (latency / 2.0)
        
        # Display latency values
        if latency >= 5.0:
            ax.text(
                mid_x,
                y_idx,
                f"{latency:.1f} ms",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10
            )
        else:
            # Clean callout for narrow 1.8 ms segment
            ax.text(
                mid_x,
                y_idx - 0.28,
                f"{latency:.1f} ms",
                ha="center",
                va="bottom",
                color="#2B2B2B",
                fontweight="bold",
                fontsize=9.5
            )
            ax.plot([mid_x, mid_x], [y_idx - 0.20, y_idx - 0.26], color="#666666", linewidth=0.8)
            
        left_offset += latency
        
    # Total latency annotation on right side
    fps_equiv = 1000.0 / total_latency
    ax.text(
        total_latency + 1.2,
        y_idx,
        f"Total: {total_latency:.1f} ms  (~{fps_equiv:.1f} FPS)",
        va="center",
        ha="left",
        fontweight="bold",
        color="#1B365D",
        fontsize=10.5
    )

# 33.3 ms Real-Time Budget Line (30 FPS)
threshold_30fps = 33.33
ax.axvline(
    x=threshold_30fps,
    color="#C0392B",
    linestyle="--",
    linewidth=1.8,
    zorder=4
)

# Place threshold label at top above data area
ax.text(
    threshold_30fps,
    -0.55,
    "33.3 ms (30 FPS Real-Time Budget)",
    color="#C0392B",
    ha="center",
    va="bottom",
    fontweight="bold",
    fontsize=10
)

# Axis configuration
ax.set_yticks(y_positions)
ax.set_yticklabels([c[0] for c in conditions], fontweight="bold")
ax.invert_yaxis()  # Ensures y=0 (Steady-State) is at the top
ax.set_xlabel("Latency [Milliseconds]", labelpad=10)
ax.set_title("Per-Frame Processing Latency Decomposition Across Operating Modes", pad=28, weight="bold")

# Limits & grid
ax.set_xlim(0, 64)
ax.set_ylim(1.55, -0.65)
ax.xaxis.grid(True, linestyle=":", alpha=0.6, color="#B0B0B0")
ax.set_axisbelow(True)

# Clean spines
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#666666")

# Legend below plot
stage_labels_ordered = [
    "YOLOv8s Detection",
    "MediaPipe Pose",
    "DeepFace SFace",
    "Temporal Tracking & Rules"
]
ordered_handles = [legend_handles[s] for s in stage_labels_ordered]

ax.legend(
    ordered_handles,
    stage_labels_ordered,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=4,
    frameon=True,
    facecolor="#FDFDFD",
    edgecolor="#DCDCDC",
    fontsize=9.5
)

plt.tight_layout()
output_path = "latency_breakdown.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"[+] Successfully generated revised {output_path}")

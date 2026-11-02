"""
generate_figures.py
Generates academic publication-ready figures for the research paper:
  - Figure 4: Full two-column comparative confusion matrices for domestic knife hazard detection:
    (a) Proposed fine-tuned YOLOv8s (15/15 TP, 0 FP, 0 FN, 125 TN)
    (b) YOLOv10s baseline (12 TP, 6 FP, 3 FN, 119 TN)
    (c) YOLO11s baseline (13 TP, 6 FP, 2 FN, 119 TN)

Fixes applied:
  1. Top-right cell in subplots (b) and (c) formatted as 'FP = 6\n(False Pos.)'
     with 8.5pt font and ha='center' to prevent visual clipping against right border.
  2. Bottom-right cell (Background / Background) formatted with exact negative counts:
     'TN = 125\n(100.0%)' for YOLOv8s, and 'TN = 119\n(95.2%)' for YOLOv10s and YOLO11s.
  3. Re-exported at 300 DPI in both PDF and PNG formats to Images/ and root directory.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Academic styling configuration
sns.set_theme(style="white", font="serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
    "figure.titlesize": 12.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def generate_figure_4():
    """Generates comparative confusion matrices for YOLOv8s, YOLOv10s, and YOLO11s."""
    # Data definitions for the 3 models on the 140-sample evaluation split:
    # 15 positive knife instances + 125 hard-negative metallic kitchenware background images
    models = [
        {
            "title": "(a) Proposed Fine-Tuned YOLOv8s",
            "tp": 15, "tp_pct": "100.0%",
            "fp": 0,  "fp_text": "0\n(0.0%)",
            "fn": 0,  "fn_pct": "0.0%",
            "tn": 125, "tn_pct": "100.0%",
            "matrix": np.array([[15, 0], [0, 125]]),
        },
        {
            "title": "(b) YOLOv10s Baseline",
            "tp": 12, "tp_pct": "80.0%",
            "fp": 6,  "fp_text": "6\n(False Pos.)",
            "fn": 3,  "fn_pct": "20.0%",
            "tn": 119, "tn_pct": "95.2%",
            "matrix": np.array([[12, 6], [3, 119]]),
        },
        {
            "title": "(c) YOLO11s Baseline",
            "tp": 13, "tp_pct": "86.7%",
            "fp": 6,  "fp_text": "6\n(False Pos.)",
            "fn": 2,  "fn_pct": "13.3%",
            "tn": 119, "tn_pct": "95.2%",
            "matrix": np.array([[13, 6], [2, 119]]),
        },
    ]

    # Standard IEEE two-column width: ~7.16 inches (using 11.2 x 3.6 for wide 3-panel display)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=300)

    classes = ["Knife", "Background"]

    for i, (ax, m) in enumerate(zip(axes, models)):
        # Normalize cell color shading by class column totals
        # Col 0 (Knife): total 15 instances
        # Col 1 (Background): total 125 instances
        norm_matrix = np.zeros((2, 2))
        norm_matrix[0, 0] = m["matrix"][0, 0] / 15.0  # TP rate
        norm_matrix[1, 0] = m["matrix"][1, 0] / 15.0  # FN rate
        norm_matrix[0, 1] = m["matrix"][0, 1] / 125.0 # FP rate
        norm_matrix[1, 1] = m["matrix"][1, 1] / 125.0 # TN rate

        # Custom cell color mapping:
        # Diagonal (correct predictions): shades of deep blue / steel blue
        # Off-diagonal (errors): soft coral / neutral
        cell_colors = np.empty((2, 2), dtype=object)
        
        # Draw matrix grid manually for precise control over text and borders
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)

        # Plot cells
        for r in range(2):
            for c in range(2):
                val = norm_matrix[r, c]
                if (r, c) == (0, 0): # TP
                    color = plt.cm.Blues(0.35 + 0.55 * val)
                elif (r, c) == (1, 1): # TN
                    color = plt.cm.Blues(0.25 + 0.55 * val)
                elif (r, c) == (0, 1): # FP
                    color = plt.cm.Reds(0.05 + 0.65 * (val * 10)) if val > 0 else "#F8F9FA"
                else: # FN
                    color = plt.cm.Oranges(0.05 + 0.75 * val) if val > 0 else "#F8F9FA"

                rect = plt.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                                     facecolor=color, edgecolor="#555555", linewidth=1.1)
                ax.add_patch(rect)

        # Cell 0,0: TP
        tp_text = f"TP = {m['tp']}\n({m['tp_pct']})"
        ax.text(0, 0, tp_text, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white" if m["tp"] >= 10 else "black")

        # Cell 0,1: FP
        # Requirement: In subplots (b) and (c), change to '(False Pos.)' with 8.5pt font and ha='center'
        if i == 0:
            fp_text = f"FP = 0\n(0.0%)"
            fp_font = 9.5
        else:
            fp_text = f"FP = {m['fp']}\n(False Pos.)"
            fp_font = 8.5

        ax.text(1, 0, fp_text, ha="center", va="center",
                fontsize=fp_font, fontweight="bold" if i > 0 else "normal",
                color="#A00000" if m["fp"] > 0 else "#555555")

        # Cell 1,0: FN
        fn_text = f"FN = {m['fn']}\n({m['fn_pct']})" if m["fn"] > 0 else "FN = 0\n(0.0%)"
        ax.text(0, 1, fn_text, ha="center", va="center",
                fontsize=9.5,
                fontweight="bold" if m["fn"] > 0 else "normal",
                color="#803000" if m["fn"] > 0 else "#555555")

        # Cell 1,1: TN
        # Requirement: Replace 'TN = —' with exact negative background counts:
        # 'TN = 125\n(100.0%)' for YOLOv8s, and 'TN = 119\n(95.2%)' for YOLOv10s and YOLO11s
        tn_text = f"TN = {m['tn']}\n({m['tn_pct']})"
        ax.text(1, 1, tn_text, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Knife", "Background"], fontsize=9.5)
        ax.set_yticklabels(["Knife", "Background"], fontsize=9.5)
        ax.set_xlabel("Actual Class (Ground Truth)", fontsize=10, labelpad=6)
        if i == 0:
            ax.set_ylabel("Predicted Class", fontsize=10, labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_title(m["title"], fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()

    # Save to both locations: Images/fig4_yolo_cm_wide.pdf (for LaTeX) and root directory
    os.makedirs("Images", exist_ok=True)
    out_pdf_images = os.path.join("Images", "fig4_yolo_cm_wide.pdf")
    out_png_images = os.path.join("Images", "fig4_yolo_cm_wide.png")
    out_pdf_root = "fig4_yolo_cm_wide.pdf"
    out_png_root = "fig4_yolo_cm_wide.png"

    plt.savefig(out_pdf_images, dpi=300, bbox_inches="tight")
    plt.savefig(out_png_images, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf_root, dpi=300, bbox_inches="tight")
    plt.savefig(out_png_root, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Figure 4 successfully generated and exported to:")
    print(f"    - {out_pdf_images} (300 DPI PDF)")
    print(f"    - {out_png_images} (300 DPI PNG)")
    print(f"    - {out_pdf_root}")
    print(f"    - {out_png_root}")

if __name__ == "__main__":
    generate_figure_4()

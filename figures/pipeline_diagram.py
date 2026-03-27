"""
Pipeline diagram for the final report.
Top-to-bottom flow, 5 phases, clean orthogonal layout.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
})

# ── Palette ──────────────────────────────────────────────────────────
C_DATA = "#dae8fc"   # light blue — datasets
C_PROC = "#fff2cc"   # light yellow — models / processes
C_OUT  = "#d5e8d4"   # light green — outputs
C_EDGE = "#333333"
C_TEXT = "#1a1a1a"
C_NOTE = "#555555"
C_PHASE = "#888888"

fig, ax = plt.subplots(figsize=(7.0, 6.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

BH = 7   # box height
BW = 17  # default box width


# ── Helpers ──────────────────────────────────────────────────────────
def box(cx, cy, w, h, text, color, fs=7, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3",
        facecolor=color, edgecolor=C_EDGE, linewidth=0.8))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, fontweight="bold" if bold else "normal",
            linespacing=1.3)


def arr(x1, y1, x2, y2, style="-", lw=0.8):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.18,head_length=0.14",
            color=C_EDGE, lw=lw, linestyle=style))


def arr_L(pts, style="-", lw=0.8):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]],
                [pts[i][1], pts[i + 1][1]],
                color=C_EDGE, lw=lw, linestyle=style,
                solid_capstyle="butt")
    arr(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
        style=style, lw=lw)


def note(x, y, text, **kw):
    defaults = dict(ha="center", va="center", fontsize=5.5,
                    color=C_NOTE, style="italic")
    defaults.update(kw)
    ax.text(x, y, text, **defaults)


def phase_label(y, text):
    ax.text(1.5, y, text, ha="left", va="center", fontsize=6.5,
            color=C_PHASE, fontweight="bold", rotation=90)


# ===================================================================
# PHASE 1  –  Data Preprocessing  (y = 91)
# ===================================================================
Y1 = 91
phase_label(Y1, "Phase 1")

box(15, Y1, 16, BH, "TripAdvisor\nReviews\n(10,232)", C_DATA)
box(37, Y1, 16, BH, "Filter &\nSegment\n(spaCy)", C_PROC)
box(59, Y1, 16, BH, "Clean &\nDeduplicate", C_PROC)
box(81, Y1, 16, BH, "MBS Sentences\n(70,085)", C_DATA, bold=True)

arr(23, Y1, 29, Y1)
note(26, Y1 + 4, "4-5 stars")
arr(45, Y1, 51, Y1)
arr(67, Y1, 73, Y1)
note(70, Y1 + 4, "4-100 tokens")

# ===================================================================
# PHASE 2-3  –  Data Sources  (y = 77)
# ===================================================================
Y2 = 77
phase_label(Y2, "Phase 2-3")

box(30, Y2, 18, BH, "SemEval Train\n(8,500 sent.)", C_DATA, bold=True)
box(70, Y2, 18, BH, "MBS Annotations\n(400 sent.)", C_DATA, bold=True)

# MBS sentences -> annotate sample
arr_L([(81, Y1 - BH / 2),
       (81, Y2 + 5),
       (79, Y2 + BH / 2)])
note(83, Y2 + 7.5, "sample &\nannotate", ha="left", fontsize=5)

# ===================================================================
# PHASE 4  –  Models  (y = 61)
# ===================================================================
Y3 = 61
phase_label(Y3, "Phase 4")

box(15, Y3, 16, BH, "Regex\nBaseline", C_PROC)
box(37, Y3, 16, BH, "TF-IDF + LR", C_PROC)
box(59, Y3, 16, BH, "BERT\nStage 1", C_PROC)
box(81, Y3, 16, BH, "BERT\nStage 2", C_PROC, bold=True)

# SemEval -> models (train)
arr(24, Y2 - BH / 2, 15, Y3 + BH / 2)
arr(30, Y2 - BH / 2, 37, Y3 + BH / 2)
arr(36, Y2 - BH / 2, 59, Y3 + BH / 2)
note(20, Y2 - 5, "train", fontsize=5)

# BERT S1 -> S2 (checkpoint)
arr(67, Y3, 73, Y3)
note(70, Y3 + 3, "checkpoint")

# MBS annotations -> BERT S2 (fine-tune)
arr(70, Y2 - BH / 2, 81, Y3 + BH / 2)
note(78, Y2 - 5, "fine-tune", fontsize=5)

# ===================================================================
# Evaluation  (y = 47)
# ===================================================================
Y4 = 47
phase_label(Y4, "Eval")

box(37, Y4, 26, BH, "Evaluation\n(SemEval test + MBS test)", C_OUT)

# All 4 models -> evaluation (dashed)
arr(15, Y3 - BH / 2, 28, Y4 + BH / 2, style="--", lw=0.6)
arr(37, Y3 - BH / 2, 35, Y4 + BH / 2, style="--", lw=0.6)
arr(59, Y3 - BH / 2, 42, Y4 + BH / 2, style="--", lw=0.6)
arr_L([(81, Y3 - BH / 2),
       (81, Y4 + 2),
       (48, Y4 + BH / 2)],
      style="--", lw=0.6)

# ===================================================================
# PHASE 5  –  Insight Generation  (y = 33)
# ===================================================================
Y5 = 33
phase_label(Y5, "Phase 5")

box(59, Y5, 18, BH, "Full-Dataset\nPredictions\n(4,179 sugg.)", C_OUT, bold=True)
box(81, Y5, 16, BH, "BERTopic\nClustering\n(56 topics)", C_OUT)

# BERT S2 -> predict full MBS
arr_L([(89, Y3 - BH / 2),
       (89, Y5),
       (68, Y5 + BH / 2)])
note(91, Y4, "predict\n70,085 sent.", fontsize=5, ha="left")

# Predictions -> BERTopic
arr(68, Y5, 73, Y5)

# ===================================================================
# PHASE 6  –  Final Output  (y = 19)
# ===================================================================
Y6 = 19
phase_label(Y6, "Phase 6")

box(37, Y6, 22, BH, "Error Analysis\n(FP / FN by category)", C_OUT)
box(70, Y6, 22, BH, "Aspect Insights\n(7 hotel aspects)", C_OUT, bold=True)

# BERTopic -> Aspect
arr(81, Y5 - BH / 2, 76, Y6 + BH / 2)

# Evaluation -> Error Analysis
arr(37, Y4 - BH / 2, 37, Y6 + BH / 2)

# ===================================================================
# Legend
# ===================================================================
legend_patches = [
    mpatches.Patch(facecolor=C_DATA, edgecolor=C_EDGE, label="Dataset"),
    mpatches.Patch(facecolor=C_PROC, edgecolor=C_EDGE, label="Model / Process"),
    mpatches.Patch(facecolor=C_OUT, edgecolor=C_EDGE, label="Output / Analysis"),
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=6.5,
          frameon=True, fancybox=False, edgecolor=C_EDGE,
          borderpad=0.6, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout(pad=0.3)
plt.savefig("figures/pipeline_diagram.png", dpi=600,
            bbox_inches="tight", facecolor="white")
plt.savefig("figures/pipeline_diagram.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved figures/pipeline_diagram.png and .pdf")

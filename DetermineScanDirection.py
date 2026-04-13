"""
DetermineScanDirection.py
================
Reads a .up2 EBSD pattern file and:
  1. Creates two videos (top row sweep + first column sweep), each with a
     centre crosshair so you can track features by eye.
  2. Plots cumulative feature-shift curves using phase cross-correlation
     between consecutive patterns — one plot per sweep direction — so you
     can see how far features move without needing to watch the video.

Edit the INPUTS section, then run:
    python view_patterns.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Data.py is at the project root (same directory as this script)
sys.path.insert(0, os.path.dirname(__file__))
from Data import UP2

# ============================================================
# INPUTS
# ============================================================

up2_file   = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/SiGe_largerRegion.up2'
Rows       = 132
Columns    = 132

output_dir = '/Users/crestiennedechaine/Scripts/DIC-HREBSD/DIC-HREBSD/results/SiGe/PCShiftInUpdatedUP2/'
fps        = 10      # frames per second

# Process patterns before display? (high-pass + CLAHE — slower but clearer)
process = False

# Flip patterns vertically before display?
flip_ud = False

# ============================================================
# HELPERS
# ============================================================

def pattern_index(row, col, ncols):
    """Flat index into the UP2 for scan position (row, col)."""
    return row * ncols + col


def _read_frames(up2, indices, process):
    """Read, normalise, and mildly boost contrast of a list of patterns."""
    frames = []
    for k, idx in enumerate(indices):
        pat = up2.read_pattern(idx, process=process).astype(np.float32)
        if flip_ud:
            pat = np.flipud(pat)
        # Percentile stretch: clip the bottom 2 % and top 2 % of intensities
        # for a mild contrast boost without the full processing pipeline
        lo, hi = np.percentile(pat, 2), np.percentile(pat, 98)
        if hi > lo:
            pat = (pat - lo) / (hi - lo)
        pat = np.clip(pat, 0, 1)
        frames.append(pat)
        if (k + 1) % 10 == 0 or (k + 1) == len(indices):
            print(f"  read {k+1}/{len(indices)}", end="\r")
    print()
    return frames


def make_video(frames, labels, output_path, fps):
    """
    Save an animated GIF/MP4 from a list of normalised pattern frames.
    A centre crosshair is drawn on every frame.

    Parameters
    ----------
    frames      : list of 2-D float32 arrays, already normalised to [0, 1]
    labels      : list of title strings, one per frame
    output_path : desired output path (.mp4 — falls back to .gif if no ffmpeg)
    fps         : frames per second
    """
    print(f"Building {len(frames)}-frame video → {output_path}")

    H, W = frames[0].shape
    cy, cx = H / 2, W / 2          # centre in data coordinates
    cross_len = min(H, W) * 0.06   # arm length = 6 % of pattern size

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1, animated=True,
                   extent=[0, W, 0, H])   # extent so data coords = pixels

    # Static crosshair lines (drawn once, updated by blit)
    hline, = ax.plot([cx - cross_len, cx + cross_len], [cy, cy],
                     color="red", linewidth=1.0, animated=True)
    vline, = ax.plot([cx, cx], [cy - cross_len, cy + cross_len],
                     color="red", linewidth=1.0, animated=True)

    title = ax.set_title(labels[0], fontsize=11)
    plt.tight_layout()

    def _update(k):
        im.set_data(frames[k])
        title.set_text(labels[k])
        return im, hline, vline, title

    ani = animation.FuncAnimation(
        fig, _update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True,
    )

    if animation.FFMpegWriter.isAvailable():
        writer   = animation.FFMpegWriter(fps=fps, bitrate=2000)
        save_path = output_path
    else:
        print("  ffmpeg not found — saving as GIF instead")
        writer   = animation.PillowWriter(fps=fps)
        save_path = output_path.replace(".mp4", ".gif")

    ani.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Saved {save_path}")


def _make_overlay(ref, curr):
    """
    Build a red/blue two-channel overlay of two normalised patterns.

    ref and curr are both float32 arrays in [0, 1].
    Result is an (H, W, 3) uint8 RGB image:
        R channel = reference pattern
        G channel = 0
        B channel = current pattern
    Aligned features appear magenta/grey; shifted features show
    red fringing on one side and blue fringing on the other.
    """
    rgb = np.stack([ref, np.zeros_like(curr), curr], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def plot_overlay_grid(frames, labels, title, save_path, n_panels=12):
    """
    Save a static grid figure showing red/cyan overlays of the reference
    pattern (frame[0]) against N evenly-spaced patterns along the sweep.

    Red   = reference (first pattern in the sweep)
    Cyan  = current position
    White/grey = features that still align; colour fringing = shift

    Parameters
    ----------
    frames    : list of normalised 2-D float32 arrays
    labels    : label for each frame (used as subplot title)
    title     : figure suptitle
    save_path : where to save the .png
    n_panels  : how many overlay panels to show (default 12)
    """
    n = len(frames)
    indices = np.linspace(0, n - 1, min(n_panels, n), dtype=int)

    ncols = min(6, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 3 * nrows + 0.6))
    axes_flat = np.array(axes).flatten()

    ref = frames[0]
    for k, idx in enumerate(indices):
        ax = axes_flat[k]
        overlay = _make_overlay(ref, frames[idx])
        ax.imshow(overlay)
        ax.set_title(labels[idx], fontsize=8)
        ax.axis("off")

    for ax in axes_flat[len(indices):]:
        ax.axis("off")

    fig.suptitle(title + "\n(red = reference, blue = current position)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    print(f"Saved {save_path}")


# ============================================================
# MAIN
# ============================================================

up2 = UP2(up2_file)
print(up2)

total_needed = Rows * Columns
if up2.nPatterns < total_needed:
    raise ValueError(
        f"UP2 file only has {up2.nPatterns} patterns but "
        f"{Rows}×{Columns} = {total_needed} were requested."
    )

os.makedirs(output_dir, exist_ok=True)

# ── Top row: read once, use for both video and shift plot ────────────────────
print("=== Top row (Row 0) ===")
row_indices = [pattern_index(0, col, Columns) for col in range(Columns)]
row_labels  = [f"Col {col}" for col in range(Columns)]
row_frames  = _read_frames(up2, row_indices, process)

make_video(
    frames      = row_frames,
    labels      = [f"Row 0 | Col {col}" for col in range(Columns)],
    output_path = os.path.join(output_dir, "top_row_scan.mp4"),
    fps         = fps,
)

plot_overlay_grid(
    frames    = row_frames,
    labels    = row_labels,
    title     = "Top row sweep (Row 0) — feature drift from reference",
    save_path = os.path.join(output_dir, "top_row_overlay_grid.png"),
)

# ── First column: read once, use for both video and shift plot ───────────────
print("=== First column (Col 0) ===")
col_indices = [pattern_index(row, 0, Columns) for row in range(Rows)]
col_labels  = [f"Row {row}" for row in range(Rows)]
col_frames  = _read_frames(up2, col_indices, process)

make_video(
    frames      = col_frames,
    labels      = [f"Col 0 | Row {row}" for row in range(Rows)],
    output_path = os.path.join(output_dir, "first_col_scan.mp4"),
    fps         = fps,
)

plot_overlay_grid(
    frames    = col_frames,
    labels    = col_labels,
    title     = "First column sweep (Col 0) — feature drift from reference",
    save_path = os.path.join(output_dir, "first_col_overlay_grid.png"),
)

plt.show()
print("Done.")

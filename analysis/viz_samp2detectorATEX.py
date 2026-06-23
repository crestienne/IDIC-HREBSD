"""
Visualize the three passive (frame) rotations that carry the global reference
frame onto the EDAX/TSL EBSD sample frame.

Passive rotation sequence (intrinsic — each rotation is about the current
body-fixed axis):

  Step 1:  90° CW  about  X      (global X)
  Step 2:  90° CCW about  Z'     (body Z after step 1)
  Step 3:  (90 − tilt)° CW about Y''  (body Y after steps 1-2)
           → lands on the EDAX sample frame

Convention used throughout:
  passive rotation CW by α  =  R_active(+α)  [right-hand convention]
  passive rotation CCW by α =  R_active(−α)

The global reference frame is shown FAINT in every panel.
The body-frame axes are labeled  x_s, y_s, z_s  (subscript s = sample)
in the starting panel and carried through each rotation step.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── geometry parameters ───────────────────────────────────────────────────────
TILT     = 70.0   # sample tilt in degrees
DET_TILT = 10.0   # detector tilt in degrees

# ── active rotation matrix building blocks ────────────────────────────────────

def Rx(deg):
    θ = np.deg2rad(deg)
    return np.array([[1, 0,           0          ],
                     [0, np.cos(θ), -np.sin(θ)  ],
                     [0, np.sin(θ),  np.cos(θ)  ]])

def Ry(deg):
    θ = np.deg2rad(deg)
    return np.array([[ np.cos(θ), 0, np.sin(θ)],
                     [ 0,         1, 0         ],
                     [-np.sin(θ), 0, np.cos(θ)]])

def Rz(deg):
    θ = np.deg2rad(deg)
    return np.array([[np.cos(θ), -np.sin(θ), 0],
                     [np.sin(θ),  np.cos(θ), 0],
                     [0,          0,          1]])

# ── passive rotation matrices (intrinsic sequence) ───────────────────────────
#   passive CW  α  about axis  =  R_active(+α)
#   passive CCW α  about axis  =  R_active(−α)

P1 = Rx(+90)               # passive 90° CW  about X
P2 = Rz(-90)               # passive 90° CCW about Z'
P3 = Ry(+(90 - TILT))      # passive (90−tilt)° CW about Y''

# Cumulative passive rotation matrices (intrinsic: prepend each new step)
P_1   = P1
P_12  = P2 @ P1
P_123 = P3 @ P2 @ P1

# For DRAWING: the body-frame axes expressed in global coordinates are the
# COLUMNS of P^{-T} = P^T (since P is orthogonal).
# draw_frame() draws the columns of the matrix it receives as arrows.
F0 = np.eye(3)   # global / initial frame
F1 = P_1.T       # body frame after step 1,  in global coords
F2 = P_12.T      # body frame after step 2
F3 = P_123.T     # body frame after step 3  →  EDAX sample frame

# ── drawing helper ────────────────────────────────────────────────────────────
COLORS = ['tab:red', 'tab:green', 'tab:blue']


def draw_frame(ax, F, labels, scale=0.9, alpha=1.0):
    """Draw three arrows (columns of F) with given labels and colour."""
    for i in range(3):
        v = F[:, i] * scale
        ax.quiver(*np.zeros(3), *v,
                  color=COLORS[i], alpha=alpha,
                  arrow_length_ratio=0.15, linewidth=2)
        tip = F[:, i] * scale * 1.18
        ax.text(*tip, labels[i],
                color=COLORS[i], fontsize=11, fontweight='bold', ha='center')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_box_aspect([-1, -1, 1])


GLOBAL_LABELS = ['x', 'y', 'z']
SAMPLE_LABELS = [r'$x_s$', r'$y_s$', r'$z_s$']   # subscript s throughout

steps = [
    (F0, SAMPLE_LABELS,
     f"Start: global ref = initial\nEDAX sample frame"),

    (F1, SAMPLE_LABELS,
     r"Step 1: passive 90° CW about $X$"),

    (F2, SAMPLE_LABELS,
     r"Step 2: + passive 90° CCW about $Z'$"),

    (F3, SAMPLE_LABELS,
     rf"Step 3: + passive {90 - TILT:.0f}° CW about $Y''$"
     "\n→ EDAX sample frame"),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle(
    rf"Passive rotations: global frame $\rightarrow$ EDAX sample frame"
    f"   (tilt = {TILT}°)",
    fontsize=13, fontweight='bold')

for idx, (F, labels, title) in enumerate(steps):
    ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
    ax.set_title(title, fontsize=10, pad=6)
    # faint global reference axes in every panel
    draw_frame(ax, np.eye(3), GLOBAL_LABELS, scale=0.5, alpha=0.15)
    # current body (sample) frame
    draw_frame(ax, F, labels, scale=0.9, alpha=1.0)

plt.tight_layout()
plt.savefig("samp2detectorATEX_viz.png", dpi=150, bbox_inches='tight')
print("Saved samp2detectorATEX_viz.png")
plt.show()

# ── sanity print (global → EDAX) ──────────────────────────────────────────────
print(f"\nFinal EDAX sample-frame axes in global coordinates (tilt = {TILT}°):")
for i, lbl in enumerate(['x_s', 'y_s', 'z_s']):
    print(f"  {lbl}  =  {np.round(F3[:, i], 4)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — EDAX sample frame  →  ATEX detector frame
#  Four passive (intrinsic) rotations starting from the EDAX sample frame:
#    Step 1: CW (90 − samp_tilt)° about  Y_s
#    Step 2: CW 90°               about  Z'
#    Step 3: reverse Z axis        (improper: det = −1, frame becomes LH)
#    Step 4: CW det_tilt°          about  X'''
# ═══════════════════════════════════════════════════════════════════════════════

# passive rotation matrices for this chain
#   passive CW  α = R_active(+α)
#   passive CCW α = R_active(−α)
Q1 = Ry(-(90 - TILT))   # step 1: passive CCW (90−tilt)° about Y_s
Q2 = Rz(+90)             # step 2: passive CW  90°         about Z'
Q3 = Ry(+180)            # step 3: passive 180°            about Y''
Q4 = Rx(-DET_TILT)       # step 4: passive CCW det_tilt°   about X'''

# cumulative passive matrices (intrinsic: prepend each new step)
QQ1 = Q1
QQ2 = Q2 @ Q1
QQ3 = Q3 @ Q2 @ Q1
QQ4 = Q4 @ Q3 @ Q2 @ Q1  # full sample → detector rotation

# body-frame axes in global coords at each step.
# Starting point is the EDAX sample frame (P_123).
# After applying QQ_i on top: (QQ_i @ P_123)^T
G0 = P_123.T              # EDAX sample frame (starting point)
G1 = (QQ1 @ P_123).T
G2 = (QQ2 @ P_123).T
G3 = (QQ3 @ P_123).T
G4 = (QQ4 @ P_123).T     # ATEX detector frame

DETECTOR_LABELS = [r'$x_d$', r'$y_d$', r'$z_d$']

steps2 = [
    (G0, SAMPLE_LABELS,
     r"Start: EDAX sample frame"),

    (G1, SAMPLE_LABELS,
     rf"Step 1: passive {90-TILT:.0f}° CCW about $Y_s$"),

    (G2, SAMPLE_LABELS,
     r"Step 2: + passive 90° CW about $Z'$"),

    (G3, SAMPLE_LABELS,
     r"Step 3: + passive 180° about $Y''$"),

    (G4, DETECTOR_LABELS,
     rf"Step 4: + passive {DET_TILT:.0f}° CCW about $X'''$"
     "\n→ ATEX detector frame"),
]

fig2 = plt.figure(figsize=(22, 5))
fig2.suptitle(
    rf"Passive rotations: EDAX sample frame $\rightarrow$ ATEX detector frame"
    f"   (sample tilt = {TILT}°,  det tilt = {DET_TILT}°)",
    fontsize=13, fontweight='bold')

for idx, (G, labels, title) in enumerate(steps2):
    ax = fig2.add_subplot(1, 5, idx + 1, projection='3d')
    ax.set_title(title, fontsize=10, pad=6)
    # faint global reference in every panel
    draw_frame(ax, np.eye(3), GLOBAL_LABELS, scale=0.5, alpha=0.15)
    # current frame
    draw_frame(ax, G, labels, scale=0.9, alpha=1.0)

plt.tight_layout()
plt.savefig("edax2atex_viz.png", dpi=150, bbox_inches='tight')
print("Saved edax2atex_viz.png")
plt.show()

# ── sanity print (EDAX → ATEX) ────────────────────────────────────────────────
print(f"\nFinal ATEX detector-frame axes in global coordinates:")
for i, lbl in enumerate(['x_d', 'y_d', 'z_d']):
    print(f"  {lbl}  =  {np.round(G4[:, i], 4)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Start frame vs end frame side by side
#  Left:  global reference frame  (beginning of the full chain)
#  Right: ATEX detector frame     (end of the full chain)
# ═══════════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5),
                            subplot_kw={'projection': '3d'})
fig3.suptitle(
    rf"Start $\rightarrow$ End:  global frame  →  ATEX detector frame"
    f"\n(sample tilt = {TILT}°,  detector tilt = {DET_TILT}°)",
    fontsize=13, fontweight='bold')

# left panel — global (start) frame
ax_start = axes3[0]
ax_start.set_title("Global reference frame\n(start)", fontsize=11, pad=8)
draw_frame(ax_start, F0, GLOBAL_LABELS, scale=0.9, alpha=1.0)

# right panel — ATEX detector frame (end)
ax_end = axes3[1]
ax_end.set_title("ATEX detector frame\n(end)", fontsize=11, pad=8)
draw_frame(ax_end, np.eye(3), GLOBAL_LABELS, scale=0.5, alpha=0.15)
draw_frame(ax_end, G4, DETECTOR_LABELS, scale=0.9, alpha=1.0)

plt.tight_layout()
plt.savefig("start_vs_end_frames.png", dpi=150, bbox_inches='tight')
print("Saved start_vs_end_frames.png")
plt.show()



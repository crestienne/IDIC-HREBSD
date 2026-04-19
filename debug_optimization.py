"""
debug_optimization.py
---------------------
Runs the IC-GN optimization pipeline step-by-step for one pattern from SiGe
and one pattern from SiIndent, visualising every intermediate quantity so you
can compare what the optimizer is doing in each dataset.

Figures produced
----------------
  Fig 1 — Reference patterns: raw, processed, GRx, GRy, gradient magnitude
  Fig 2 — FMT/FCC initial guess diagnostics for both patterns
  Fig 3 — Warp quality: raw target, processed, warped @ init, warped @ final, residuals
  Fig 4 — Convergence curves (residual + dp_norm per iteration, log scale)
  Fig 5 — Residual image snapshots across iterations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import linalg, interpolate, signal

import Data
import utilities
import warp
import conversions
from get_homography_cpu import (
    initial_guess_run,
    optimize_run,
    window_and_normalize_new as window_and_normalize,
    FMT,
    dp_norm as compute_dp_norm,
)

# ============================================================
# USER INPUTS
# ============================================================

CONFIGS = {
    "SiGe": dict(
        up2        = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/SiGe_largerRegion_20260322_512x512.up2',
        ang        = '/Users/crestiennedechaine/OriginalData/Si_Ge_Dataset/dp-Si-new-refined.ang',
        target_idx = 20,          # flat pattern index to analyse
        low_pass_sigma        = 1.5,
        high_pass_sigma       = 70.0,
        truncate_std_scale    = 3.0,
        mask_type             = "none",
        center_cross_half_width = 6,
        clahe_kernel          = (10, 10),   # smaller → more local contrast at band edges
        clahe_clip            = 0.005,     # higher → more aggressive enhancement
        clahe_nbins           = 256,
        use_clahe             = True,
        flip_x                = True,
        rescale_to_uint16     = True,
        unsharp_sigma         = 0.0,      # sharpens band edges; try 1–4
        unsharp_strength      = 0.0,      # boost amount; try 0.5–3
        crop_fraction         = 0.9,
        color                 = "steelblue",
        pixel_size            = 30.0,      # detector pixel size (e.g. nm/pixel) — fill in real value
    ),
    "SiIndent": dict(
        up2        = '/Users/crestiennedechaine/OriginalData/Si-Indent/001_Si_spherical_indent_20kV.up2',
        ang        = '/Users/crestiennedechaine/OriginalData/Si-Indent/dp2-refined.ang',
        target_idx = 20,        # flat pattern index to analyse (~row 20, halfway)
        low_pass_sigma        = 1.5,
        high_pass_sigma       = 80.0,
        truncate_std_scale    = 3.0,
        mask_type             = "center_cross",
        center_cross_half_width = 6,
        clahe_kernel          = (10, 10),
        clahe_clip            = 0.005,
        clahe_nbins           = 256,
        use_clahe             = True,
        flip_x                = False,
        rescale_to_uint16     = True,
        unsharp_sigma         = 0.0,
        unsharp_strength      = 0.0,
        crop_fraction         = 0.9,
        color                 = "tomato",
        pixel_size            = 55.0,      # detector pixel size (e.g. nm/pixel) — fill in real value
    ),
}

MAX_ITER      = 150
CONV_TOL      = 1e-3
N_ITER_FRAMES = 10    # number of iteration snapshots in Fig 5

SAVE_DIR = "debug_optimization_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# BUILD PIPELINE FOR ONE DATASET
# ============================================================

def build_pipeline(cfg, label):
    """Set up pat_obj, reference precomputation, Jacobian, Hessian, FMT."""
    print(f"\n{'='*50}")
    print(f"Setting up {label} …")

    pat_obj = Data.UP2(cfg["up2"])
    pat_obj.set_processing(
        low_pass_sigma          = cfg["low_pass_sigma"],
        high_pass_sigma         = cfg["high_pass_sigma"],
        truncate_std_scale      = cfg["truncate_std_scale"],
        mask_type               = cfg["mask_type"],
        center_cross_half_width = cfg["center_cross_half_width"],
        clahe_kernel            = cfg["clahe_kernel"],
        clahe_clip              = cfg["clahe_clip"],
        clahe_nbins             = cfg["clahe_nbins"],
        flip_x                  = cfg["flip_x"],
        use_clahe               = cfg["use_clahe"],
        rescale_to_uint16       = cfg.get("rescale_to_uint16", False),
        unsharp_sigma           = cfg.get("unsharp_sigma", 0.0),
        unsharp_strength        = cfg.get("unsharp_strength", 1.0),
    )

    ang_data = utilities.read_ang(cfg["ang"], pat_obj.patshape, segment_grain_threshold=None)
    x0_flat  = np.ravel_multi_index((0, 0), ang_data.shape)
    patshape = pat_obj.patshape
    mask     = pat_obj.get_mask()

    # Pattern-center in physical coords expected by h2F:
    #   ang_data.pc = (xstar, ystar, zstar) in EDAX/TSL fractional convention
    #   x01 = (0.5 - xstar) * N_x * pixel_size   (x offset from pattern centre)
    #   x02 = (0.5 - ystar) * N_y * pixel_size   (y offset from pattern centre)
    #   DD  = zstar * N_x * pixel_size            (detector distance)
    # pixel_size is the same scale factor in all three terms, so it cancels inside
    # h2F (which only uses ratios like x01/DD).  Using physical units here keeps
    # the xo values interpretable and consistent.
    _pc        = np.asarray(ang_data.pc, dtype=float)
    pixel_size = cfg.get("pixel_size", 1.0)
    xo = np.array([
        (0.5 - _pc[0]) * patshape[1],   # x offset in physical units
        (0.5 - _pc[1]) * patshape[0],   # y offset in physical units
        _pc[2]         * patshape[1] * pixel_size,    # DD in physical units
    ])

    R_raw = pat_obj.read_pattern(x0_flat, process=False)
    R     = pat_obj.read_pattern(x0_flat, process=True)

    h0         = (patshape[1] // 2, patshape[0] // 2)
    crop_frac  = cfg["crop_fraction"]
    crop_row   = int(patshape[0] * (1 - crop_frac) / 2)
    crop_col   = int(patshape[1] * (1 - crop_frac) / 2)
    subset_slice = (slice(crop_row, -crop_row), slice(crop_col, -crop_col))

    x = np.arange(R.shape[1]) - h0[0]
    y = np.arange(R.shape[0]) - h0[1]
    X, Y = np.meshgrid(x, y, indexing="xy")
    xi = np.array([X[subset_slice].flatten(), Y[subset_slice].flatten()])
    subset_shape = X[subset_slice].shape

    valid = None
    if mask is not None:
        valid = mask[subset_slice].flatten()
        xi = xi[:, valid]
        print(f"  Mask: {valid.sum()} / {valid.size} pixels ({100*valid.mean():.1f}%)")

    def to_2d(arr):
        if valid is None:
            return arr.reshape(subset_shape)
        img = np.full(subset_shape[0] * subset_shape[1], np.nan)
        img[valid] = arr
        return img.reshape(subset_shape)

    ref_spline = interpolate.RectBivariateSpline(x, y, R.T, kx=5, ky=5)
    GRx = ref_spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    GRy = ref_spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    r   = ref_spline(xi[0], xi[1], grid=False).flatten()
    r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
    r = (r - r.mean()) / r_zmsv

    grad_mag = np.sqrt(GRx**2 + GRy**2)

    GR  = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)
    _1  = np.ones(xi.shape[1])
    _0  = np.zeros(xi.shape[1])
    out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0]**2, -xi[1]*xi[0]]])
    out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0]*xi[1], -xi[1]**2]])
    Jac  = np.vstack((out0, out1))
    NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]
    H_mat = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)
    cho_params = linalg.cho_factor(H_mat)

    cond = np.linalg.cond(H_mat)
    print(f"  Hessian condition number: {cond:.3e}")

    # FMT precompute
    _s = 2 ** (min(patshape).bit_length() - 1)
    rs = (patshape[0] - _s) // 2
    cs = (patshape[1] - _s) // 2
    init_subset_slice = (slice(rs, rs + _s), slice(cs, cs + _s))

    r_init = window_and_normalize(R[init_subset_slice])
    height, width = r_init.shape
    theta_arr  = np.linspace(0, np.pi, int(height), endpoint=False)
    radius_arr = np.linspace(0, height / 2, int(height + 1), endpoint=False)[1:]
    rg, tg = np.meshgrid(radius_arr, theta_arr, indexing="ij")
    x_fmt = 2**(np.log2(height)-1) + rg.flatten() * np.cos(tg.flatten())
    y_fmt = 2**(np.log2(height)-1) - rg.flatten() * np.sin(tg.flatten())
    X_fmt = np.arange(width)
    Y_fmt = np.arange(height)
    r_fft = np.fft.fftshift(np.fft.fft2(r_init))
    r_fmt, _ = FMT(r_fft, X_fmt, Y_fmt, x_fmt, y_fmt)

    get_pat = lambda idx: pat_obj.read_pattern(idx, process=True)

    return dict(
        label=label, cfg=cfg,
        pat_obj=pat_obj, get_pat=get_pat,
        xo=xo,
        R_raw=R_raw, R=R,
        patshape=patshape, h0=h0,
        crop_row=crop_row, crop_col=crop_col, subset_slice=subset_slice,
        xi=xi, subset_shape=subset_shape, valid=valid, to_2d=to_2d,
        GRx=GRx, GRy=GRy, grad_mag=grad_mag,
        r=r, r_zmsv=r_zmsv,
        NablaR_dot_Jac=NablaR_dot_Jac, cho_params=cho_params,
        r_init=r_init, r_fmt=r_fmt,
        X_fmt=X_fmt, Y_fmt=Y_fmt, x_fmt=x_fmt, y_fmt=y_fmt,
        init_subset_slice=init_subset_slice,
    )


# ============================================================
# PER-PATTERN FUNCTIONS
# ============================================================

def run_init_verbose(pipe, idx):
    get_pat = pipe["get_pat"]
    T = get_pat(idx)
    t_init = window_and_normalize(
        T[pipe["init_subset_slice"][0], pipe["init_subset_slice"][1]], alpha=0.2
    )
    t_fft = np.fft.fftshift(np.fft.fft2(t_init))
    t_fmt, _ = FMT(t_fft, pipe["X_fmt"], pipe["Y_fmt"], pipe["x_fmt"], pipe["y_fmt"])

    cc_angle = signal.fftconvolve(pipe["r_fmt"], t_fmt[::-1], mode="same").real
    theta = (np.argmax(cc_angle) - len(cc_angle) / 2) * np.pi / len(cc_angle)

    h_rot = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
    h0_loc = (t_init.shape[1]//2, t_init.shape[0]//2)
    t_rot = warp.deform_image(t_init, h_rot, h0_loc)
    cc_trans = signal.fftconvolve(pipe["r_init"], t_rot[::-1, ::-1], mode="same").real
    shift = np.unravel_index(np.argmax(cc_trans), cc_trans.shape) - np.array(cc_trans.shape) / 2

    h_init = conversions.xyt2h_partial(np.array([[-shift[1], -shift[0], -theta]]))[0]
    return h_init, cc_angle, cc_trans, t_init, t_rot, theta, shift


def run_optimize_verbose(pipe, idx, h_init):
    h_final, num_iter, residuals, norms = optimize_run(
        pipe["get_pat"], idx, h_init.copy(),
        pipe["r"], pipe["r_zmsv"], pipe["xi"],
        pipe["NablaR_dot_Jac"], pipe["cho_params"],
        max_iter=MAX_ITER, conv_tol=CONV_TOL, return_full=True,
    )
    return h_final, num_iter, residuals, norms


def warped_residual_image(pipe, idx, h):
    T = pipe["get_pat"](idx)
    h0_loc = (T.shape[1]//2, T.shape[0]//2)
    x_loc = np.arange(T.shape[1]) - h0_loc[0]
    y_loc = np.arange(T.shape[0]) - h0_loc[1]
    T_spline = interpolate.RectBivariateSpline(x_loc, y_loc, T.T, kx=5, ky=5)
    xi = pipe["xi"]
    t_def = warp.deform(xi, T_spline, h)
    t_p1, t_p99 = np.percentile(t_def, [1, 99])
    t_def = np.clip(t_def, t_p1, t_p99)
    t_zmsv = np.sqrt(((t_def - t_def.mean())**2).sum())
    if t_zmsv > 0:
        t_def = (t_def - t_def.mean()) / t_zmsv
    return pipe["to_2d"](pipe["r"] - t_def), pipe["to_2d"](t_def)


def run_with_snapshots(pipe, idx, h_start, snap_iters):
    T = pipe["get_pat"](idx)
    h0_loc = (T.shape[1]//2, T.shape[0]//2)
    x_loc = np.arange(T.shape[1]) - h0_loc[0]
    y_loc = np.arange(T.shape[0]) - h0_loc[1]
    T_spline = interpolate.RectBivariateSpline(x_loc, y_loc, T.T, kx=5, ky=5)
    xi = pipe["xi"]
    r  = pipe["r"]
    r_zmsv = pipe["r_zmsv"]
    NRJ = pipe["NablaR_dot_Jac"]
    cho = pipe["cho_params"]

    h = h_start.copy()
    snaps = {}
    for it in range(1, MAX_ITER + 1):
        t_def = warp.deform(xi, T_spline, h)
        t_p1, t_p99 = np.percentile(t_def, [1, 99])
        t_def = np.clip(t_def, t_p1, t_p99)
        t_zmsv = np.sqrt(((t_def - t_def.mean())**2).sum())
        if t_zmsv > 0:
            t_def = (t_def - t_def.mean()) / t_zmsv
        e = r - t_def
        if it in snap_iters:
            snaps[it] = pipe["to_2d"](e.copy())
        dC = 2 / r_zmsv * np.matmul(e, NRJ.T)
        dp = linalg.cho_solve(cho, -dC.reshape(-1, 1))[:, 0]
        norm = compute_dp_norm(dp, xi)
        Wp  = warp.W(h)
        Wdp = warp.W(dp)
        M   = np.matmul(Wp, np.linalg.inv(Wdp))
        h   = ((M / M[2, 2]) - np.eye(3)).reshape(9)[:8]
        if norm < CONV_TOL:
            for s in snap_iters:
                if s > it and s not in snaps:
                    t2 = warp.deform(xi, T_spline, h)
                    tz = np.sqrt(((t2 - t2.mean())**2).sum())
                    if tz > 0:
                        t2 = (t2 - t2.mean()) / tz
                    snaps[s] = pipe["to_2d"]((r - t2).copy())
            break
    return snaps


# ============================================================
# BUILD PIPELINES AND RUN
# ============================================================

pipes = {name: build_pipeline(cfg, name) for name, cfg in CONFIGS.items()}

results = {}
for name, pipe in pipes.items():
    idx = pipe["cfg"]["target_idx"]
    print(f"\nRunning {name}  (pattern idx={idx}) …")
    h_init, cc_ang, cc_trans, t_init_crop, t_rot, theta, shift = run_init_verbose(pipe, idx)
    h_final, niters, resids, norms = run_optimize_verbose(pipe, idx, h_init)
    T_raw = pipe["pat_obj"].read_pattern(idx, process=False)
    T_proc = pipe["get_pat"](idx)
    print(f"  Converged in {niters} iters,  final residual={resids[-1]:.4e}")
    print(f"  init h:  {np.round(h_init,  6)}")
    print(f"  final h: {np.round(h_final, 6)}")

    # --- Residual decomposition diagnostics ---
    R_raw_ref = pipe["pat_obj"].read_pattern(np.ravel_multi_index((0, 0), utilities.read_ang(pipe["cfg"]["ang"], pipe["patshape"], segment_grain_threshold=None).shape), process=False)
    print(f"  [diag] Raw ref  uint16 range:  [{R_raw_ref.min()}, {R_raw_ref.max()}]  "
          f"(fills {100*(R_raw_ref.max()-R_raw_ref.min())/65535:.1f}% of uint16)")
    print(f"  [diag] Raw tgt  uint16 range:  [{T_raw.min()}, {T_raw.max()}]  "
          f"(fills {100*(T_raw.max()-T_raw.min())/65535:.1f}% of uint16)")
    print(f"  [diag] r_zmsv (ref contrast) : {pipe['r_zmsv']:.4e}")
    print(f"  [diag] mean |grad R| (texture): {np.mean(np.abs(pipe['grad_mag'])):.4e}")
    r_vec = pipe["r"]
    T_proc_vals = pipe["get_pat"](idx)
    h0_loc = (T_proc_vals.shape[1]//2, T_proc_vals.shape[0]//2)
    x_loc = np.arange(T_proc_vals.shape[1]) - h0_loc[0]
    y_loc = np.arange(T_proc_vals.shape[0]) - h0_loc[1]
    from scipy import interpolate as _interp
    T_sp = _interp.RectBivariateSpline(x_loc, y_loc, T_proc_vals.T, kx=5, ky=5)
    t_raw_vec = T_sp(pipe["xi"][0], pipe["xi"][1], grid=False).flatten()
    t_zmsv = np.sqrt(((t_raw_vec - t_raw_vec.mean())**2).sum())
    t_norm = (t_raw_vec - t_raw_vec.mean()) / t_zmsv if t_zmsv > 0 else t_raw_vec
    print(f"  [diag] t_zmsv (tgt contrast) : {t_zmsv:.4e}")
    print(f"  [diag] MAE(r, t) before optim: {np.mean(np.abs(r_vec - t_norm)):.4e}  "
          f"(residual if h=0)")

    # --- Strain matrix from final homography ---
    Fe = conversions.h2F(h_final, pipe["xo"])
    epsilon, omega = conversions.F2strain(Fe, small_strain=True)
    np.set_printoptions(precision=6, suppress=True)
    print(f"\n  PC (xstar, ystar, zstar): {np.round(np.asarray(utilities.read_ang(pipe['cfg']['ang'], pipe['patshape'], segment_grain_threshold=None).pc, float), 4)}")
    print(f"  xo (pixel coords):        {np.round(pipe['xo'], 2)}")
    print(f"  Deformation gradient Fe:\n{Fe}")
    print(f"  Strain tensor epsilon (small-strain):\n{epsilon}")
    print(f"  Lattice rotation omega (rad):\n{omega}")
    print(f"  Diagonal strains:  e11={epsilon[0,0]:.4e}  e22={epsilon[1,1]:.4e}  e33={epsilon[2,2]:.4e}")
    print(f"  Shear strains:     e12={epsilon[0,1]:.4e}  e13={epsilon[0,2]:.4e}  e23={epsilon[1,2]:.4e}")
    print(f"  Rotations (mrad):  w12={1e3*omega[0,1]:.3f}  w13={1e3*omega[0,2]:.3f}  w23={1e3*omega[1,2]:.3f}")

    results[name] = dict(
        idx=idx, h_init=h_init, h_final=h_final,
        niters=niters, resids=resids, norms=norms,
        cc_ang=cc_ang, cc_trans=cc_trans,
        t_init_crop=t_init_crop, t_rot=t_rot,
        theta=theta, shift=shift,
        T_raw=T_raw, T_proc=T_proc,
    )

names  = list(pipes.keys())
HLABELS = ["h11","h12","h13","h21","h22","h23","h31","h32"]

# ============================================================
# FIGURE 1 — Reference pattern diagnostics (one column per dataset)
# ============================================================

fig1, axes = plt.subplots(2, 2*len(names), figsize=(8*len(names), 9))
fig1.suptitle("Reference pattern diagnostics", fontsize=13)

for col, name in enumerate(names):
    pipe = pipes[name]
    c    = col * 2
    color = pipe["cfg"]["color"]

    axes[0, c].imshow(pipe["R_raw"], cmap="gray")
    axes[0, c].set_title(f"{name}\nRaw reference")

    axes[0, c+1].imshow(pipe["R"], cmap="gray")
    axes[0, c+1].set_title("Processed reference")
    rect = Rectangle(
        (pipe["crop_col"], pipe["crop_row"]),
        pipe["patshape"][1] - 2*pipe["crop_col"],
        pipe["patshape"][0] - 2*pipe["crop_row"],
        linewidth=1.5, edgecolor="lime", facecolor="none",
    )
    axes[0, c+1].add_patch(rect)

    vgx = np.percentile(np.abs(pipe["GRx"]), 99)
    vgy = np.percentile(np.abs(pipe["GRy"]), 99)
    axes[1, c].imshow(pipe["to_2d"](pipe["GRx"]), cmap="RdBu", vmin=-vgx, vmax=vgx)
    axes[1, c].set_title("Gradient X (GRx)")

    axes[1, c+1].imshow(pipe["to_2d"](pipe["GRy"]), cmap="RdBu", vmin=-vgy, vmax=vgy)
    axes[1, c+1].set_title("Gradient Y (GRy)")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
fig1.savefig(f"{SAVE_DIR}/fig1_reference_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("\nFig 1 saved.")

# ============================================================
# FIGURE 2 — FMT / FCC initial guess diagnostics
# ============================================================

fig2, axes = plt.subplots(3, 2*len(names), figsize=(8*len(names), 12))
fig2.suptitle("Initial guess diagnostics (FMT angle → FCC translation)", fontsize=12)

for col, name in enumerate(names):
    pipe = pipes[name]
    res  = results[name]
    c    = col * 2
    idx  = res["idx"]

    # Row 0: windowed crop  |  angle CC curve
    axes[0, c].imshow(res["t_init_crop"], cmap="gray")
    axes[0, c].set_title(f"{name}  (idx={idx})\nWindowed crop for FMT")
    axes[0, c].axis("off")

    ax_cc = axes[0, c+1]
    ax_cc.plot(res["cc_ang"], color=pipe["cfg"]["color"])
    ax_cc.axvline(np.argmax(res["cc_ang"]), color="red", linestyle="--", linewidth=1.2)
    ax_cc.set_title(f"FMT angle CC\nθ = {np.degrees(res['theta']):.3f}°")
    ax_cc.set_xlabel("angle bin"); ax_cc.set_ylabel("CC value")

    # Row 1: rotated crop  |  translation CC map
    axes[1, c].imshow(res["t_rot"], cmap="gray")
    axes[1, c].set_title("After rotation correction")
    axes[1, c].axis("off")

    im = axes[1, c+1].imshow(res["cc_trans"], cmap="hot", origin="lower")
    axes[1, c+1].set_title(
        f"FCC translation CC\nΔx={res['shift'][1]:.1f} px, Δy={res['shift'][0]:.1f} px"
    )
    axes[1, c+1].axis("off")
    fig2.colorbar(im, ax=axes[1, c+1], fraction=0.046, pad=0.04)

    # Row 2: init h bar chart  |  residual map at init
    ax_h = axes[2, c]
    ax_h.bar(range(8), res["h_init"], color=pipe["cfg"]["color"])
    ax_h.set_xticks(range(8)); ax_h.set_xticklabels(HLABELS, fontsize=8)
    ax_h.set_title("Init homography values")
    ax_h.axhline(0, color="k", linewidth=0.8)

    resid_init, _ = warped_residual_image(pipe, idx, res["h_init"])
    vr = np.nanpercentile(np.abs(resid_init), 98)
    im2 = axes[2, c+1].imshow(resid_init, cmap="RdBu", vmin=-vr, vmax=vr)
    axes[2, c+1].set_title(f"Residual @ init\nMAE={np.nanmean(np.abs(resid_init)):.4e}")
    axes[2, c+1].axis("off")
    fig2.colorbar(im2, ax=axes[2, c+1], fraction=0.046, pad=0.04)

plt.tight_layout()
fig2.savefig(f"{SAVE_DIR}/fig2_init_guess_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("Fig 2 saved.")

# ============================================================
# FIGURE 3 — Warp quality: raw, processed, warped @init/@final, residuals
# ============================================================

fig3, axes = plt.subplots(len(names), 6, figsize=(22, 5*len(names)))
fig3.suptitle("Warp quality: init vs converged", fontsize=12)

for row, name in enumerate(names):
    pipe = pipes[name]
    res  = results[name]
    idx  = res["idx"]

    resid_init,  t_init_w  = warped_residual_image(pipe, idx, res["h_init"])
    resid_final, t_final_w = warped_residual_image(pipe, idx, res["h_final"])
    vr = np.nanpercentile(np.abs(resid_final), 99)

    # Shared vmin/vmax for the processed (z-score) patterns so reference and
    # target are on an identical colour scale — any brightness difference is real.
    proc_all = np.concatenate([pipe["R"].ravel(), res["T_proc"].ravel()])
    vp = np.percentile(np.abs(proc_all), 99)

    # Shared vmin/vmax for the ZMSV-normalised warped subsets.
    warp_all = np.concatenate([t_init_w[~np.isnan(t_init_w)], t_final_w[~np.isnan(t_final_w)]])
    vw = np.nanpercentile(np.abs(warp_all), 99)

    axes[row, 0].imshow(pipe["R"],     cmap="gray", vmin=-vp, vmax=vp); axes[row, 0].set_title("Processed ref")
    axes[row, 1].imshow(res["T_proc"], cmap="gray", vmin=-vp, vmax=vp); axes[row, 1].set_title("Processed target")
    axes[row, 2].imshow(t_init_w,      cmap="gray", vmin=-vw, vmax=vw); axes[row, 2].set_title("Warped @ init")
    axes[row, 3].imshow(t_final_w,     cmap="gray", vmin=-vw, vmax=vw); axes[row, 3].set_title("Warped @ final")

    im_ri = axes[row, 4].imshow(resid_init,  cmap="RdBu", vmin=-vr, vmax=vr)
    axes[row, 4].set_title(f"Residual @ init\nMAE={np.nanmean(np.abs(resid_init)):.4e}")
    fig3.colorbar(im_ri, ax=axes[row, 4], fraction=0.046, pad=0.04)

    im_rf = axes[row, 5].imshow(resid_final, cmap="RdBu", vmin=-vr, vmax=vr)
    axes[row, 5].set_title(f"Residual @ final\nMAE={np.nanmean(np.abs(resid_final)):.4e}")
    fig3.colorbar(im_rf, ax=axes[row, 5], fraction=0.046, pad=0.04)

    axes[row, 0].set_ylabel(f"{name}\n(idx={idx})", fontsize=9)
    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
fig3.savefig(f"{SAVE_DIR}/fig3_warp_quality.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("Fig 3 saved.")

# ============================================================
# FIGURE 4 — Convergence curves
# ============================================================

fig4, axes = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle("Convergence curves", fontsize=12)

for name in names:
    pipe = pipes[name]
    res  = results[name]
    iters = np.arange(1, len(res["resids"]) + 1)
    kw = dict(color=pipe["cfg"]["color"], linewidth=1.5,
               label=f"{name} (idx={res['idx']}, {res['niters']} iters)")
    axes[0].plot(iters, res["resids"], **kw)
    axes[1].plot(iters, res["norms"],  **kw)

for ax, title, ylabel in [
    (axes[0], "Residual vs iteration", "MAE residual"),
    (axes[1], "dp_norm vs iteration",  "dp_norm"),
]:
    ax.axhline(CONV_TOL, color="k", linestyle="--", linewidth=1, label=f"conv_tol={CONV_TOL}")
    ax.set_title(title); ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig4.savefig(f"{SAVE_DIR}/fig4_convergence_curves.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("Fig 4 saved.")

# ============================================================
# FIGURE 5 — Residual snapshots across iterations
# ============================================================

snap_iters = sorted(set(
    [1, 2, 3, 5] + list(np.linspace(1, MAX_ITER, N_ITER_FRAMES, dtype=int))
))

print("\nGenerating iteration snapshots for Fig 5 …")
snaps_all = {}
for name, pipe in pipes.items():
    idx = results[name]["idx"]
    snaps_all[name] = run_with_snapshots(pipe, idx, results[name]["h_init"], snap_iters)

n_snaps = len(snap_iters)
fig5, axes = plt.subplots(len(names), n_snaps, figsize=(3*n_snaps, 5*len(names)))
fig5.suptitle("Residual images across iterations", fontsize=11)

all_vals = np.concatenate([
    v.ravel() for snaps in snaps_all.values() for v in snaps.values()
    if not np.all(np.isnan(v))
])
vmax_global = np.nanpercentile(np.abs(all_vals), 99)

for row, name in enumerate(names):
    for col, it in enumerate(snap_iters):
        ax  = axes[row, col]
        img = snaps_all[name].get(it, np.full(pipes[name]["subset_shape"], np.nan))
        ax.imshow(img, cmap="RdBu", vmin=-vmax_global, vmax=vmax_global)
        ax.set_title(f"iter {it}", fontsize=8)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"{name}", fontsize=9)

plt.tight_layout()
fig5.savefig(f"{SAVE_DIR}/fig5_residual_snapshots.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("Fig 5 saved.")

# ============================================================
# FIGURE 6 — Intensity distributions: raw vs processed
# ============================================================

fig6, axes = plt.subplots(2, len(names), figsize=(7*len(names), 8))
fig6.suptitle("Pixel intensity distributions", fontsize=13)

for col, name in enumerate(names):
    pipe  = pipes[name]
    res   = results[name]
    color = pipe["cfg"]["color"]

    raw  = pipe["R_raw"].ravel().astype(float)
    proc = pipe["R"].ravel()

    # --- Raw ---
    ax_raw = axes[0, col]
    ax_raw.hist(raw, bins=128, color=color, alpha=0.8, density=True)
    ax_raw.axvline(np.mean(raw),   color="k",    linestyle="--", linewidth=1.2, label=f"mean={np.mean(raw):.1f}")
    ax_raw.axvline(np.median(raw), color="gray",  linestyle=":",  linewidth=1.2, label=f"median={np.median(raw):.1f}")
    ax_raw.set_title(f"{name}\nRaw pattern")
    ax_raw.set_xlabel("Pixel value"); ax_raw.set_ylabel("Density")
    ax_raw.legend(fontsize=8)
    stats_raw = (
        f"min={raw.min():.0f}  max={raw.max():.0f}\n"
        f"std={raw.std():.1f}  skew={float(np.mean(((raw-raw.mean())/raw.std())**3)):.2f}"
    )
    ax_raw.text(0.98, 0.97, stats_raw, transform=ax_raw.transAxes,
                fontsize=7.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # --- Processed ---
    ax_proc = axes[1, col]
    ax_proc.hist(proc, bins=128, color=color, alpha=0.8, density=True)
    ax_proc.axvline(np.mean(proc),   color="k",   linestyle="--", linewidth=1.2, label=f"mean={np.mean(proc):.3f}")
    ax_proc.axvline(np.median(proc), color="gray", linestyle=":",  linewidth=1.2, label=f"median={np.median(proc):.3f}")
    ax_proc.set_title("Processed pattern")
    ax_proc.set_xlabel("Pixel value"); ax_proc.set_ylabel("Density")
    ax_proc.legend(fontsize=8)
    stats_proc = (
        f"min={proc.min():.3f}  max={proc.max():.3f}\n"
        f"std={proc.std():.3f}  skew={float(np.mean(((proc-proc.mean())/proc.std())**3)):.2f}"
    )
    ax_proc.text(0.98, 0.97, stats_proc, transform=ax_proc.transAxes,
                 fontsize=7.5, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plt.tight_layout()
fig6.savefig(f"{SAVE_DIR}/fig6_intensity_distributions.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
print("Fig 6 saved.")

print(f"\nAll figures saved to: {SAVE_DIR}/")
plt.show()

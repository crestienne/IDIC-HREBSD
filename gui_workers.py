"""
gui_workers.py — QThread subclasses that run heavy work off the main thread.

  _StdoutCapture       — redirects stdout to a Qt signal during pipeline runs
  PipelineWorker       — runs the full DIC-HREBSD optimisation pipeline
  IPFWorker            — computes IPF colour maps from .ang Euler angles
  SegmentWorker        — segments grains by misorientation flood-fill
  VisWorker            — loads homographies and computes strain/rotation tensors
  PatternPreviewWorker — loads and processes a single pattern for live preview
"""

import sys
import io
import os
import traceback

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal


# ─────────────────────────────────────────────────────────────────────────────
# stdout capture (used inside PipelineWorker)
# ─────────────────────────────────────────────────────────────────────────────

class _StdoutCapture(io.TextIOBase):
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def write(self, msg):
        if msg.strip():
            self._signal.emit(msg.rstrip())
        return len(msg)

    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_PARAMS = {"roi_slice", "_ang_data", "_grain_ids", "ref_pattern_set", "_roi_grain_mask", "_roi_grain_id"}   # non-serialisable / not useful to log

def _write_params_txt(path: str, params: dict):
    """Write all GUI run parameters to a human-readable .txt file."""
    lines = ["DIC-HREBSD Run Parameters", "=" * 40, ""]
    for key, val in sorted(params.items()):
        if key in _SKIP_PARAMS:
            continue
        try:
            val_str = str(val)
        except Exception:
            val_str = "<unrepresentable>"
        lines.append(f"{key:<30} {val_str}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _parse_params_txt(path: str) -> dict:
    """Inverse of `_write_params_txt`: parse a saved params .txt back to a dict.

    Each non-header line is `{key:<30} {value_str}` where `value_str` came from
    `str(val)`.  Values are decoded via `ast.literal_eval` (covers int / float /
    bool / None / str / tuple / list / dict); on failure the raw string is kept
    so the caller can still see the original text.
    """
    import ast
    out: dict = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("=") or line.startswith("DIC-HREBSD"):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, val_str = parts[0], parts[1].strip()
            try:
                out[key] = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                out[key] = val_str
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ANG background loader
# ─────────────────────────────────────────────────────────────────────────────

class AngLoaderWorker(QThread):
    """Load a .ang file in the background and cache the result on the wizard."""
    done_signal  = pyqtSignal(object)   # ang_data namedtuple
    error_signal = pyqtSignal(str)

    def __init__(self, ang_path: str, patshape: tuple):
        super().__init__()
        self.ang_path = ang_path
        self.patshape = patshape

    def run(self):
        try:
            import utilities
            ang_data = utilities.read_ang(
                self.ang_path, self.patshape, segment_grain_threshold=None
            )
            self.done_signal.emit(ang_data)
        except Exception:
            self.error_signal.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    log_signal  = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = _StdoutCapture(self.log_signal)
        try:
            self._run_pipeline()
            self.done_signal.emit(True, "Pipeline complete.")
        except Exception:
            self.done_signal.emit(False, traceback.format_exc())
        finally:
            sys.stdout = old_stdout

    def _run_pipeline(self):
        import Data
        import utilities

        p = self.params

        # Pick the optimizer module — IC-GN by default, reversed-role IC-GN
        # when selected on the run page.  Both expose `optimize(...)` with a
        # compatible signature for InitType.NONE.
        _opt = p.get("optimizer", "icgn")
        if _opt == "icgn_reversed":
            import get_homography_cpu_reversed as core
            self.log_signal.emit(
                "Optimizer: IC-GN reversed roles (get_homography_cpu_reversed) — "
                "per-pattern image plays REFERENCE; designated reference plays "
                "TARGET.  Returned homographies will be inverted before strain."
            )
        else:
            import get_homography_cpu as core
            self.log_signal.emit("Optimizer: IC-GN (homography, get_homography_cpu)")

        self.log_signal.emit("Loading UP2 file…")
        pat_obj = Data.UP2(p["up2"])
        pat_obj.set_processing(
            low_pass_sigma=p["low_pass_sigma"],
            high_pass_sigma=p["high_pass_sigma"],
            truncate_std_scale=3.0,
            mask_type=p["mask_type"],
            center_cross_half_width=6,
            flip_x=p["flip_x"],
            gamma=p.get("gamma", 0.8),
        )
        print(pat_obj)

        ang_data = p.get("_ang_data")
        if ang_data is None:
            self.log_signal.emit("Reading ANG file…")
            ang_data = utilities.read_ang(
                p["ang"], pat_obj.patshape, segment_grain_threshold=None
            )
        else:
            self.log_signal.emit("Using cached ANG data.")

        pc_ref = ang_data.pc
        self.log_signal.emit(f"PC: {pc_ref}")

        # Frame transforms (used for strain frame conversion)
        if p.get("identity_rotation", False):
            R_s2d = np.eye(3)
            R_d2s = np.eye(3)
            self.log_signal.emit(
                "Rotation matrix R = identity (identity_rotation checkbox is on) — "
                "no sample↔detector frame change will be applied to ε / ω."
            )
        else:
            R_s2d = utilities.get_sample_to_detector_rotation(p["det_tilt"], p["tilt"])
            R_d2s = R_s2d.T
            with np.printoptions(precision=4, suppress=True, sign="+"):
                self.log_signal.emit(
                    f"Rotation matrix (sample → detector)  "
                    f"[sample_tilt={p['tilt']}°, det_tilt={p['det_tilt']}°]:\n"
                    f"{R_s2d}\n"
                    f"Rotation matrix (detector → sample)  (inverse / transpose):\n"
                    f"{R_d2s}"
                )

        os.makedirs(p["output_dir"], exist_ok=True)

        _comp = p.get("component", "run")
        _date = p.get("date", "")
        params_txt_path = os.path.join(p["output_dir"], f"{_comp}_params_{_date}.txt")
        _write_params_txt(params_txt_path, p)
        self.log_signal.emit(f"Parameters saved → {os.path.basename(params_txt_path)}")

        ref_mode = p.get("ref_mode", "single")

        if ref_mode == "per_grain":
            h, h_guess, iterations, residuals, dp_norms, eff_rows, eff_cols = \
                self._run_per_grain(p, pat_obj, ang_data, core)
        elif ref_mode == "simulated":
            h, h_guess, iterations, residuals, dp_norms, eff_rows, eff_cols = \
                self._run_simulated(p, pat_obj, ang_data, core)
            # Use geometry-page PC (tuned / refined by user) for h2F and drift
            # correction — it matches what was used to generate the simulated pattern.
            pc_ref = tuple(p.get("pc_edax", ang_data.pc))
            self.log_signal.emit(f"PC (geometry page, used for simulated ref): {pc_ref}")
        else:
            h, h_guess, iterations, residuals, dp_norms, eff_rows, eff_cols = \
                self._run_single(p, pat_obj, ang_data, core)

        # ── Reversed-role IC-GN: invert homographies before downstream ────────
        # core.optimize for icgn_reversed returns h that warps new-reference →
        # new-target (= former target → former reference).  Invert so PC
        # correction, h2F, and F2strain see the standard convention.
        if _opt == "icgn_reversed":
            from get_homography_cpu_reversed import invert_homographies
            self.log_signal.emit(
                "Inverting reversed-role homographies → standard convention "
                "(former-ref → former-target) so PC correction and strain "
                "match standard IC-GN."
            )
            h = invert_homographies(h)
            if h_guess is not None:
                h_guess = invert_homographies(h_guess)

        # ── PC drift correction ───────────────────────────────────────────────
        # For per-grain mode h spans the full scan (NaN where grain_id == 0).
        # Temporarily replace NaN rows with zeros so the matrix math doesn't
        # propagate NaN into neighbouring positions, then restore afterwards.
        nan_mask = None
        if p.get("apply_pc_correction", True):
            from pc_homography_correction import correct_homographies
            self.log_signal.emit("Applying PC drift correction…")
            if ref_mode == "per_grain":
                nan_mask = np.any(np.isnan(h), axis=1)
                h[nan_mask] = 0.0

            # Compute ROI-relative ref_position so the PC-correction "zero
            # drift" anchor lands at the actual IC-GN reference pattern.
            # For single / simulated modes the reference is at p["ref_position"]
            # in FULL-scan coords; for per_grain there is no single reference,
            # so we fall back to (0, 0).
            if ref_mode == "per_grain":
                pc_ref_position = (0, 0)
            else:
                _ref_full = p.get("ref_position", (0, 0))
                _roi      = p.get("roi_slice", None)
                if _roi is not None:
                    _rr = int(_ref_full[0]) - _roi[0].start
                    _rc = int(_ref_full[1]) - _roi[1].start
                else:
                    _rr, _rc = int(_ref_full[0]), int(_ref_full[1])
                # Clamp if the reference falls outside the optimised region —
                # fall back to (0, 0) and warn rather than crashing.
                if not (0 <= _rr < eff_rows and 0 <= _rc < eff_cols):
                    self.log_signal.emit(
                        f"⚠ ref_position {_ref_full} outside ROI ({eff_rows}×{eff_cols}); "
                        f"PC correction will anchor at (0, 0) instead."
                    )
                    pc_ref_position = (0, 0)
                else:
                    pc_ref_position = (_rr, _rc)
            self.log_signal.emit(
                f"PC correction ref_position (ROI-relative): {pc_ref_position}"
            )

            h, _ = correct_homographies(
                h=h,
                scan_shape=(eff_rows, eff_cols),
                step_size_um=p["step_size"],
                pc_ref=pc_ref,
                patshape=pat_obj.patshape,
                pixel_size_um=p["pixel_size"],
                sample_tilt_deg=p["tilt"],
                detector_tilt_deg=p["det_tilt"],
                convention=p.get("scan_strategy", "standard"),
                ref_position=pc_ref_position,
            )
            if nan_mask is not None:
                h[nan_mask] = np.nan
            self.log_signal.emit("PC drift correction applied.")

        comp   = p["component"]
        date   = p["date"]
        folder = p["output_dir"]

        np.save(os.path.join(folder, f"{comp}_homographies_{date}.npy"), h)
        np.save(os.path.join(folder, f"{comp}_iterations_{date}.npy"),   iterations)
        np.save(os.path.join(folder, f"{comp}_residuals_{date}.npy"),    residuals)
        np.save(os.path.join(folder, f"{comp}_dp_norms_{date}.npy"),     dp_norms)
        if h_guess is not None:
            np.save(os.path.join(folder, f"{comp}_h_guess_{date}.npy"), h_guess)

        # ── Strain / rotation ─────────────────────────────────────────────────
        self.log_signal.emit("Computing strain and rotation tensors…")
        import conversions

        pc_edax        = np.asarray(pc_ref, dtype=float)
        pc_bruker      = conversions.Edax_to_Bruker_PC(pc_edax)
        xo             = conversions.Bruker_to_fractional_PC(pc_bruker, pat_obj.patshape)
        self.log_signal.emit(
            f"[strain step]  PC chain  "
            f"EDAX={tuple(np.round(pc_edax, 4))}  →  "
            f"Bruker={tuple(np.round(pc_bruker, 4))}  →  "
            f"xo={tuple(np.round(xo, 4))}"
        )

        # h2F / F2strain don't handle NaN; compute only on valid rows.
        # This applies to both per-grain mode and single-reference grain-ROI runs.
        small_strain = bool(p.get("small_strain", False))
        self.log_signal.emit(
            f"[strain step]  formulation = {'small strain' if small_strain else 'finite strain'}"
        )
        if ref_mode == "per_grain" or np.any(np.isnan(h)):
            valid = ~np.any(np.isnan(h), axis=1)
            h_valid = h[valid]
            F_valid        = conversions.h2F(h_valid, xo)
            eps_v, omega_v = conversions.F2strain(F_valid, small_strain=small_strain)
            R = (np.eye(3) if p.get("identity_rotation", False)
             else utilities.get_sample_to_detector_rotation(p["det_tilt"], p["tilt"]))
            for i in range(eps_v.shape[0]):
                eps_v[i]   = R.T @ eps_v[i]   @ R
                omega_v[i] = R.T @ omega_v[i] @ R
            N   = eff_rows * eff_cols
            nan3 = np.full((N, 3, 3), np.nan)
            F       = nan3.copy()
            epsilon = nan3.copy()
            omega   = nan3.copy()
            F[valid]       = F_valid
            epsilon[valid] = eps_v
            omega[valid]   = omega_v
        else:
            F              = conversions.h2F(h, xo)
            epsilon, omega = conversions.F2strain(F, small_strain=small_strain)
            R = (np.eye(3) if p.get("identity_rotation", False)
             else utilities.get_sample_to_detector_rotation(p["det_tilt"], p["tilt"]))
            for i in range(epsilon.shape[0]):
                epsilon[i] = R.T @ epsilon[i] @ R
                omega[i]   = R.T @ omega[i]   @ R

        def _2d(arr):
            return arr.reshape(eff_rows, eff_cols)

        # ── Euler-angle estimates from omega ──────────────────────────────────
        # Relative: omega itself exponentiated to a rotation, then Bunge.
        # Absolute: compose with the per-pixel Hough Euler from the .ang file
        #          → R_abs = R_omega · R_hough, then extract Bunge.
        # NaN-safe: omega rows with NaNs propagate via skew_to_R; we mask.
        nan_mask_eu = np.any(np.isnan(omega.reshape(-1, 9)), axis=1)
        omega_safe = np.where(nan_mask_eu[:, None, None], 0.0, omega)
        rel_deg = conversions.omega_array_to_bunge(omega_safe, degrees=True)
        rel_deg[nan_mask_eu] = np.nan

        # Per-pattern reference Euler from the .ang file at the same scan
        # position.  Slice the ROI if one is active so shapes match.
        eulers_full = ang_data.eulers           # (full_R, full_C, 3) in radians
        full_R, full_C = ang_data.shape
        roi_slice = p.get("roi_slice", None)
        if roi_slice is not None:
            eulers_roi = eulers_full[roi_slice[0], roi_slice[1], :]
        else:
            eulers_roi = eulers_full
        eulers_flat = eulers_roi.reshape(-1, 3)
        if eulers_flat.shape[0] != omega.shape[0]:
            # Shape mismatch — fall back to broadcasting a single Euler so
            # the abs panel still works.  This shouldn't happen in practice.
            eulers_flat = np.tile(eulers_flat[:1], (omega.shape[0], 1))
        R_omega = conversions.skew_to_rotation_matrix_array(omega_safe)
        R_hough = conversions.bunge_to_rotation_matrix(
            eulers_flat[:, 0], eulers_flat[:, 1], eulers_flat[:, 2], degrees=False
        )
        R_abs   = R_omega @ R_hough
        abs_deg = conversions.rotation_matrix_array_to_bunge(R_abs, degrees=True)
        abs_deg[nan_mask_eu] = np.nan

        results_npy_path = os.path.join(folder, f"{comp}_results_{date}.npy")
        _results_dict = {
            "h11": _2d(h[:, 0]), "h12": _2d(h[:, 1]), "h13": _2d(h[:, 2]),
            "h21": _2d(h[:, 3]), "h22": _2d(h[:, 4]), "h23": _2d(h[:, 5]),
            "h31": _2d(h[:, 6]), "h32": _2d(h[:, 7]),
            "e11": _2d(epsilon[:, 0, 0]), "e12": _2d(epsilon[:, 0, 1]), "e13": _2d(epsilon[:, 0, 2]),
            "e22": _2d(epsilon[:, 1, 1]), "e23": _2d(epsilon[:, 1, 2]), "e33": _2d(epsilon[:, 2, 2]),
            "w13": _2d(np.degrees(omega[:, 0, 2])),
            "w21": _2d(np.degrees(omega[:, 1, 0])),
            "w32": _2d(np.degrees(omega[:, 2, 1])),
            "phi1_rel": _2d(rel_deg[:, 0]),
            "Phi_rel":  _2d(rel_deg[:, 1]),
            "phi2_rel": _2d(rel_deg[:, 2]),
            "phi1_abs": _2d(abs_deg[:, 0]),
            "Phi_abs":  _2d(abs_deg[:, 1]),
            "phi2_abs": _2d(abs_deg[:, 2]),
            "F":   F,
            "rows": np.array(eff_rows),
            "cols": np.array(eff_cols),
        }
        # FMT-FCC initial-guess homography (per pattern, pre-IC-GN refinement).
        # Saved alongside the converged h so the visualizer can plot the strain
        # the initial guess alone would have produced — useful for diagnosing
        # whether IC-GN refinement is doing meaningful work or just locking in
        # the FMT-FCC guess.
        if h_guess is not None:
            _results_dict["h_guess"] = h_guess.reshape(eff_rows, eff_cols, 8)
        np.save(results_npy_path, _results_dict)
        self.log_signal.emit(f"Strain/rotation saved → {comp}_results_{date}.npy")
        self.log_signal.emit(f"All results saved to: {folder}")

    # ── Single-reference path ─────────────────────────────────────────────────

    def _run_single(self, p, pat_obj, ang_data, core):
        import time
        import conversions

        x0 = np.ravel_multi_index(p["ref_position"], ang_data.shape)
        self.log_signal.emit(
            f"Reference pattern index: {x0}  "
            f"(row={p['ref_position'][0]}, col={p['ref_position'][1]})"
        )

        euler_angles_ref = ang_data.eulers[np.unravel_index(x0, ang_data.shape)]
        pc_ref           = ang_data.pc

        # PC vector needed by the FULL initial-guess path (xyt2h)
        pc_edax_arr = np.asarray(pc_ref, dtype=float)
        pc_bruker   = conversions.Edax_to_Bruker_PC(pc_edax_arr)
        pc_xo       = conversions.Bruker_to_fractional_PC(pc_bruker, pat_obj.patshape)
        self.log_signal.emit(
            f"[init guess / single ref]  PC chain  "
            f"EDAX={tuple(np.round(pc_edax_arr, 4))}  →  "
            f"Bruker={tuple(np.round(pc_bruker, 4))}  →  "
            f"xo={tuple(np.round(pc_xo, 4))}"
        )

        # Reference-pattern preprocessing override:
        # If the user enabled "different preprocessing for the reference" in
        # Step 5, read pattern x0 with the override settings *now*, restore
        # pat_obj's regular settings, and pass the precomputed ref pattern to
        # the optimizer.  All other patterns continue to use the main settings.
        ref_pat_override = None
        ref_pp = p.get("ref_preprocessing")
        if ref_pp and ref_pp.get("enabled"):
            saved = (
                pat_obj.high_pass_sigma, pat_obj.low_pass_sigma, pat_obj.gamma,
                pat_obj.mask_type,
            )
            pat_obj.high_pass_sigma = float(ref_pp["high_pass_sigma"])
            pat_obj.low_pass_sigma  = float(ref_pp["low_pass_sigma"])
            pat_obj.gamma           = float(ref_pp["gamma"])
            pat_obj.mask_type       = (None if ref_pp["mask_type"] == "None"
                                        else ref_pp["mask_type"])
            try:
                ref_pat_override = pat_obj.read_pattern(int(x0), process=True)
                self.log_signal.emit(
                    f"Reference preprocessing override: hp={ref_pp['high_pass_sigma']}  "
                    f"lp={ref_pp['low_pass_sigma']}  γ={ref_pp['gamma']}  "
                    f"mask={ref_pp['mask_type']}"
                )
            finally:
                (pat_obj.high_pass_sigma, pat_obj.low_pass_sigma, pat_obj.gamma,
                 pat_obj.mask_type) = saved

        optimize_params = dict(
            init_type=p["init_type"],
            crop_fraction=p["crop_fraction"],
            max_iter=p["max_iter"],
            conv_tol=1e-3,
            n_jobs=p["n_jobs"],
            verbose=True,
            roi_slice=p.get("roi_slice", None),
            scan_shape=ang_data.shape,
            mask=pat_obj.get_mask(),
            use_simulated_reference=False,
            master_pattern_path=None,
            euler_angles_ref=euler_angles_ref,
            pc_ref=pc_ref,
            pc_xo=pc_xo,
            tilt_deg=p["tilt"],
            ref_pat_override=ref_pat_override,
            spectral_match_ref=p.get("spectral_match_ref", False),
            perspective_regularization=p.get("perspective_regularization", 0.0),
            rotate_patterns_90=p.get("rotate_patterns_90", False),
            debug_gradients=False,
        )

        # Drop ref_pat_override if it's None to avoid passing it to optimizers
        # that don't accept the keyword.
        if optimize_params.get("ref_pat_override") is None:
            optimize_params.pop("ref_pat_override", None)

        self.log_signal.emit("Starting optimization…")
        t0     = time.perf_counter()
        result = core.optimize(pat_obj, x0, **optimize_params)
        if len(result) == 5:
            h, h_guess, iterations, residuals, dp_norms = result
        else:
            h, iterations, residuals, dp_norms = result
            h_guess = None
        dt    = time.perf_counter() - t0
        n_pat = iterations.size
        self.log_signal.emit(
            f"Optimization complete: {dt:.1f} s total, "
            f"{dt / n_pat * 1000:.2f} ms/pattern"
        )

        if h.ndim == 3:
            eff_rows, eff_cols = h.shape[0], h.shape[1]
            h           = h.reshape(-1, 8)
            if h_guess is not None:
                h_guess = h_guess.reshape(-1, 8)
            iterations  = iterations.flatten()
            residuals   = residuals.flatten()
            dp_norms    = dp_norms.flatten()
        else:
            roi_slice = p.get("roi_slice", None)
            if roi_slice is not None:
                eff_rows = roi_slice[0].stop - roi_slice[0].start
                eff_cols = roi_slice[1].stop - roi_slice[1].start
            else:
                eff_rows, eff_cols = ang_data.shape

        # NaN out pixels inside the bounding box that don't belong to the
        # selected grain (only applies when grain ROI mode is active)
        grain_mask_full = p.get("_roi_grain_mask")
        if grain_mask_full is not None:
            roi_slice = p.get("roi_slice", None)
            if roi_slice is not None:
                bbox_mask = grain_mask_full[roi_slice[0], roi_slice[1]].ravel()
            else:
                bbox_mask = grain_mask_full.ravel()
            non_grain = ~bbox_mask
            h[non_grain]          = np.nan
            iterations[non_grain] = np.nan
            residuals[non_grain]  = np.nan
            dp_norms[non_grain]   = np.nan
            if h_guess is not None:
                h_guess[non_grain] = np.nan
            n_kept = int(bbox_mask.sum())
            self.log_signal.emit(
                f"Grain mask applied — {n_kept}/{len(bbox_mask)} pixels kept."
            )

        return h, h_guess, iterations, residuals, dp_norms, eff_rows, eff_cols

    # ── Simulated-reference path ──────────────────────────────────────────────

    def _run_simulated(self, p, pat_obj, ang_data, core):
        import time
        import conversions

        euler_rad = np.deg2rad(p["euler_deg"])   # (phi1, Phi, phi2) radians
        # Use geometry-page PC (which the user tuned via spinboxes, live tuner,
        # or PC/Euler refinement), not ang_data.pc from the scan file.
        pc_ref = tuple(p["pc_edax"])

        self.log_signal.emit(
            f"Simulated reference — φ₁={p['euler_deg'][0]:.2f}°  "
            f"Φ={p['euler_deg'][1]:.2f}°  φ₂={p['euler_deg'][2]:.2f}°"
        )
        self.log_signal.emit(f"PC (geometry page): {pc_ref}")
        self.log_signal.emit(f"Master pattern: {p['master_pattern_path']}")

        # PC vector needed by the FULL initial-guess path (xyt2h)
        pc_edax_arr = np.asarray(pc_ref, dtype=float)
        pc_bruker   = conversions.Edax_to_Bruker_PC(pc_edax_arr)
        pc_xo       = conversions.Bruker_to_fractional_PC(pc_bruker, pat_obj.patshape)
        self.log_signal.emit(
            f"[init guess / simulated ref]  PC chain  "
            f"EDAX={tuple(np.round(pc_edax_arr, 4))}  →  "
            f"Bruker={tuple(np.round(pc_bruker, 4))}  →  "
            f"xo={tuple(np.round(pc_xo, 4))}"
        )

        optimize_params = dict(
            init_type            = p["init_type"],
            crop_fraction        = p["crop_fraction"],
            max_iter             = p["max_iter"],
            conv_tol             = 1e-3,
            n_jobs               = p["n_jobs"],
            verbose              = True,
            roi_slice            = p.get("roi_slice", None),
            scan_shape           = ang_data.shape,
            mask                 = pat_obj.get_mask(),
            use_simulated_reference = True,
            master_pattern_path  = p["master_pattern_path"],
            euler_angles_ref     = euler_rad,
            pc_ref               = pc_ref,
            pc_xo                = pc_xo,
            tilt_deg             = p["tilt"],
            detector_tilt_deg    = p.get("det_tilt", 0.0),
            spectral_match_ref   = p.get("spectral_match_ref", False),
            perspective_regularization = p.get("perspective_regularization", 0.0),
            rotate_patterns_90   = p.get("rotate_patterns_90", False),
            debug_gradients      = False,
        )

        self.log_signal.emit("Starting optimization with simulated reference…")
        t0     = time.perf_counter()
        result = core.optimize(pat_obj, 0, **optimize_params)
        if len(result) == 5:
            h, h_guess, iterations, residuals, dp_norms = result
        else:
            h, iterations, residuals, dp_norms = result
            h_guess = None
        dt    = time.perf_counter() - t0
        n_pat = iterations.size
        self.log_signal.emit(
            f"Optimization complete: {dt:.1f} s total, "
            f"{dt / n_pat * 1000:.2f} ms/pattern"
        )

        if h.ndim == 3:
            eff_rows, eff_cols = h.shape[0], h.shape[1]
            h          = h.reshape(-1, 8)
            if h_guess is not None:
                h_guess = h_guess.reshape(-1, 8)
            iterations = iterations.flatten()
            residuals  = residuals.flatten()
            dp_norms   = dp_norms.flatten()
        else:
            roi_slice = p.get("roi_slice", None)
            if roi_slice is not None:
                eff_rows = roi_slice[0].stop - roi_slice[0].start
                eff_cols = roi_slice[1].stop - roi_slice[1].start
            else:
                eff_rows, eff_cols = ang_data.shape

        return h, h_guess, iterations, residuals, dp_norms, eff_rows, eff_cols

    # ── Per-grain path ────────────────────────────────────────────────────────

    def _run_per_grain(self, p, pat_obj, ang_data, core):
        import time
        import conversions

        rps       = p["ref_pattern_set"]
        grain_ids = p.get("_grain_ids")
        if grain_ids is None:
            raise RuntimeError(
                "Per-grain mode requires grain_ids — run Step 3 segmentation first."
            )

        eff_rows, eff_cols = ang_data.shape
        N = eff_rows * eff_cols

        h_full          = np.full((N, 8),  np.nan, dtype=np.float64)
        h_guess_full    = np.full((N, 8),  np.nan, dtype=np.float64)
        iterations_full = np.full(N,       np.nan, dtype=np.float64)
        residuals_full  = np.full(N,       np.nan, dtype=np.float64)
        dp_norms_full   = np.full(N,       np.nan, dtype=np.float64)
        has_h_guess     = False

        # PC vector needed by the FULL initial-guess path (xyt2h)
        _pc_edax_pg   = np.asarray(ang_data.pc, dtype=float)
        _pc_bruker_pg = conversions.Edax_to_Bruker_PC(_pc_edax_pg)
        _pc_xo_pg     = conversions.Bruker_to_fractional_PC(_pc_bruker_pg, pat_obj.patshape)
        self.log_signal.emit(
            f"[init guess / per-grain]  PC chain  "
            f"EDAX={tuple(np.round(_pc_edax_pg, 4))}  →  "
            f"Bruker={tuple(np.round(_pc_bruker_pg, 4))}  →  "
            f"xo={tuple(np.round(_pc_xo_pg, 4))}"
        )

        common_opt = dict(
            init_type=p["init_type"],
            crop_fraction=p["crop_fraction"],
            max_iter=p["max_iter"],
            conv_tol=1e-3,
            n_jobs=p["n_jobs"],
            verbose=True,
            scan_shape=ang_data.shape,
            mask=pat_obj.get_mask(),
            use_simulated_reference=False,
            master_pattern_path=None,
            pc_ref=ang_data.pc,
            pc_xo=_pc_xo_pg,
            tilt_deg=p["tilt"],
            spectral_match_ref=p.get("spectral_match_ref", False),
            perspective_regularization=p.get("perspective_regularization", 0.0),
            rotate_patterns_90=p.get("rotate_patterns_90", False),
            debug_gradients=False,
        )

        t_total = time.perf_counter()

        for gi, entry in enumerate(rps):
            gid = entry.grain_id
            self.log_signal.emit(
                f"Grain {gid}  ({gi+1}/{len(rps)})  —  "
                f"ref at row={entry.ref_row}, col={entry.ref_col}"
            )

            # Pixel mask for this grain and its bounding box
            grain_mask = (grain_ids == gid)
            row_idx, col_idx = np.where(grain_mask)
            rmin, rmax = int(row_idx.min()), int(row_idx.max())
            cmin, cmax = int(col_idx.min()), int(col_idx.max())
            roi_slice = (slice(rmin, rmax + 1), slice(cmin, cmax + 1))

            euler_ref = np.array(entry.euler) if entry.euler is not None \
                else ang_data.eulers[entry.ref_row, entry.ref_col]

            t0     = time.perf_counter()
            result = core.optimize(
                pat_obj, entry.ref_pat_idx,
                roi_slice=roi_slice,
                euler_angles_ref=euler_ref,
                **common_opt,
            )
            if len(result) == 5:
                h_g, h_guess_g, iters_g, resid_g, dp_g = result
                has_h_guess = True
            else:
                h_g, iters_g, resid_g, dp_g = result
                h_guess_g = None
            dt = time.perf_counter() - t0

            # Flatten to (bbox_rows * bbox_cols, ...)
            if h_g.ndim == 3:
                h_g      = h_g.reshape(-1, 8)
                if h_guess_g is not None:
                    h_guess_g = h_guess_g.reshape(-1, 8)
                iters_g  = iters_g.flatten()
                resid_g  = resid_g.flatten()
                dp_g     = dp_g.flatten()

            # Indices within the bbox that belong to this grain
            bbox_mask = grain_mask[rmin:rmax+1, cmin:cmax+1]
            bbox_flat = np.where(bbox_mask.ravel())[0]

            # Corresponding flat indices in the full scan
            full_flat = row_idx * eff_cols + col_idx

            h_full[full_flat]          = h_g[bbox_flat]
            iterations_full[full_flat] = iters_g[bbox_flat]
            residuals_full[full_flat]  = resid_g[bbox_flat]
            dp_norms_full[full_flat]   = dp_g[bbox_flat]
            if h_guess_g is not None:
                h_guess_full[full_flat] = h_guess_g[bbox_flat]

            n_pat = iters_g.size
            self.log_signal.emit(
                f"  → done in {dt:.1f} s  ({dt / n_pat * 1000:.2f} ms/pattern)"
            )

        self.log_signal.emit(
            f"All grains complete in {time.perf_counter() - t_total:.1f} s."
        )

        h_guess_out = h_guess_full if has_h_guess else None
        return (h_full, h_guess_out, iterations_full, residuals_full,
                dp_norms_full, eff_rows, eff_cols)


# ─────────────────────────────────────────────────────────────────────────────
# IPF colour-map worker
# ─────────────────────────────────────────────────────────────────────────────

class IPFWorker(QThread):
    done_signal = pyqtSignal(object, str)   # rgb_map (ndarray) | None, error_msg

    def __init__(self, ang_path: str, patshape: tuple, direction: np.ndarray,
                 ang_data=None):
        super().__init__()
        self.ang_path  = ang_path
        self.patshape  = patshape
        self.direction = direction
        self.ang_data  = ang_data

    def run(self):
        try:
            from ipf_map import compute_ipf_colors
            ang_data = self.ang_data
            if ang_data is None:
                import utilities
                ang_data = utilities.read_ang(
                    self.ang_path, self.patshape, segment_grain_threshold=None
                )
            rgb = compute_ipf_colors(ang_data.eulers, self.direction)
            self.done_signal.emit(rgb, "")
        except Exception:
            self.done_signal.emit(None, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Grain segmentation worker
# ─────────────────────────────────────────────────────────────────────────────

class SegmentWorker(QThread):
    done_signal = pyqtSignal(object, object, str)  # grain_ids, kam, error_msg

    def __init__(self, ang_path: str, patshape: tuple, threshold: float,
                 min_grain_size: int = 1, ang_data=None):
        super().__init__()
        self.ang_path       = ang_path
        self.patshape       = patshape
        self.threshold      = threshold
        self.min_grain_size = min_grain_size
        self.ang_data       = ang_data

    def run(self):
        try:
            import utilities, segment
            ang_data = self.ang_data
            if ang_data is None:
                ang_data = utilities.read_ang(
                    self.ang_path, self.patshape, segment_grain_threshold=None
                )
            grain_ids, kam = segment.segment_grains(
                ang_data.quats, self.threshold, self.min_grain_size, progress=False
            )
            self.done_signal.emit(grain_ids, kam, "")
        except Exception:
            self.done_signal.emit(None, None, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation computation worker
# ─────────────────────────────────────────────────────────────────────────────

class VisWorker(QThread):
    """Loads homographies and computes strain/rotation purely in numpy (no GUI)."""
    results_signal = pyqtSignal(dict)
    error_signal   = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.results_signal.emit(self._compute())
        except Exception:
            self.error_signal.emit(traceback.format_exc())

    def _compute(self) -> dict:
        import conversions, utilities
        p = self.params

        # ── Fast path: load pre-computed results from npy ─────────────────────
        npy_results = p.get("npz_path", "")   # field name kept for compat
        if npy_results and os.path.isfile(npy_results):
            return self._load_from_npz(npy_results, p)

        # ── Legacy path: compute from raw homographies npy ────────────────────
        h = np.load(p["npy_path"])
        rows, cols = p["rows"], p["cols"]
        if h.ndim != 2 or h.shape[-1] != 8:
            h = h.reshape(-1, 8)
        if h.shape[0] != rows * cols:
            raise ValueError(
                f"Homography file has {h.shape[0]} entries but rows×cols = "
                f"{rows}×{cols} = {rows * cols}. Check the Rows / Columns values."
            )

        h11, h12, h13 = h[:, 0], h[:, 1], h[:, 2]
        h21, h22, h23 = h[:, 3], h[:, 4], h[:, 5]
        h31, h32       = h[:, 6], h[:, 7]
        h_calc = np.stack((h11, h12, h13, h21, h22, h23, h31, h32), axis=1)

        # EDAX → Bruker → xo vector expected by conversions.h2F
        patshape  = (p["pat_h"], p["pat_w"])
        pc_edax   = np.asarray(p["pc_edax"], dtype=float)

        if p.get("apply_pc_correction", False):
            from pc_homography_correction import correct_homographies
            # ROI-relative ref_position for PC-correction anchor (defaults to
            # (0, 0) if neither ref_position nor a fall-back location was
            # carried into the re-vis params).
            _ref_full = p.get("ref_position", (0, 0))
            _roi      = p.get("roi_slice", None)
            if _roi is not None:
                _rr = int(_ref_full[0]) - _roi[0].start
                _rc = int(_ref_full[1]) - _roi[1].start
            else:
                _rr, _rc = int(_ref_full[0]), int(_ref_full[1])
            if not (0 <= _rr < rows and 0 <= _rc < cols):
                _rr, _rc = 0, 0
            h_calc = correct_homographies(
                h=h_calc,
                scan_shape=(rows, cols),
                step_size_um=p.get("step_size", 1.0),
                pc_ref=pc_edax,
                patshape=patshape,
                pixel_size_um=p.get("pixel_size", 1.0),
                sample_tilt_deg=p["tilt"],
                detector_tilt_deg=p["det_tilt"],
                convention=p.get("scan_strategy", "standard"),
                ref_position=(_rr, _rc),
            )

        pc_bruker = conversions.Edax_to_Bruker_PC(pc_edax)
        xo        = conversions.Bruker_to_fractional_PC(pc_bruker, patshape)

        F = conversions.h2F(h_calc, xo)
        epsilon, omega = conversions.F2strain(F, small_strain=bool(p.get("small_strain", False)))

        R = (np.eye(3) if p.get("identity_rotation", False)
             else utilities.get_sample_to_detector_rotation(p["det_tilt"], p["tilt"]))
        if p["samp_frame"]:
            for i in range(epsilon.shape[0]):
                epsilon[i] = R.T @ epsilon[i] @ R
                omega[i]   = R.T @ omega[i]   @ R

        def _2d(arr):
            return arr.reshape(rows, cols)

        # ── Euler-angle estimates from omega (same logic as PipelineWorker) ──
        nan_mask_eu = np.any(np.isnan(omega.reshape(-1, 9)), axis=1)
        omega_safe  = np.where(nan_mask_eu[:, None, None], 0.0, omega)
        rel_deg = conversions.omega_array_to_bunge(omega_safe, degrees=True)
        rel_deg[nan_mask_eu] = np.nan

        # Absolute orientation needs the per-pixel Hough Euler from the .ang
        # file at the same ROI; if unavailable, the abs panel is filled with
        # NaN so the plotter just skips it.
        abs_deg = np.full_like(rel_deg, np.nan)
        ang_path = p.get("ang_path", "")
        if ang_path and os.path.isfile(ang_path):
            ang_data = p.get("_ang_data") or utilities.read_ang(
                ang_path, patshape, segment_grain_threshold=None
            )
            full_R, full_C = ang_data.shape
            eu_2d = ang_data.eulers.reshape(full_R, full_C, 3)
            roi_slice = p.get("roi_slice", None)
            eu_roi = eu_2d[roi_slice[0], roi_slice[1], :] if roi_slice is not None else eu_2d
            eulers_flat = eu_roi.reshape(-1, 3)
            if eulers_flat.shape[0] == omega.shape[0]:
                R_omega = conversions.skew_to_rotation_matrix_array(omega_safe)
                R_hough = conversions.bunge_to_rotation_matrix(
                    eulers_flat[:, 0], eulers_flat[:, 1], eulers_flat[:, 2],
                    degrees=False,
                )
                R_abs = R_omega @ R_hough
                abs_deg = conversions.rotation_matrix_array_to_bunge(R_abs, degrees=True)
                abs_deg[nan_mask_eu] = np.nan

        result = {
            "h11": _2d(h11), "h12": _2d(h12), "h13": _2d(h13),
            "h21": _2d(h21), "h22": _2d(h22), "h23": _2d(h23),
            "h31": _2d(h31), "h32": _2d(h32),
            "e11": _2d(epsilon[:, 0, 0]), "e12": _2d(epsilon[:, 0, 1]),
            "e13": _2d(epsilon[:, 0, 2]), "e22": _2d(epsilon[:, 1, 1]),
            "e23": _2d(epsilon[:, 1, 2]), "e33": _2d(epsilon[:, 2, 2]),
            "w13": _2d(np.degrees(omega[:, 0, 2])),
            "w21": _2d(np.degrees(omega[:, 1, 0])),
            "w32": _2d(np.degrees(omega[:, 2, 1])),
            "phi1_rel": _2d(rel_deg[:, 0]),
            "Phi_rel":  _2d(rel_deg[:, 1]),
            "phi2_rel": _2d(rel_deg[:, 2]),
            "phi1_abs": _2d(abs_deg[:, 0]),
            "Phi_abs":  _2d(abs_deg[:, 1]),
            "phi2_abs": _2d(abs_deg[:, 2]),
            "rows": rows, "cols": cols,
            "F_flat": F,          # (N, 3, 3) — needed for TFBC polar decomposition
        }

        # Load per-pattern base orientations from .ang file if provided
        ang_path = p.get("ang_path", "")
        if ang_path and os.path.isfile(ang_path):
            ang_data = p.get("_ang_data") or utilities.read_ang(
                ang_path, patshape, segment_grain_threshold=None
            )
            full_r, full_c = ang_data.shape   # full scan shape from the ang file itself
            # Reshape to 2-D so we can slice the ROI out cleanly
            quats_2d  = ang_data.quats.reshape(full_r, full_c, 4)
            roi_slice = p.get("roi_slice", None)
            if roi_slice is not None:
                quats_roi = quats_2d[roi_slice[0], roi_slice[1], :]
            else:
                quats_roi = quats_2d
            result["base_quats"] = quats_roi.reshape(-1, 4)

        return result

    def _load_from_npz(self, npy_path: str, p: dict) -> dict:
        """Load pre-computed strain/rotation results saved by PipelineWorker."""
        import utilities
        d    = np.load(npy_path, allow_pickle=True).item()
        rows = int(d["rows"])
        cols = int(d["cols"])

        result = {
            "h11": d["h11"], "h12": d["h12"], "h13": d["h13"],
            "h21": d["h21"], "h22": d["h22"], "h23": d["h23"],
            "h31": d["h31"], "h32": d["h32"],
            "e11": d["e11"], "e12": d["e12"], "e13": d["e13"],
            "e22": d["e22"], "e23": d["e23"], "e33": d["e33"],
            "w13": d["w13"], "w21": d["w21"], "w32": d["w32"],
            "rows": rows, "cols": cols,
            "F_flat": d["F"],
        }
        # Optional: FMT-FCC initial-guess homography (saved by PipelineWorker
        # when init_type != NONE).  Older runs / NONE init won't have it.
        if "h_guess" in d:
            result["h_guess"] = d["h_guess"]

        # Optional: Euler-angle estimates (saved by recent PipelineWorker runs).
        for k in ("phi1_rel", "Phi_rel", "phi2_rel",
                  "phi1_abs", "Phi_abs", "phi2_abs"):
            if k in d:
                result[k] = d[k]

        # Load base orientations from .ang if provided (needed for TFBC)
        ang_path = p.get("ang_path", "")
        if ang_path and os.path.isfile(ang_path):
            patshape       = (p.get("pat_h", 512), p.get("pat_w", 512))
            ang_data       = p.get("_ang_data") or utilities.read_ang(
                ang_path, patshape, segment_grain_threshold=None
            )
            full_r, full_c = ang_data.shape
            quats_2d       = ang_data.quats.reshape(full_r, full_c, 4)
            roi_slice      = p.get("roi_slice", None)
            if roi_slice is not None:
                quats_roi = quats_2d[roi_slice[0], roi_slice[1], :]
            else:
                quats_roi = quats_2d[:rows, :cols, :]
            result["base_quats"] = quats_roi.reshape(-1, 4)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweep worker
# ─────────────────────────────────────────────────────────────────────────────

class SweepWorker(QThread):
    """
    Processes a single pattern with every combination in *param_list*.
    Emits progress_signal(idx, total, processed_img, params) after each one,
    then done_signal when finished.  Call abort() to stop early.
    """
    progress_signal = pyqtSignal(int, int, object, dict)
    done_signal     = pyqtSignal()
    error_signal    = pyqtSignal(str)

    def __init__(self, up2_path: str, pat_idx: int, param_list: list):
        super().__init__()
        self.up2_path   = up2_path
        self.pat_idx    = pat_idx
        self.param_list = param_list
        self._abort     = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            import Data
            total = len(self.param_list)
            for i, params in enumerate(self.param_list):
                if self._abort:
                    break
                pat_obj   = Data.UP2(self.up2_path)
                mask_type = params["mask_type"]
                if mask_type == "None":
                    mask_type = None
                pat_obj.set_processing(
                    low_pass_sigma         = params["low_pass_sigma"],
                    high_pass_sigma        = params["high_pass_sigma"],
                    truncate_std_scale     = 3.0,
                    mask_type              = mask_type,
                    center_cross_half_width= 6,
                    use_clahe              = params.get("use_clahe", False),
                    clahe_kernel           = (params["clahe_kernel"], params["clahe_kernel"]),
                    clahe_clip             = params["clahe_clip"],
                    clahe_nbins            = 256,
                    flip_x                 = params["flip_x"],
                    gamma                  = params.get("gamma", 0.66),
                )
                processed = pat_obj.read_pattern(self.pat_idx, process=True)
                self.progress_signal.emit(i, total, processed, params)
            if not self._abort:
                self.done_signal.emit()
        except Exception:
            self.error_signal.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Pattern preview worker
# ─────────────────────────────────────────────────────────────────────────────

class SimRefWorker(QThread):
    """Generate a simulated reference pattern via SimPatGen / HREBSD.py."""
    done_signal  = pyqtSignal(object)   # numpy float32 array, normalised 0-1
    error_signal = pyqtSignal(str)

    def __init__(self, master_path: str, euler_deg: tuple, pc: tuple,
                 det_shape: tuple, det_tilt_deg: float, sample_tilt_deg: float = 70.0):
        super().__init__()
        self.master_path     = master_path
        self.euler_deg       = euler_deg       # (phi1, Phi, phi2) in degrees
        self.pc              = pc              # (x*, y*, z*) EDAX convention
        self.det_shape       = det_shape       # (H, W)
        self.det_tilt_deg    = det_tilt_deg    # detector tilt from Step 2
        self.sample_tilt_deg = sample_tilt_deg # sample tilt from Step 2

    def run(self):
        try:
            from PatternSimulation.SimPatGen import patternSimulation
            sim = patternSimulation()
            # ── Geometry from Step 2 ──────────────────────────────────────────
            sim.detector_height  = self.det_shape[0]
            sim.detector_width   = self.det_shape[1]
            sim.det_shape        = self.det_shape
            sim.detector_tilt_deg  = self.det_tilt_deg    # e.g. 10 °
            sim.sample_tilt_deg    = self.sample_tilt_deg  # e.g. 70 °
            # HREBSD.py uses alpha = π/2 + (detector_tilt - sample_tilt)·π/180
            print(f"Detector shape:   {sim.det_shape}")
            print(f"Sample tilt:      {sim.sample_tilt_deg} °")
            print(f"Detector tilt:    {sim.detector_tilt_deg} °")
            print(f"alpha (deg):      {90.0 + (sim.detector_tilt_deg - sim.sample_tilt_deg):.1f} °")
            # ─────────────────────────────────────────────────────────────────
            sim.mastersetup(self.master_path)
            import conversions
            euler_rad = np.deg2rad(self.euler_deg)

            # ── PC convention passed to EandPCSet ─────────────────────────────
            # self.pc is in EDAX/TSL convention (x*, y*, z*).
            # EandPCSet calls bruker_geometry_to_SE3 internally, so Bruker PC
            # must be supplied.  Change the line below if your PC is already
            # in Bruker convention (comment out the conversion and pass self.pc).
            pc_edax   = self.pc                              # received as EDAX
            pc_bruker = conversions.Edax_to_Bruker_PC(pc_edax)   # → Bruker (pcy = 1 - y*)
            print(f"PC (EDAX):   {pc_edax}")
            print(f"PC (Bruker): {pc_bruker}")
            # ─────────────────────────────────────────────────────────────────

            sim.EandPCSet(euler_rad, list(pc_bruker), verbose=False)
            pat = sim.GenPattern()
            pat_np = pat.detach().cpu().numpy().astype(np.float32)
            # project_hrebsd returns (B, H*W) — reshape to (H, W)
            H, W = self.det_shape
            pat_np = pat_np.reshape(H, W)
            # PC sign convention fixed in HREBSD.detector_coords_to_ksphere_via_pc
            # (pcx_ems uses pcx - 0.5, matching EMsoft) — no fliplr needed.
            lo, hi = pat_np.min(), pat_np.max()
            pat_np = (pat_np - lo) / (hi - lo + 1e-9)
            self.done_signal.emit(pat_np)
        except Exception:
            self.error_signal.emit(traceback.format_exc())


class PcEulerRefineWorker(QThread):
    """Refine pattern-centre and Euler angles by minimising ZNSSD (Nelder-Mead
    in so(3) ⊕ ℝ³)."""
    log_signal   = pyqtSignal(str)
    done_signal  = pyqtSignal(object, object)  # euler_opt (rad ndarray), pc_opt (tuple)
    error_signal = pyqtSignal(str)

    def __init__(self, up2_path: str, pat_idx: int, processing_params: dict,
                 master_pattern_path: str, euler_angles_init: np.ndarray,
                 pc_init: tuple, sample_tilt_deg: float, detector_tilt_deg: float,
                 max_iter: int = 300, euler_step_deg: float = 0.5,
                 pc_step: float = 0.005,
                 n_restarts: int = 4,
                 restart_rotvec_std_deg: float = 3.0,
                 restart_pc_std: float = 0.01,
                 restart_seed: int = 0,
                 symmetry_restarts: bool = False,
                 laue_group_id: int = 11,
                 save_dir: str = "debug",
                 sim_high_pass_sigma: float = None,
                 sim_low_pass_sigma: float = None,
                 sim_gamma: float = None):
        super().__init__()
        self.up2_path                = up2_path
        self.pat_idx                 = pat_idx
        self.processing_params       = processing_params
        self.master_pattern_path     = master_pattern_path
        self.euler_angles_init       = euler_angles_init
        self.pc_init                 = pc_init
        self.sample_tilt_deg         = sample_tilt_deg
        self.detector_tilt_deg       = detector_tilt_deg
        self.max_iter                = max_iter
        self.euler_step_deg          = euler_step_deg
        self.pc_step                 = pc_step
        self.n_restarts              = n_restarts
        self.restart_rotvec_std_deg  = restart_rotvec_std_deg
        self.restart_pc_std          = restart_pc_std
        self.restart_seed            = restart_seed
        self.symmetry_restarts       = symmetry_restarts
        self.laue_group_id           = laue_group_id
        self.save_dir                = save_dir
        self.sim_high_pass_sigma     = sim_high_pass_sigma
        self.sim_low_pass_sigma      = sim_low_pass_sigma
        self.sim_gamma               = sim_gamma

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = _StdoutCapture(self.log_signal)
        try:
            import Data
            from optimize_reference import optimize_pc_and_euler

            p = self.processing_params
            mask_type = p["mask_type"] if p["mask_type"] != "None" else None
            pat_obj = Data.UP2(self.up2_path)
            pat_obj.set_processing(
                low_pass_sigma          = p["low_pass_sigma"],
                high_pass_sigma         = p["high_pass_sigma"],
                truncate_std_scale      = 3.0,
                mask_type               = mask_type,
                center_cross_half_width = 6,
                flip_x                  = p["flip_x"],
                gamma                   = p.get("gamma", 0.8),
            )
            euler_opt, pc_opt = optimize_pc_and_euler(
                pat_obj                = pat_obj,
                x0                     = self.pat_idx,
                master_pattern_path    = self.master_pattern_path,
                euler_angles_init      = self.euler_angles_init,
                pc_init                = self.pc_init,
                sample_tilt_deg        = self.sample_tilt_deg,
                detector_tilt_deg      = self.detector_tilt_deg,
                max_iter               = self.max_iter,
                euler_step_deg         = self.euler_step_deg,
                pc_step                = self.pc_step,
                n_restarts             = self.n_restarts,
                restart_rotvec_std_deg = self.restart_rotvec_std_deg,
                restart_pc_std         = self.restart_pc_std,
                restart_seed           = self.restart_seed,
                symmetry_restarts      = self.symmetry_restarts,
                laue_group_id          = self.laue_group_id,
                save_dir               = self.save_dir,
                sim_high_pass_sigma    = self.sim_high_pass_sigma,
                sim_low_pass_sigma     = self.sim_low_pass_sigma,
                sim_gamma              = self.sim_gamma,
            )
            self.done_signal.emit(euler_opt, pc_opt)
        except Exception:
            self.error_signal.emit(traceback.format_exc())
        finally:
            sys.stdout = old_stdout


class PatternPreviewWorker(QThread):
    """Load one pattern from a UP2 file, return raw and processed versions."""
    done_signal  = pyqtSignal(object, object)  # raw ndarray, processed ndarray
    error_signal = pyqtSignal(str)

    def __init__(self, up2_path: str, pat_idx: int, params: dict):
        super().__init__()
        self.up2_path = up2_path
        self.pat_idx  = pat_idx
        self.params   = params

    def run(self):
        try:
            import Data
            p = self.params
            pat_obj = Data.UP2(self.up2_path)
            mask_type = p["mask_type"]
            if mask_type == "None":
                mask_type = None
            pat_obj.set_processing(
                low_pass_sigma         = p["low_pass_sigma"],
                high_pass_sigma        = p["high_pass_sigma"],
                truncate_std_scale     = 3.0,
                mask_type              = mask_type,
                center_cross_half_width= 6,
                flip_x                 = p["flip_x"],
                gamma                  = p.get("gamma", 0.8),
            )
            raw       = pat_obj.read_pattern(self.pat_idx, process=False)
            processed = pat_obj.read_pattern(self.pat_idx, process=True)
            self.done_signal.emit(raw, processed)
        except Exception:
            self.error_signal.emit(traceback.format_exc())

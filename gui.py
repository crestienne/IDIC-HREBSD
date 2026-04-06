"""
gui.py — Panel-based GUI for DIC-HREBSD pipeline

Launch with:
    panel serve gui.py --show

Or from Python:
    python gui.py

This is one way to run the pipeline. All runner scripts and notebooks
continue to work exactly as before — this file is completely standalone.
"""

import os
import sys
import threading
import datetime
import io

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panel as pn
import param

pn.extension(sizing_mode="stretch_width")

# ── helpers ──────────────────────────────────────────────────────────────────

def _fig_to_pane(fig, max_width=700):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return pn.pane.PNG(buf, max_width=max_width)


def rotation_matrix_from_tilt(tilt_deg: float) -> np.ndarray:
    theta_x = np.deg2rad(90.0 - tilt_deg)
    Rx = np.array([
        [1.0,  0.0,              0.0             ],
        [0.0,  np.cos(theta_x), -np.sin(theta_x) ],
        [0.0,  np.sin(theta_x),  np.cos(theta_x) ],
    ])
    Rz_180 = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    return Rx @ Rz_180


# ── GUI class ─────────────────────────────────────────────────────────────────

class HREBSD_GUI(param.Parameterized):

    # ── File inputs ──────────────────────────────────────────────────────────
    up2_path   = param.String(default="", label="UP2 file path")
    ang_path   = param.String(default="", label="ANG file path")
    output_dir = param.String(
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        label="Results folder",
    )
    component  = param.String(default="MyRun", label="Run name")

    # ── Scan geometry ────────────────────────────────────────────────────────
    ref_row = param.Integer(default=0,    bounds=(0, 9999), label="Reference row (y)")
    ref_col = param.Integer(default=0,    bounds=(0, 9999), label="Reference col (x)")
    tilt    = param.Number( default=70.0, bounds=(0, 90),   label="Sample tilt (°)")

    # ── Region of interest ───────────────────────────────────────────────────
    use_roi   = param.Boolean(default=False, label="Use region of interest (ROI)")
    roi_y0    = param.Integer(default=0,   bounds=(0, 9999), label="ROI y start")
    roi_y1    = param.Integer(default=100, bounds=(1, 9999), label="ROI y stop")
    roi_x0    = param.Integer(default=0,   bounds=(0, 9999), label="ROI x start")
    roi_x1    = param.Integer(default=100, bounds=(1, 9999), label="ROI x stop")

    # ── Pattern processing ───────────────────────────────────────────────────
    low_pass_sigma     = param.Number( default=0.0,  bounds=(0, 100), label="Low-pass sigma")
    high_pass_sigma    = param.Number( default=15.0, bounds=(0, 200), label="High-pass sigma")
    truncate_std_scale = param.Number( default=3.0,  bounds=(0, 10),  label="Truncate std scale")
    mask_type          = param.ObjectSelector(
        default="center_cross", objects=["circular", "center_cross", "none"], label="Mask type"
    )
    center_cross_hw    = param.Integer(default=6,    bounds=(1, 50),  label="Cross half-width (px)")
    clahe_kernel       = param.Integer(default=5,    bounds=(1, 32),  label="CLAHE kernel size")
    clahe_clip         = param.Number( default=0.01, bounds=(0, 1),   label="CLAHE clip")
    flip_x             = param.Boolean(default=False, label="Flip patterns (flip_x)")

    # ── Optimization ─────────────────────────────────────────────────────────
    init_type     = param.ObjectSelector(
        default="partial", objects=["none", "partial", "full"], label="Init type"
    )
    crop_fraction = param.Number( default=0.9,  bounds=(0.1, 0.99), label="Crop fraction")
    max_iter      = param.Integer(default=100,  bounds=(1, 1000),   label="Max iterations")
    n_jobs        = param.Integer(default=8,    bounds=(-1, 64),    label="n_jobs (parallel)")

    # ── Simulated reference ──────────────────────────────────────────────────
    use_sim_ref         = param.Boolean(default=False, label="Use simulated reference")
    master_pattern_path = param.String(default="", label="Master pattern path (.h5)")

    # ── Pattern center override ──────────────────────────────────────────────
    use_custom_pc = param.Boolean(default=False, label="Override pattern center")
    pc_xstar      = param.Number(default=0.5,  bounds=(0, 1),    label="xstar")
    pc_ystar      = param.Number(default=0.5,  bounds=(0, 1),    label="ystar")
    pc_zstar      = param.Number(default=0.65, bounds=(0.01, 2), label="zstar")

    def __init__(self, **params):
        super().__init__(**params)
        self._log_buffer = []
        self._results    = None
        self._running    = False
        self._N_total    = 0

        # ── Buttons ─────────────────────────────────────────────────────────
        self._run_btn   = pn.widgets.Button(name="▶  Run Pipeline", button_type="success", width=160)
        self._clear_btn = pn.widgets.Button(name="Clear log",       button_type="light",   width=110)
        self._run_btn.on_click(self._on_run)
        self._clear_btn.on_click(lambda e: self._clear_log())

        # ── Progress bar ─────────────────────────────────────────────────────
        self._progress     = pn.widgets.Progress(
            name="", value=0, max=100, bar_color="success",
            width=440, visible=False,
        )
        self._progress_txt = pn.pane.Markdown("", width=440)

        # ── Status badge ────────────────────────────────────────────────────
        self._status = pn.pane.Markdown("🟡 **Ready**", width=200)

        # ── Log ──────────────────────────────────────────────────────────────
        self._log_pane = pn.widgets.TextAreaInput(
            value="", height=180, disabled=True, name="Log"
        )

        # ── Result tabs (populated after run) ───────────────────────────────
        self._result_tabs = pn.Tabs(visible=False)

        # ── Color-reactive toggles ───────────────────────────────────────────
        self._roi_toggle = pn.widgets.Toggle(
            name="Use region of interest (ROI)", value=False, button_type="default", width=260
        )
        self._roi_toggle.param.watch(
            lambda e: (setattr(self, "use_roi", e.new),
                       setattr(self._roi_toggle, "button_type", "success" if e.new else "default")),
            "value",
        )

        self._pc_toggle = pn.widgets.Toggle(
            name="Override pattern center", value=False, button_type="default", width=260
        )
        self._pc_toggle.param.watch(
            lambda e: (setattr(self, "use_custom_pc", e.new),
                       setattr(self._pc_toggle, "button_type", "warning" if e.new else "default")),
            "value",
        )

        # ── File picker buttons (open native OS dialog) ──────────────────────
        self._up2_btn = pn.widgets.Button(name="Browse…", button_type="primary", width=100)
        self._ang_btn = pn.widgets.Button(name="Browse…", button_type="primary", width=100)
        self._up2_btn.on_click(lambda _: self._pick_file("up2"))
        self._ang_btn.on_click(lambda _: self._pick_file("ang"))

        # ── UP2 preview ──────────────────────────────────────────────────────
        self._preview_pane = pn.Column()
        self.param.watch(self._on_up2_path_change, "up2_path")

    # ── logging & status ─────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._log_buffer.append(msg)
        self._log_pane.value = "\n".join(self._log_buffer[-200:])

    def _clear_log(self):
        self._log_buffer.clear()
        self._log_pane.value = ""

    def _set_status(self, state: str):
        icons = {"ready": "🟡 **Ready**", "running": "🔵 **Running…**",
                 "done": "🟢 **Done**", "error": "🔴 **Error**"}
        self._status.object = icons.get(state, state)

    def _set_progress(self, n_done, n_total):
        pct = int(100 * n_done / max(n_total, 1))
        self._progress.value       = pct
        self._progress_txt.object  = f"Patterns: **{n_done} / {n_total}** ({pct}%)"

    # ── Native file picker ────────────────────────────────────────────────────

    def _pick_file(self, kind: str):
        import tkinter as tk
        from tkinter import filedialog
        filetypes = {
            "up2": [("UP2 files", "*.up2"), ("All files", "*.*")],
            "ang": [("ANG files", "*.ang"), ("All files", "*.*")],
        }
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=f"Select {kind.upper()} file",
            filetypes=filetypes[kind],
            initialdir=os.path.expanduser("~"),
        )
        root.destroy()
        if path:
            if kind == "up2":
                self.up2_path = path
            else:
                self.ang_path = path

    # ── UP2 preview ───────────────────────────────────────────────────────────

    def _on_up2_path_change(self, event):
        path = event.new.strip()
        self._preview_pane.clear()
        if not path or not os.path.exists(path):
            return
        try:
            import Data
            pat_obj = Data.UP2(path)
            pat = pat_obj.read_pattern(0, process=False)
            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            ax.imshow(pat, cmap="gray")
            ax.set_title(f"shape {pat.shape}  |  N={pat_obj.nPatterns}", fontsize=8)
            ax.axis("off")
            plt.tight_layout()
            self._preview_pane.append(
                pn.pane.Markdown("#### UP2 preview — pattern 0")
            )
            self._preview_pane.append(_fig_to_pane(fig, max_width=300))
        except Exception as exc:
            self._preview_pane.append(pn.pane.Markdown(f"⚠ Could not load preview: {exc}"))

    # ── run ──────────────────────────────────────────────────────────────────

    def _on_run(self, event):
        if self._running:
            self._log("⚠  Already running — please wait.")
            return
        self._running = True
        self._run_btn.disabled = True
        self._run_btn.name = "Running…"
        self._set_status("running")
        self._progress.visible = True
        self._progress.value   = 0
        self._progress_txt.object = ""
        self._result_tabs.visible = False
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        try:
            self._do_run()
            self._set_status("done")
        except Exception as exc:
            import traceback
            self._log(f"\n❌ ERROR: {exc}\n{traceback.format_exc()}")
            self._set_status("error")
        finally:
            self._running = False
            self._run_btn.disabled = False
            self._run_btn.name = "▶  Run Pipeline"
            self._progress.visible = False

    def _do_run(self):
        import Data
        import utilities
        import get_homography_cpu as core
        import conversions

        self._log("=" * 60)
        self._log(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ── Validate paths ───────────────────────────────────────────────────
        up2_path = self.up2_path.strip()
        ang_path = self.ang_path.strip()
        for label, path in [("UP2", up2_path), ("ANG", ang_path)]:
            if not path or not os.path.exists(path):
                self._log(f"❌ {label} file not found: {path!r}")
                self._set_status("error")
                return

        # ── Output folder ────────────────────────────────────────────────────
        date_str   = datetime.datetime.now().strftime("%b%d%Y")
        base_folder = f"{self.component}_{date_str}_npyfiles"
        foldername  = os.path.join(self.output_dir.strip(), base_folder) + os.sep
        os.makedirs(foldername, exist_ok=True)
        os.makedirs("debug", exist_ok=True)
        self._log(f"Output folder: {foldername}")

        # ── Load UP2 ─────────────────────────────────────────────────────────
        self._log("Loading UP2 file…")
        pat_obj = Data.UP2(up2_path)
        mt = None if self.mask_type == "none" else self.mask_type
        pat_obj.set_processing(
            low_pass_sigma=self.low_pass_sigma,
            high_pass_sigma=self.high_pass_sigma,
            truncate_std_scale=self.truncate_std_scale,
            mask_type=mt,
            center_cross_half_width=self.center_cross_hw,
            clahe_kernel=(self.clahe_kernel, self.clahe_kernel),
            clahe_clip=self.clahe_clip,
            flip_x=self.flip_x,
        )
        self._log(f"Pattern shape: {pat_obj.patshape}  |  N patterns: {pat_obj.nPatterns}")

        # ── Read ANG ─────────────────────────────────────────────────────────
        self._log("Reading ANG file…")
        ang_data = utilities.read_ang(ang_path, pat_obj.patshape, segment_grain_threshold=None)
        x0 = np.ravel_multi_index((self.ref_row, self.ref_col), ang_data.shape)
        self._log(f"Reference pattern flat index: {x0}")
        euler_angles_ref = ang_data.eulers[np.unravel_index(x0, ang_data.shape)]
        if self.use_custom_pc:
            pc_ref = (self.pc_xstar, self.pc_ystar, self.pc_zstar)
            self._log(f"Using custom PC: xstar={self.pc_xstar}, ystar={self.pc_ystar}, zstar={self.pc_zstar}")
        else:
            pc_ref = ang_data.pc
            self._log(f"Using PC from ANG: {pc_ref}")

        # ── Simulated reference ──────────────────────────────────────────────
        mp_path = self.master_pattern_path.strip() if self.use_sim_ref else None
        if self.use_sim_ref:
            if not mp_path or not os.path.exists(mp_path):
                self._log(f"❌ Master pattern not found: {mp_path!r}")
                self._set_status("error")
                return
            self._log("Generating simulated reference pattern…")
            real_pat = pat_obj.read_pattern(x0, process=True)
            sim_pat  = core.simulate_reference_pattern(
                master_pattern_path=mp_path,
                euler_angles=euler_angles_ref,
                PC=pc_ref,
                patshape=pat_obj.patshape,
                tilt_deg=self.tilt,
            )
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            axes[0].imshow(real_pat, cmap="gray"); axes[0].set_title("Real ref");      axes[0].axis("off")
            axes[1].imshow(sim_pat,  cmap="gray"); axes[1].set_title("Simulated ref"); axes[1].axis("off")
            plt.tight_layout()
            self._log("Simulated reference generated.")
            self._queue_ref_tab(_fig_to_pane(fig))

        # ── Progress callback ────────────────────────────────────────────────
        N_total = pat_obj.nPatterns

        def _progress_cb(n_done, n_total):
            self._set_progress(n_done, n_total)
            if n_done % max(1, n_total // 20) == 0 or n_done == n_total:
                self._log(f"  Patterns done: {n_done} / {n_total}")

        # ── ROI ──────────────────────────────────────────────────────────────
        if self.use_roi:
            roi_slice = [slice(self.roi_y0, self.roi_y1), slice(self.roi_x0, self.roi_x1)]
            self._log(f"ROI: rows {self.roi_y0}:{self.roi_y1}, cols {self.roi_x0}:{self.roi_x1}")
        else:
            roi_slice = None

        # ── Optimize ─────────────────────────────────────────────────────────
        self._log("Starting optimization…")
        optimize_params = dict(
            init_type=self.init_type,
            crop_fraction=self.crop_fraction,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            verbose=True,
            roi_slice=roi_slice,
            mask=pat_obj.get_mask(),
            use_simulated_reference=self.use_sim_ref,
            master_pattern_path=mp_path,
            euler_angles_ref=euler_angles_ref,
            pc_ref=pc_ref,
            tilt_deg=self.tilt,
            progress_callback=_progress_cb,
        )
        result = core.optimize(pat_obj, x0, **optimize_params)

        if self.init_type == "none":
            h, iterations, residuals, dp_norms = result
            h_guess = None
        else:
            h, h_guess, iterations, residuals, dp_norms = result

        self._log(f"Optimization complete. Patterns processed: {len(h)}")

        # ── Save ─────────────────────────────────────────────────────────────
        np.save(f"{foldername}{self.component}_homographies_{date_str}.npy", h)
        np.save(f"{foldername}{self.component}_iterations_{date_str}.npy",   iterations)
        np.save(f"{foldername}{self.component}_residuals_{date_str}.npy",    residuals)
        np.save(f"{foldername}{self.component}_dp_norms_{date_str}.npy",     dp_norms)
        if h_guess is not None:
            np.save(f"{foldername}{self.component}_h_guess_{date_str}.npy",  h_guess)
        self._log(f"Results saved to {foldername}")

        # ── Strain ───────────────────────────────────────────────────────────
        self._log("Computing strain…")
        N    = len(h)
        h_2d = h.reshape(N, 8)
        h_calc = np.stack([h_2d[:, i] for i in range(8)], axis=1)

        xstar, ystar, zstar = pc_ref
        patH   = pat_obj.patshape[0]
        PC_vec = np.array([(xstar - 0.5)*patH, (ystar - 0.5)*patH, patH*zstar])
        F = conversions.h2F(h_calc, PC_vec)
        epsilon, omega = conversions.F2strain(F)

        R = rotation_matrix_from_tilt(self.tilt)
        for i in range(N):
            epsilon[i] = R.T @ epsilon[i] @ R
            omega[i]   = R.T @ omega[i]   @ R

        Rows, Cols = ang_data.shape[0], ang_data.shape[1]
        if Rows * Cols == N:
            shape2d = (Rows, Cols)
        else:
            side = int(np.sqrt(N))
            shape2d = (side, side) if side * side == N else (N, 1)
            self._log(f"⚠  Could not infer scan grid — using shape {shape2d}")

        def comp(arr, i, j):
            return arr[:, i, j].reshape(shape2d)

        e11 = comp(epsilon,0,0); e12 = comp(epsilon,0,1); e13 = comp(epsilon,0,2)
        e22 = comp(epsilon,1,1); e23 = comp(epsilon,1,2); e33 = comp(epsilon,2,2)
        w13 = np.degrees(comp(omega,0,2))
        w21 = np.degrees(comp(omega,1,0))
        w32 = np.degrees(comp(omega,2,1))

        von_mises = np.sqrt(
            (2/3) * (e11**2 + e22**2 + e33**2 + 2*e12**2 + 2*e13**2 + 2*e23**2)
        )

        for name, arr in [("e11",e11),("e12",e12),("e13",e13),("e22",e22),
                           ("e23",e23),("e33",e33),("w13",w13),("w21",w21),
                           ("w32",w32),("von_mises",von_mises)]:
            np.save(f"{foldername}{self.component}_{name}_{date_str}.npy", arr)
        self._log("Strain components saved.")

        # ── Build result tabs ────────────────────────────────────────────────
        self._log("Generating plots…")
        vmin, vmax = -5e-2, 5e-2

        # Tab 1: strain grid
        fig1, axes = plt.subplots(3, 3, figsize=(13, 8))
        pairs = [
            (e11, r"$\varepsilon_{11}$"), (e12, r"$\varepsilon_{12}$"), (e13, r"$\varepsilon_{13}$"),
            (w21, r"$\omega_{21}$ (°)"),  (e22, r"$\varepsilon_{22}$"), (e23, r"$\varepsilon_{23}$"),
            (w13, r"$\omega_{13}$ (°)"),  (w32, r"$\omega_{32}$ (°)"), (e33, r"$\varepsilon_{33}$"),
        ]
        for ax, (data, title) in zip(axes.ravel(), pairs):
            im = ax.imshow(data, cmap="coolwarm", vmin=vmin, vmax=vmax)
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        fig1.suptitle(f"{self.component} — strain components (sample frame)", fontsize=13)
        plt.tight_layout()
        fig1.savefig(f"{foldername}strain_grid_{date_str}.png", dpi=150, bbox_inches="tight")

        # Tab 2: Von Mises
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(von_mises, cmap="viridis", vmin=0, vmax=vmax)
        fig2.colorbar(im2, ax=ax2, label="Von Mises equivalent strain")
        ax2.set_title(f"{self.component} — Von Mises strain", fontweight="bold")
        ax2.axis("off")
        plt.tight_layout()
        fig2.savefig(f"{foldername}von_mises_{date_str}.png", dpi=150, bbox_inches="tight")

        # Tab 3: diagnostics (iterations + residuals)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 3.5))
        im3a = ax3a.imshow(iterations.reshape(shape2d), cmap="plasma")
        fig3.colorbar(im3a, ax=ax3a, fraction=0.046, pad=0.04, label="Iterations")
        ax3a.set_title("Iterations to converge"); ax3a.axis("off")
        im3b = ax3b.imshow(residuals.reshape(shape2d), cmap="inferno")
        fig3.colorbar(im3b, ax=ax3b, fraction=0.046, pad=0.04, label="Residual")
        ax3b.set_title("Final residuals"); ax3b.axis("off")
        plt.tight_layout()

        # Populate tabs on the main thread
        self._result_tabs.clear()
        self._result_tabs.extend([
            ("Strain components", pn.Column(_fig_to_pane(fig1, max_width=900))),
            ("Von Mises",         pn.Column(_fig_to_pane(fig2, max_width=500))),
            ("Diagnostics",       pn.Column(_fig_to_pane(fig3, max_width=800))),
        ])
        self._result_tabs.visible = True

        self._log("✅ Done!")
        self._results = dict(epsilon=epsilon, omega=omega, von_mises=von_mises)

    def _queue_ref_tab(self, pane):
        """Add a simulated-reference comparison tab (called before main result tabs)."""
        self._result_tabs.clear()
        self._result_tabs.append(("Reference patterns", pn.Column(pane)))
        self._result_tabs.visible = True

    # ── layout ────────────────────────────────────────────────────────────────

    def view(self):
        # ── sidebar tabs ────────────────────────────────────────────────────
        tab_files = pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("**UP2 file**"),
            pn.Row(self._up2_btn, pn.widgets.TextInput.from_param(self.param.up2_path, placeholder="Click Browse… or paste path")),
            pn.layout.Divider(),
            pn.pane.Markdown("**ANG file**"),
            pn.Row(self._ang_btn, pn.widgets.TextInput.from_param(self.param.ang_path, placeholder="Click Browse… or paste path")),
            pn.layout.Divider(),
            pn.widgets.TextInput.from_param(self.param.output_dir),
            pn.widgets.TextInput.from_param(self.param.component),
        )

        tab_geometry = pn.Column(
            pn.layout.Divider(),
            pn.widgets.IntInput.from_param(self.param.ref_row),
            pn.widgets.IntInput.from_param(self.param.ref_col),
            pn.widgets.FloatInput.from_param(self.param.tilt),
            pn.layout.Divider(),
            self._roi_toggle,
            pn.pane.Markdown("*Row/col indices into the scan grid (not pixel coordinates)*",
                             styles={"font-size": "11px", "color": "gray"}),
            pn.Row(
                pn.widgets.IntInput.from_param(self.param.roi_y0),
                pn.widgets.IntInput.from_param(self.param.roi_y1),
            ),
            pn.Row(
                pn.widgets.IntInput.from_param(self.param.roi_x0),
                pn.widgets.IntInput.from_param(self.param.roi_x1),
            ),
            pn.layout.Divider(),
            self._pc_toggle,
            pn.pane.Markdown("*Overrides the PC read from the ANG file*",
                             styles={"font-size": "11px", "color": "gray"}),
            pn.Row(
                pn.widgets.FloatInput.from_param(self.param.pc_xstar),
                pn.widgets.FloatInput.from_param(self.param.pc_ystar),
                pn.widgets.FloatInput.from_param(self.param.pc_zstar),
            ),
        )

        tab_processing = pn.Column(
            pn.layout.Divider(),
            pn.widgets.FloatInput.from_param(self.param.low_pass_sigma),
            pn.widgets.FloatInput.from_param(self.param.high_pass_sigma),
            pn.widgets.FloatInput.from_param(self.param.truncate_std_scale),
            pn.layout.Divider(),
            pn.widgets.Select.from_param(self.param.mask_type),
            pn.widgets.IntInput.from_param(self.param.center_cross_hw),
            pn.layout.Divider(),
            pn.widgets.IntInput.from_param(self.param.clahe_kernel),
            pn.widgets.FloatInput.from_param(self.param.clahe_clip),
            pn.widgets.Toggle.from_param(self.param.flip_x),
        )

        tab_optimization = pn.Column(
            pn.layout.Divider(),
            pn.widgets.Select.from_param(self.param.init_type),
            pn.widgets.FloatInput.from_param(self.param.crop_fraction),
            pn.widgets.IntInput.from_param(self.param.max_iter),
            pn.widgets.IntInput.from_param(self.param.n_jobs),
            pn.layout.Divider(),
            pn.widgets.Toggle.from_param(self.param.use_sim_ref),
            pn.widgets.TextInput.from_param(
                self.param.master_pattern_path, placeholder="/path/to/master.h5"
            ),
            pn.layout.Divider(),
            pn.Row(self._run_btn, self._clear_btn),
        )

        sidebar_tabs = pn.Tabs(
            ("Files",        tab_files),
            ("Geometry",     tab_geometry),
            ("Processing",   tab_processing),
            ("Optimization", tab_optimization),
            tabs_location="above",
        )

        # ── main area ───────────────────────────────────────────────────────
        main = pn.Column(
            pn.Row(self._status, self._progress, self._progress_txt),
            self._preview_pane,
            pn.layout.Divider(),
            self._log_pane,
            pn.layout.Divider(),
            self._result_tabs,
            sizing_mode="stretch_width",
        )

        # ── FastListTemplate ────────────────────────────────────────────────
        template = pn.template.FastListTemplate(
            title="DIC-HREBSD Pipeline",
            sidebar=[sidebar_tabs],
            main=[main],
            accent_base_color="#1f77b4",
            header_background="#1f77b4",
            sidebar_width=340,
        )
        return template


# ── entry point ───────────────────────────────────────────────────────────────

gui = HREBSD_GUI()
gui.view().servable()

if __name__ == "__main__":
    gui.view().show(title="DIC-HREBSD Pipeline", port=5006)

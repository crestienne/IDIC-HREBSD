"""
gui_qt.py  —  PyQt6 wizard GUI for DIC-HREBSD

Requirements:
    pip install PyQt6 matplotlib

Run:
    python gui_qt.py

The .py runner scripts continue to work exactly as before — this file
is completely standalone and does not modify any existing code.
"""

import sys
import os
import io
import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QWizard, QWizardPage,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QTextEdit, QProgressBar, QWidget, QScrollArea,
    QSizePolicy, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_browse_row(parent_page, edit: QLineEdit, filt: str, title: str) -> QWidget:
    """Return a widget containing [QLineEdit] [Browse…]."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(90)
    btn.clicked.connect(lambda: _pick_file(parent_page, edit, filt, title))
    h.addWidget(edit)
    h.addWidget(btn)
    return w


def _make_browse_dir(parent_page, edit: QLineEdit) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(90)
    btn.clicked.connect(lambda: _pick_dir(parent_page, edit))
    h.addWidget(edit)
    h.addWidget(btn)
    return w


def _pick_file(parent, edit: QLineEdit, filt: str, title: str):
    path, _ = QFileDialog.getOpenFileName(
        parent, title, os.path.expanduser("~"), filt
    )
    if path:
        edit.setText(path)


def _pick_dir(parent, edit: QLineEdit):
    path = QFileDialog.getExistingDirectory(
        parent, "Select folder", os.path.expanduser("~")
    )
    if path:
        edit.setText(path)


def _note(text: str) -> QLabel:
    """Small grey helper-text label."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: gray; font-size: 11px;")
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
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
        import time
        import Data
        import utilities
        import get_homography_cpu as core

        p = self.params

        self.log_signal.emit("Loading UP2 file…")
        pat_obj = Data.UP2(p["up2"])
        pat_obj.set_processing(
            low_pass_sigma=p["low_pass_sigma"],
            high_pass_sigma=p["high_pass_sigma"],
            truncate_std_scale=3.0,
            mask_type=p["mask_type"],
            center_cross_half_width=6,
            clahe_kernel=(p["clahe_kernel"], p["clahe_kernel"]),
            clahe_clip=p["clahe_clip"],
            clahe_nbins=256,
            flip_x=p["flip_x"],
        )
        print(pat_obj)

        self.log_signal.emit("Reading ANG file…")
        ang_data = utilities.read_ang(
            p["ang"], pat_obj.patshape, segment_grain_threshold=None
        )

        x0 = np.ravel_multi_index(p["ref_position"], ang_data.shape)
        self.log_signal.emit(
            f"Reference pattern index: {x0}  "
            f"(row={p['ref_position'][0]}, col={p['ref_position'][1]})"
        )

        euler_angles_ref = ang_data.eulers[np.unravel_index(x0, ang_data.shape)]
        pc_ref = ang_data.pc
        self.log_signal.emit(f"PC: {pc_ref}")

        os.makedirs(p["output_dir"], exist_ok=True)

        optimize_params = dict(
            init_type=p["init_type"],
            crop_fraction=p["crop_fraction"],
            max_iter=p["max_iter"],
            conv_tol=1e-3,
            n_jobs=p["n_jobs"],
            verbose=True,
            roi_slice=None,
            scan_shape=ang_data.shape,
            mask=pat_obj.get_mask(),
            use_simulated_reference=False,
            master_pattern_path=None,
            euler_angles_ref=euler_angles_ref,
            pc_ref=pc_ref,
            tilt_deg=p["tilt"],
            debug_gradients=False,
        )

        self.log_signal.emit("Starting optimization…")
        t0 = time.perf_counter()
        _result = core.optimize(pat_obj, x0, **optimize_params)
        if len(_result) == 5:
            h, h_guess, iterations, residuals, dp_norms = _result
        else:
            h, iterations, residuals, dp_norms = _result
            h_guess = None
        dt = time.perf_counter() - t0
        n_pat = iterations.size
        self.log_signal.emit(
            f"Optimization complete: {dt:.1f} s total, "
            f"{dt / n_pat * 1000:.2f} ms/pattern"
        )

        comp   = p["component"]
        date   = p["date"]
        folder = p["output_dir"]

        np.save(os.path.join(folder, f"{comp}_homographies_{date}.npy"), h)
        np.save(os.path.join(folder, f"{comp}_iterations_{date}.npy"),   iterations)
        np.save(os.path.join(folder, f"{comp}_residuals_{date}.npy"),    residuals)
        np.save(os.path.join(folder, f"{comp}_dp_norms_{date}.npy"),     dp_norms)
        if h_guess is not None:
            np.save(os.path.join(folder, f"{comp}_h_guess_{date}.npy"), h_guess)

        self.log_signal.emit(f"Results saved to: {folder}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Load Files
# ─────────────────────────────────────────────────────────────────────────────

class LoadFilesPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 1 of 5 — Load Files")
        self.setSubTitle(
            "Select your UP2 pattern file and ANG scan file, "
            "then choose where to save results."
        )

        layout = QVBoxLayout()

        # ── Input files ───────────────────────────────────────────────────────
        file_group  = QGroupBox("Input Files")
        file_layout = QFormLayout()

        self.up2_edit = QLineEdit()
        self.up2_edit.setPlaceholderText("Path to .up2 file…")
        file_layout.addRow(
            "UP2 pattern file:",
            _make_browse_row(self, self.up2_edit, "UP2 files (*.up2);;All files (*)", "Select UP2 file"),
        )

        self.ang_edit = QLineEdit()
        self.ang_edit.setPlaceholderText("Path to .ang file…")
        file_layout.addRow(
            "ANG scan file:",
            _make_browse_row(self, self.ang_edit, "ANG files (*.ang);;All files (*)", "Select ANG file"),
        )

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ── Output ────────────────────────────────────────────────────────────
        out_group  = QGroupBox("Output")
        out_layout = QFormLayout()

        self.run_name_edit = QLineEdit("MyRun")
        out_layout.addRow("Run name:", self.run_name_edit)

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Folder to save .npy results…")
        out_layout.addRow("Results folder:", _make_browse_dir(self, self.out_edit))

        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # ── Pattern preview ───────────────────────────────────────────────────
        preview_group  = QGroupBox("Pattern Preview  (first pattern in file)")
        preview_layout = QVBoxLayout()

        self._fig    = Figure(figsize=(3, 3), tight_layout=True)
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_visible(False)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setFixedHeight(260)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._info_label = QLabel("Load a UP2 file to see a pattern preview.")
        self._info_label.setStyleSheet("color: gray;")

        preview_layout.addWidget(self._canvas)
        preview_layout.addWidget(self._info_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        self.setLayout(layout)

        self.registerField("up2_path*", self.up2_edit)
        self.registerField("ang_path*", self.ang_edit)
        self.registerField("run_name*", self.run_name_edit)
        self.registerField("output_dir*", self.out_edit)

        self.up2_edit.textChanged.connect(self._update_preview)

    def _update_preview(self, path: str):
        path = path.strip()
        self._ax.set_visible(False)
        if not path or not os.path.exists(path):
            self._info_label.setText("Load a UP2 file to see a pattern preview.")
            self._canvas.draw()
            return
        try:
            import Data
            pat_obj = Data.UP2(path)
            pat = pat_obj.read_pattern(0, process=False).astype(np.float32)
            lo, hi = pat.min(), pat.max()
            pat = (pat - lo) / (hi - lo + 1e-9)
            self._ax.set_visible(True)
            self._ax.clear()
            self._ax.imshow(pat, cmap="gray", origin="upper")
            self._ax.axis("off")
            self._fig.tight_layout(pad=0.3)
            self._canvas.draw()
            self._info_label.setText(
                f"Pattern: {pat_obj.patshape[0]} × {pat_obj.patshape[1]} px    "
                f"N patterns: {pat_obj.nPatterns:,}"
            )
        except Exception as exc:
            self._info_label.setText(f"Could not read UP2: {exc}")
            self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Scan Geometry
# ─────────────────────────────────────────────────────────────────────────────

class ScanGeometryPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 2 of 5 — Scan Geometry")
        self.setSubTitle(
            "Fields marked with  ✦  are auto-populated from your files. "
            "Check them and adjust if needed."
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(12)

        # ── Tilts ─────────────────────────────────────────────────────────────
        tilt_group  = QGroupBox("Tilts")
        tilt_layout = QFormLayout()

        self.tilt = QDoubleSpinBox()
        self.tilt.setRange(0, 90)
        self.tilt.setValue(70.0)
        self.tilt.setSuffix(" °")
        self.tilt.setToolTip("Sample tilt angle in degrees (typically 70°).")

        self.det_tilt = QDoubleSpinBox()
        self.det_tilt.setRange(0, 30)
        self.det_tilt.setValue(10.0)
        self.det_tilt.setSuffix(" °")
        self.det_tilt.setToolTip("Detector tilt angle in degrees (typically 0–10°).")

        tilt_layout.addRow("Sample tilt:", self.tilt)
        tilt_layout.addRow("Detector tilt:", self.det_tilt)
        tilt_group.setLayout(tilt_layout)
        layout.addWidget(tilt_group)

        # ── EDAX Pattern Center ───────────────────────────────────────────────
        pc_group  = QGroupBox("EDAX Pattern Center  ✦  (auto-populated from ANG)")
        pc_layout = QFormLayout()

        self.pc_x = QDoubleSpinBox()
        self.pc_x.setRange(0.0, 1.0)
        self.pc_x.setDecimals(5)
        self.pc_x.setSingleStep(0.001)
        self.pc_x.setValue(0.5)
        self.pc_x.setToolTip("x*  — horizontal position of the pattern centre (EDAX fractional, 0–1).")

        self.pc_y = QDoubleSpinBox()
        self.pc_y.setRange(0.0, 1.0)
        self.pc_y.setDecimals(5)
        self.pc_y.setSingleStep(0.001)
        self.pc_y.setValue(0.5)
        self.pc_y.setToolTip("y*  — vertical position of the pattern centre (EDAX fractional, 0–1, measured from bottom).")

        self.pc_z = QDoubleSpinBox()
        self.pc_z.setRange(0.01, 3.0)
        self.pc_z.setDecimals(5)
        self.pc_z.setSingleStep(0.001)
        self.pc_z.setValue(0.65)
        self.pc_z.setToolTip("z*  — detector distance as a fraction of the pattern height.")

        pc_layout.addRow("x*  (xstar):", self.pc_x)
        pc_layout.addRow("y*  (ystar):", self.pc_y)
        pc_layout.addRow("z*  (zstar):", self.pc_z)
        pc_layout.addRow(
            _note("EDAX fractional coordinates as reported by OIM / TSL. "
                  "y* is measured from the bottom of the detector.")
        )
        pc_group.setLayout(pc_layout)
        layout.addWidget(pc_group)

        # ── Detector ──────────────────────────────────────────────────────────
        det_group  = QGroupBox("Detector")
        det_layout = QFormLayout()

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(1.0, 500.0)
        self.pixel_size.setDecimals(2)
        self.pixel_size.setValue(30.0)
        self.pixel_size.setSuffix(" µm")
        self.pixel_size.setToolTip("Physical size of one detector pixel in microns (NOTE: patterns should not be binned).")

        self.pat_h = QSpinBox()
        self.pat_h.setRange(1, 4096)
        self.pat_h.setValue(512)
        self.pat_h.setToolTip("Pattern height in pixels — auto-populated from UP2 file.")

        self.pat_w = QSpinBox()
        self.pat_w.setRange(1, 4096)
        self.pat_w.setValue(512)
        self.pat_w.setToolTip("Pattern width in pixels — auto-populated from UP2 file.")

        det_layout.addRow("Detector pixel size:", self.pixel_size)
        det_layout.addRow("Pattern height  ✦:", self.pat_h)
        det_layout.addRow("Pattern width   ✦:", self.pat_w)
        det_group.setLayout(det_layout)
        layout.addWidget(det_group)

        # ── Scan ──────────────────────────────────────────────────────────────
        scan_group  = QGroupBox("Scan  ✦  (auto-populated from ANG)")
        scan_layout = QFormLayout()

        self.rows = QSpinBox()
        self.rows.setRange(1, 99999)
        self.rows.setValue(1)
        self.rows.setToolTip("Number of scan rows (NROWS in ANG header).")

        self.cols = QSpinBox()
        self.cols.setRange(1, 99999)
        self.cols.setValue(1)
        self.cols.setToolTip("Number of scan columns (NCOLS_ODD in ANG header).")

        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.001, 10000.0)
        self.step_size.setDecimals(4)
        self.step_size.setValue(1.0)
        self.step_size.setSuffix(" µm")
        self.step_size.setToolTip("Step size between scan points in microns (XSTEP in ANG header).")

        scan_layout.addRow("Rows:", self.rows)
        scan_layout.addRow("Columns:", self.cols)
        scan_layout.addRow("Step size:", self.step_size)
        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)

        # ── Auto-populate status ──────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._status_label)

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout()
        outer.addWidget(scroll)
        self.setLayout(outer)

    def initializePage(self):
        """Called by the wizard when this page becomes visible. Auto-populate from files."""
        wiz      = self.wizard()
        up2_path = wiz.field("up2_path")
        ang_path = wiz.field("ang_path")
        populated = []
        errors    = []

        # ── Pattern shape from UP2 ────────────────────────────────────────────
        if up2_path and os.path.exists(up2_path):
            try:
                import Data
                pat_obj = Data.UP2(up2_path)
                self.pat_h.setValue(pat_obj.patshape[0])
                self.pat_w.setValue(pat_obj.patshape[1])
                populated.append("pattern shape")
            except Exception as exc:
                errors.append(f"UP2: {exc}")

        # ── Everything else from ANG ──────────────────────────────────────────
        if ang_path and os.path.exists(ang_path):
            try:
                import Data, utilities
                # Need patshape to call read_ang; fall back to current spinbox values
                try:
                    pat_obj   = Data.UP2(up2_path)
                    patshape  = pat_obj.patshape
                except Exception:
                    patshape  = (self.pat_h.value(), self.pat_w.value())

                ang_data = utilities.read_ang(
                    ang_path, patshape, segment_grain_threshold=None
                )

                # Pattern center
                pc = ang_data.pc
                self.pc_x.setValue(float(pc[0]))
                self.pc_y.setValue(float(pc[1]))
                self.pc_z.setValue(float(pc[2]))
                populated.append("pattern center")

                # Scan shape
                r, c = ang_data.shape
                self.rows.setValue(int(r))
                self.cols.setValue(int(c))
                populated.append("rows / columns")

            except Exception as exc:
                errors.append(f"ANG: {exc}")

            # Step size — parse header directly (read_ang doesn't expose it on the object)
            try:
                import re
                NUMERIC = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
                with open(ang_path, "r", errors="replace") as f:
                    for line in f:
                        if not line.startswith("#"):
                            break
                        if "XSTEP" in line:
                            val = re.findall(NUMERIC, line)
                            if val:
                                self.step_size.setValue(float(val[0]))
                                populated.append("step size")
                            break
            except Exception as exc:
                errors.append(f"XSTEP: {exc}")

        # ── Update status label ───────────────────────────────────────────────
        if populated:
            msg = "Auto-populated: " + ", ".join(populated) + "."
            if errors:
                msg += "  Warnings: " + "; ".join(errors)
            self._status_label.setText(msg)
        elif errors:
            self._status_label.setText("Could not auto-populate: " + "; ".join(errors))

    def get_params(self) -> dict:
        return {
            "tilt":       self.tilt.value(),
            "det_tilt":   self.det_tilt.value(),
            "pc_edax":    (self.pc_x.value(), self.pc_y.value(), self.pc_z.value()),
            "pixel_size": self.pixel_size.value(),
            "pat_h":      self.pat_h.value(),
            "pat_w":      self.pat_w.value(),
            "rows":       self.rows.value(),
            "cols":       self.cols.value(),
            "step_size":  self.step_size.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# IPF worker — computes colours in a background thread
# ─────────────────────────────────────────────────────────────────────────────

class IPFWorker(QThread):
    done_signal  = pyqtSignal(object, str)   # rgb_map (ndarray) | None, error_msg

    def __init__(self, ang_path: str, patshape: tuple, direction: np.ndarray):
        super().__init__()
        self.ang_path  = ang_path
        self.patshape  = patshape
        self.direction = direction

    def run(self):
        try:
            from ipf_map import compute_ipf_colors
            import utilities
            ang_data = utilities.read_ang(
                self.ang_path, self.patshape, segment_grain_threshold=None
            )
            rgb = compute_ipf_colors(ang_data.eulers, self.direction)
            self.done_signal.emit(rgb, "")
        except Exception:
            self.done_signal.emit(None, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Reference Pattern Selection
# ─────────────────────────────────────────────────────────────────────────────

class ReferencePatternPage(QWizardPage):

    _DIRECTIONS = {
        "ND  [001]": np.array([0.0, 0.0, 1.0]),
        "RD  [100]": np.array([1.0, 0.0, 0.0]),
        "TD  [010]": np.array([0.0, 1.0, 0.0]),
    }

    def __init__(self):
        super().__init__()
        self.setTitle("Step 3 of 5 — Reference Pattern")
        self.setSubTitle(
            "Use the IPF map to find a good reference position, then enter its "
            "row and column below and preview the raw pattern."
        )
        self._ipf_worker = None
        self._rgb_map    = None   # cached IPF colours (rows, cols, 3)

        # ── Top: position controls ────────────────────────────────────────────
        pos_group  = QGroupBox("Reference Position")
        pos_layout = QFormLayout()

        self.ref_row = QSpinBox()
        self.ref_row.setRange(0, 9999)
        self.ref_row.setValue(0)
        self.ref_row.setToolTip("Row index of the reference pattern (0 = top row).")

        self.ref_col = QSpinBox()
        self.ref_col.setRange(0, 9999)
        self.ref_col.setValue(0)
        self.ref_col.setToolTip("Column index of the reference pattern (0 = leftmost column).")

        pos_layout.addRow("Reference row  (y):", self.ref_row)
        pos_layout.addRow("Reference col  (x):", self.ref_col)
        pos_layout.addRow(
            _note("(0, 0) is the top-left corner of the scan. "
                  "Match this to x0 in your runner script.")
        )
        pos_group.setLayout(pos_layout)

        # ── Bottom: two side-by-side panels ──────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel — IPF map
        ipf_widget  = QWidget()
        ipf_layout  = QVBoxLayout(ipf_widget)
        ipf_layout.setContentsMargins(0, 0, 4, 0)

        ipf_header = QHBoxLayout()
        self._dir_combo = QComboBox()
        self._dir_combo.addItems(list(self._DIRECTIONS.keys()))
        self._dir_combo.setToolTip("Sample direction to use for IPF colouring.")
        self._ipf_btn = QPushButton("Compute IPF Map")
        self._ipf_btn.clicked.connect(self._start_ipf)
        ipf_header.addWidget(QLabel("Direction:"))
        ipf_header.addWidget(self._dir_combo)
        ipf_header.addWidget(self._ipf_btn)
        ipf_header.addStretch()

        self._ipf_fig = Figure(tight_layout=True)
        self._ipf_ax  = self._ipf_fig.add_subplot(111)
        self._ipf_ax.set_visible(False)
        # Reserve space for the IPF colour key in a second axes
        self._key_ax  = self._ipf_fig.add_axes([0.78, 0.60, 0.20, 0.36])
        self._key_ax.set_visible(False)
        self._ipf_canvas = FigureCanvas(self._ipf_fig)
        self._ipf_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._ipf_status = QLabel("Click 'Compute IPF Map' to generate the orientation map.")
        self._ipf_status.setStyleSheet("color: gray;")
        self._ipf_status.setWordWrap(True)

        ipf_layout.addLayout(ipf_header)
        ipf_layout.addWidget(self._ipf_canvas)
        ipf_layout.addWidget(self._ipf_status)

        # Right panel — raw pattern preview
        pat_widget  = QWidget()
        pat_layout  = QVBoxLayout(pat_widget)
        pat_layout.setContentsMargins(4, 0, 0, 0)

        pat_header = QHBoxLayout()
        self._pat_btn = QPushButton("Load Pattern Preview")
        self._pat_btn.clicked.connect(self._load_preview)
        pat_header.addWidget(self._pat_btn)
        pat_header.addStretch()

        self._pat_fig = Figure(tight_layout=True)
        self._pat_ax  = self._pat_fig.add_subplot(111)
        self._pat_ax.set_visible(False)
        self._pat_canvas = FigureCanvas(self._pat_fig)
        self._pat_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._pat_status = QLabel("Enter row / col above, then click 'Load Pattern Preview'.")
        self._pat_status.setStyleSheet("color: gray;")
        self._pat_status.setWordWrap(True)

        pat_layout.addLayout(pat_header)
        pat_layout.addWidget(self._pat_canvas)
        pat_layout.addWidget(self._pat_status)

        splitter.addWidget(ipf_widget)
        splitter.addWidget(pat_widget)
        splitter.setSizes([500, 300])

        # ── Assemble page ─────────────────────────────────────────────────────
        outer = QVBoxLayout()
        outer.addWidget(pos_group)
        outer.addWidget(splitter, stretch=1)
        self.setLayout(outer)

    # ── IPF map ───────────────────────────────────────────────────────────────

    def _start_ipf(self):
        wiz      = self.wizard()
        ang_path = wiz.field("ang_path")
        geom     = wiz.geometry_page.get_params()
        patshape = (geom["pat_h"], geom["pat_w"])
        direction = self._DIRECTIONS[self._dir_combo.currentText()]

        if not ang_path or not os.path.exists(ang_path):
            self._ipf_status.setText("No ANG file found. Go back to Step 1.")
            return

        self._ipf_btn.setEnabled(False)
        self._ipf_status.setText("Computing… (this may take a few seconds)")

        self._ipf_worker = IPFWorker(ang_path, patshape, direction)
        self._ipf_worker.done_signal.connect(self._on_ipf_done)
        self._ipf_worker.start()

    def _on_ipf_done(self, rgb_map, error: str):
        self._ipf_btn.setEnabled(True)
        if rgb_map is None:
            self._ipf_status.setText(f"Error: {error}")
            return

        self._rgb_map = rgb_map
        label = self._dir_combo.currentText().split()[0]   # "ND", "RD", or "TD"

        # ── Draw IPF map ─────────────────────────────────────────────────────
        self._ipf_ax.set_visible(True)
        self._ipf_ax.clear()
        self._ipf_ax.imshow(rgb_map, origin="upper", interpolation="nearest")
        self._ipf_ax.set_title(f"IPF Map  //  {label}", fontsize=9)
        self._ipf_ax.set_xlabel("Column", fontsize=8)
        self._ipf_ax.set_ylabel("Row", fontsize=8)
        self._ipf_ax.tick_params(labelsize=7)

        # Draw crosshair at current reference position
        self._draw_crosshair()

        # ── Draw colour key ──────────────────────────────────────────────────
        from ipf_map import plot_ipf_triangle
        self._key_ax.set_visible(True)
        self._key_ax.clear()
        plot_ipf_triangle(self._key_ax, n=150)

        self._ipf_fig.tight_layout(pad=0.5)
        self._ipf_canvas.draw()

        rows, cols, _ = rgb_map.shape
        self._ipf_status.setText(f"IPF map computed  ({rows} × {cols} scan points).")

        # Update crosshair whenever spinboxes change
        self.ref_row.valueChanged.connect(self._update_crosshair)
        self.ref_col.valueChanged.connect(self._update_crosshair)

    def _draw_crosshair(self):
        """Overlay a white crosshair at the current reference position."""
        if not self._ipf_ax.get_visible():
            return
        row = self.ref_row.value()
        col = self.ref_col.value()
        # Remove old crosshair lines if present
        for artist in list(self._ipf_ax.lines):
            artist.remove()
        rgb, _ = self._rgb_map.shape[0], self._rgb_map.shape[1]
        arm = max(rgb, _) * 0.04
        self._ipf_ax.plot([col - arm, col + arm], [row, row],
                          color="white", linewidth=1.5, solid_capstyle="round")
        self._ipf_ax.plot([col, col], [row - arm, row + arm],
                          color="white", linewidth=1.5, solid_capstyle="round")
        self._ipf_canvas.draw_idle()

    def _update_crosshair(self):
        self._draw_crosshair()

    # ── Pattern preview ───────────────────────────────────────────────────────

    def _load_preview(self):
        wiz   = self.wizard()
        path  = wiz.field("up2_path")
        geom  = wiz.geometry_page.get_params()
        ncols = geom["cols"]
        row   = self.ref_row.value()
        col   = self.ref_col.value()
        idx   = row * ncols + col

        if not path or not os.path.exists(path):
            self._pat_status.setText("No UP2 file found. Go back to Step 1.")
            return
        try:
            import Data
            pat_obj = Data.UP2(path)
            if idx >= pat_obj.nPatterns:
                self._pat_status.setText(
                    f"Index {idx} out of range (file has {pat_obj.nPatterns} patterns)."
                )
                return
            pat = pat_obj.read_pattern(idx, process=False).astype(np.float32)
            lo, hi = pat.min(), pat.max()
            pat = (pat - lo) / (hi - lo + 1e-9)
            self._pat_ax.set_visible(True)
            self._pat_ax.clear()
            self._pat_ax.imshow(pat, cmap="gray", origin="upper")
            self._pat_ax.set_title(f"Row {row},  Col {col}  (index {idx})", fontsize=9)
            self._pat_ax.axis("off")
            self._pat_fig.tight_layout(pad=0.3)
            self._pat_canvas.draw()
            self._pat_status.setText(
                f"Pattern  {pat_obj.patshape[0]} × {pat_obj.patshape[1]} px"
            )
        except Exception as exc:
            self._pat_status.setText(f"Error: {exc}")

    def get_params(self) -> dict:
        return {
            "ref_position": (self.ref_row.value(), self.ref_col.value()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Pattern Processing
# ─────────────────────────────────────────────────────────────────────────────

class PatternProcessingPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 4 of 5 — Pattern Processing")
        self.setSubTitle(
            "These settings control how each diffraction pattern is pre-processed "
            "before the cross-correlation. The defaults work well for most scans."
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(12)

        # ── Frequency filtering ───────────────────────────────────────────────
        filt_group  = QGroupBox("Frequency Filtering")
        filt_layout = QFormLayout()

        self.high_pass = QDoubleSpinBox()
        self.high_pass.setRange(0, 200)
        self.high_pass.setValue(10.0)
        self.high_pass.setToolTip(
            "High-pass Gaussian filter sigma. Removes slowly-varying background "
            "(e.g. intensity gradients across the detector). "
            "Larger value = more aggressive. Typical range: 5–15."
        )

        self.low_pass = QDoubleSpinBox()
        self.low_pass.setRange(0, 100)
        self.low_pass.setValue(0.0)
        self.low_pass.setToolTip(
            "Low-pass Gaussian filter sigma. Smooths high-frequency noise. "
            "0 = disabled (recommended for most cases)."
        )

        filt_layout.addRow("High-pass sigma:", self.high_pass)
        filt_layout.addRow("Low-pass sigma:", self.low_pass)
        filt_group.setLayout(filt_layout)
        layout.addWidget(filt_group)

        # ── Mask ──────────────────────────────────────────────────────────────
        mask_group  = QGroupBox("Mask")
        mask_layout = QFormLayout()

        self.mask_type = QComboBox()
        self.mask_type.addItems(["None", "circular", "center_cross"])
        self.mask_type.setToolTip(
            "Mask applied before cross-correlation.\n"
            "  None           — no mask\n"
            "  circular       — keep only a circular region of the pattern\n"
            "  center_cross   — suppress the bright beam-stop cross at the centre"
        )

        mask_layout.addRow("Mask type:", self.mask_type)
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # ── CLAHE ─────────────────────────────────────────────────────────────
        clahe_group  = QGroupBox("CLAHE  (Contrast Limited Adaptive Histogram Equalisation)")
        clahe_layout = QFormLayout()

        self.clahe_kernel = QSpinBox()
        self.clahe_kernel.setRange(1, 32)
        self.clahe_kernel.setValue(6)
        self.clahe_kernel.setToolTip(
            "Tile size for local histogram equalisation in pixels. "
            "Smaller = more local contrast enhancement. Typical: 4–8."
        )

        self.clahe_clip = QDoubleSpinBox()
        self.clahe_clip.setRange(0.001, 1.0)
        self.clahe_clip.setValue(0.01)
        self.clahe_clip.setSingleStep(0.005)
        self.clahe_clip.setToolTip(
            "Clip limit — controls how aggressively contrast is enhanced. "
            "Lower = less clipping. Typical: 0.01–0.05."
        )

        clahe_layout.addRow("Kernel size:", self.clahe_kernel)
        clahe_layout.addRow("Clip limit:", self.clahe_clip)
        clahe_group.setLayout(clahe_layout)
        layout.addWidget(clahe_group)

        # ── Orientation ───────────────────────────────────────────────────────
        orient_group  = QGroupBox("Orientation")
        orient_layout = QFormLayout()

        self.flip_x = QCheckBox("Flip patterns vertically  (flip_x)")
        self.flip_x.setToolTip(
            "Flip each pattern about the horizontal axis before processing. "
            "Enable if your patterns appear upside-down relative to the scan map."
        )

        orient_layout.addRow("", self.flip_x)
        orient_group.setLayout(orient_layout)
        layout.addWidget(orient_group)

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout()
        outer.addWidget(scroll)
        self.setLayout(outer)

    def get_params(self) -> dict:
        return {
            "high_pass_sigma": self.high_pass.value(),
            "low_pass_sigma":  self.low_pass.value(),
            "flip_x":          self.flip_x.isChecked(),
            "mask_type":       self.mask_type.currentText(),
            "clahe_kernel":    self.clahe_kernel.value(),
            "clahe_clip":      self.clahe_clip.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Optimization + Run
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationRunPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 5 of 5 — Optimization & Run")
        self.setSubTitle(
            "Review the full settings summary, adjust optimization parameters, "
            "then click Run. The Finish button unlocks once the pipeline completes."
        )
        self._worker = None
        self._done   = False

        layout = QVBoxLayout()
        layout.setSpacing(10)

        # ── Optimization params ───────────────────────────────────────────────
        opt_group  = QGroupBox("Optimization Parameters")
        opt_layout = QFormLayout()

        self.max_iter = QSpinBox()
        self.max_iter.setRange(1, 1000)
        self.max_iter.setValue(150)
        self.max_iter.setToolTip("Maximum IC-GN iterations per pattern before giving up. 100–200 is typical.")

        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 64)
        self.n_jobs.setValue(8)
        self.n_jobs.setToolTip(
            "Number of CPU cores to use in parallel. -1 = use all available cores."
        )

        self.crop_fraction = QDoubleSpinBox()
        self.crop_fraction.setRange(0.1, 0.99)
        self.crop_fraction.setValue(0.9)
        self.crop_fraction.setSingleStep(0.05)
        self.crop_fraction.setToolTip(
            "Fraction of each pattern used for correlation (crops edges slightly). "
            "0.9 is a good default."
        )

        self.init_type = QComboBox()
        self.init_type.addItems(["none", "partial", "full"])
        self.init_type.setToolTip(
            "How to initialise the homography before iterating.\n"
            "  none    — identity start (fastest, good for small strains)\n"
            "  partial — use the previous pattern as a warm start\n"
            "  full    — full initialisation (slowest, most robust)"
        )

        opt_layout.addRow("Max iterations:", self.max_iter)
        opt_layout.addRow("n_jobs:", self.n_jobs)
        opt_layout.addRow("Crop fraction:", self.crop_fraction)
        opt_layout.addRow("Init type:", self.init_type)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # ── Settings summary ──────────────────────────────────────────────────
        layout.addWidget(QLabel("Settings summary:"))
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setFixedHeight(160)
        self._summary.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._summary)

        # ── Run button ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶   Run Pipeline")
        self._run_btn.setFixedHeight(44)
        self._run_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; "
            "background-color: #2e7d32; color: white; border-radius: 6px;"
        )
        self._run_btn.clicked.connect(self._start_run)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._status_label)

        # ── Log ───────────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Log:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._log)

        self.setLayout(layout)

    def initializePage(self):
        wiz  = self.wizard()
        geom = wiz.geometry_page.get_params()
        ref  = wiz.reference_page.get_params()
        proc = wiz.processing_page.get_params()

        lines = [
            f"Run name         : {wiz.field('run_name')}",
            f"UP2 file         : {wiz.field('up2_path')}",
            f"ANG file         : {wiz.field('ang_path')}",
            f"Output folder    : {wiz.field('output_dir')}",
            f"",
            f"Reference pos    : row={ref['ref_position'][0]}, col={ref['ref_position'][1]}",
            f"Sample tilt      : {geom['tilt']} °",
            f"Detector tilt    : {geom['det_tilt']} °",
            f"Pattern center   : x*={geom['pc_edax'][0]:.5f}  y*={geom['pc_edax'][1]:.5f}  z*={geom['pc_edax'][2]:.5f}",
            f"Rows × Columns   : {geom['rows']} × {geom['cols']}",
            f"Step size        : {geom['step_size']} µm",
            f"",
            f"High-pass σ      : {proc['high_pass_sigma']}",
            f"Flip patterns    : {proc['flip_x']}",
            f"Mask type        : {proc['mask_type']}",
            f"CLAHE kernel     : {proc['clahe_kernel']}",
        ]
        self._summary.setPlainText("\n".join(lines))

        # Reset state if the user went Back and returned
        self._done = False
        self._run_btn.setEnabled(True)
        self._status_label.setText("")
        self._progress.setVisible(False)

    def _start_run(self):
        wiz    = self.wizard()
        params = {}
        params.update(wiz.geometry_page.get_params())
        params.update(wiz.reference_page.get_params())
        params.update(wiz.processing_page.get_params())
        params.update({
            "max_iter":      self.max_iter.value(),
            "n_jobs":        self.n_jobs.value(),
            "crop_fraction": self.crop_fraction.value(),
            "init_type":     self.init_type.currentText(),
            "up2":           wiz.field("up2_path"),
            "ang":           wiz.field("ang_path"),
            "component":     wiz.field("run_name"),
            "output_dir":    wiz.field("output_dir"),
            "date":          datetime.date.today().strftime("%B_%d_%Y"),
        })

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log.clear()
        self._status_label.setText("Running…")
        self._status_label.setStyleSheet("color: #1565c0; font-weight: bold;")

        self._worker = PipelineWorker(params)
        self._worker.log_signal.connect(self._append_log)
        self._worker.done_signal.connect(self._on_done)
        self._worker.start()

    def _append_log(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_done(self, success: bool, msg: str):
        self._progress.setVisible(False)
        if success:
            self._status_label.setText("Done — results saved successfully.")
            self._status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self._done = True
            self.completeChanged.emit()
        else:
            self._status_label.setText("Error — see log for details.")
            self._status_label.setStyleSheet("color: #c62828; font-weight: bold;")
            self._log.append("\n--- TRACEBACK ---\n" + msg)
            self._run_btn.setEnabled(True)

    def isComplete(self) -> bool:
        return self._done


# ─────────────────────────────────────────────────────────────────────────────
# Main wizard
# ─────────────────────────────────────────────────────────────────────────────

class HREBSDWizard(QWizard):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIC-HREBSD Pipeline")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(760, 720)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)

        self.files_page      = LoadFilesPage()
        self.geometry_page   = ScanGeometryPage()
        self.reference_page  = ReferencePatternPage()
        self.processing_page = PatternProcessingPage()
        self.run_page        = OptimizationRunPage()

        self.addPage(self.files_page)
        self.addPage(self.geometry_page)
        self.addPage(self.reference_page)
        self.addPage(self.processing_page)
        self.addPage(self.run_page)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    wiz = HREBSDWizard()
    wiz.show()
    sys.exit(app.exec())

"""
gui_pages.py — all QWizardPage subclasses for the DIC-HREBSD wizard.

  LoadFilesPage         — Step 1: select UP2, ANG, output folder
  ScanGeometryPage      — Step 2: tilts, PC, detector / scan geometry
  PatternProcessingPage — Step 3: frequency filters, mask, flip
  ROISelectionPage      — Step 4: grain segmentation + region of interest
  ReferencePatternPage  — Step 5: pick and preview the reference pattern
  OptimizationRunPage   — Step 6: run the pipeline, launch vis dialog
"""

import os
import re
import datetime

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QWizardPage,
    QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QRadioButton, QButtonGroup, QSlider,
    QTextEdit, QWidget, QScrollArea,
    QSizePolicy, QSplitter, QDialog, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, pyqtProperty

from gui_theme import THEME, _make_browse_row, _make_browse_dir, _note
from gui_workers import PipelineWorker, IPFWorker, SegmentWorker, PatternPreviewWorker, AngLoaderWorker, SimRefWorker, PcEulerRefineWorker
from gui_visualization import VisualizationDialog


from gui_materials import _load_material_presets, NewMaterialDialog
from multiple_ref import ReferencePatternSet, select_references


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Load Files
# ─────────────────────────────────────────────────────────────────────────────

class LoadFilesPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 1 of 6 — Load Files and Material Parameters")
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
            _make_browse_row(
                self, self.ang_edit, "ANG files (*.ang);;All files (*)", "Select ANG file",
                start_dir_fn=lambda: os.path.dirname(self.up2_edit.text()) if self.up2_edit.text() else None,
            ),
        )

        self.master_pattern_edit = QLineEdit()
        self.master_pattern_edit.setPlaceholderText(
            "Path to EMsoft .h5 master pattern (optional — only needed for simulated references)…"
        )
        file_layout.addRow(
            "Master pattern (.h5):",
            _make_browse_row(
                self, self.master_pattern_edit,
                "HDF5 master pattern (*.h5 *.hdf5);;All files (*)",
                "Select master pattern file",
                start_dir_fn=lambda: os.path.dirname(self.up2_edit.text()) if self.up2_edit.text() else None,
            ),
        )

        self._ang_load_status = QLabel("")
        self._ang_load_status.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addRow("", self._ang_load_status)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ── Output ────────────────────────────────────────────────────────────
        out_group  = QGroupBox("Output")
        out_layout = QFormLayout()

        self.run_name_edit = QLineEdit("MyRun")
        out_layout.addRow("Run name (subfolder):", self.run_name_edit)

        # Browse selects the parent directory; the subfolder is created from run_name
        self._base_dir_edit = QLineEdit()
        self._base_dir_edit.setPlaceholderText("Parent directory for results…")
        out_layout.addRow("Save results in:", _make_browse_dir(self, self._base_dir_edit))

        # Hidden field that stores the computed full output path for the wizard
        self.out_edit = QLineEdit()
        self.out_edit.setVisible(False)

        # Live preview of the full path that will be created
        self._path_preview = QLabel()
        self._path_preview.setStyleSheet("color: gray; font-style: italic;")
        self._path_preview.setWordWrap(True)
        out_layout.addRow("Will create:", self._path_preview)

        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        self._base_dir_edit.textChanged.connect(self._update_output_dir)
        self.run_name_edit.textChanged.connect(self._update_output_dir)
        self._update_output_dir()   # initialise preview

        # ── Bottom row: pattern preview (left) + material properties (right) ──
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        # Pattern preview
        preview_group  = QGroupBox("Pattern Preview")
        preview_layout = QVBoxLayout()

        _bg          = THEME["surface_bg"]
        self._fig    = Figure(figsize=(2, 2), tight_layout=True, facecolor=_bg)
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_facecolor(_bg)
        self._ax.set_visible(False)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_bg};")
        self._canvas.setFixedSize(170, 170)

        self._info_label = QLabel("Load a UP2 file.")
        self._info_label.setStyleSheet("color: gray; font-size: 10px;")
        self._info_label.setWordWrap(True)

        preview_layout.addWidget(self._canvas)
        preview_layout.addWidget(self._info_label)
        preview_group.setLayout(preview_layout)
        bottom_row.addWidget(preview_group)

        # Material properties
        mat_group  = QGroupBox("Material Properties")
        mat_layout = QFormLayout()

        self._presets = _load_material_presets()
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("— select preset —", userData=None)
        for preset in self._presets:
            self._preset_combo.addItem(preset["name"], userData=preset)
        self._preset_combo.currentIndexChanged.connect(self._apply_material_preset)
        mat_layout.addRow("Crystal preset:", self._preset_combo)

        # Structure / elastic constants display "—" when no material preset
        # is selected, so the user can see at a glance that the traction-free
        # boundary condition is not yet populated.
        self._struct_lbl = QLabel("—")
        mat_layout.addRow("Structure:", self._struct_lbl)

        def _make_C_spin():
            sb = QDoubleSpinBox()
            sb.setRange(0, 9999); sb.setDecimals(1); sb.setSuffix(" GPa")
            # specialValueText shows when the value == minimum (0 here).
            sb.setSpecialValueText("—")
            sb.setValue(0)
            sb.setReadOnly(True)
            sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            return sb

        self._C11 = _make_C_spin()
        self._C12 = _make_C_spin()
        self._C44 = _make_C_spin()

        self._new_material_btn = QPushButton("New material…")
        self._new_material_btn.clicked.connect(self._open_new_material_dialog)

        mat_layout.addRow("C\u2081\u2081:", self._C11)
        mat_layout.addRow("C\u2081\u2082:", self._C12)
        mat_layout.addRow("C\u2084\u2084:", self._C44)
        mat_layout.addRow("", self._new_material_btn)
        mat_layout.addRow(_note("Used to compute strain from homographies."))
        mat_group.setLayout(mat_layout)
        bottom_row.addWidget(mat_group, stretch=1)

        layout.addLayout(bottom_row)

        # NOTE: the "Open Results Viewer" button has been moved out of Step 1
        # and into the wizard's bottom button row (next to Help) so it's
        # reachable from every page.  See HREBSDWizard in Run_GUI.py.

        self.setLayout(layout)

        # No mandatory (*) so Next is never greyed out during development.
        # Default values let you click through all steps without real files.
        self.registerField("up2_path",  self.up2_edit)
        self.registerField("ang_path",  self.ang_edit)
        self.registerField("master_pattern_path", self.master_pattern_edit)
        self.registerField("run_name",  self.run_name_edit)
        self.registerField("output_dir", self.out_edit)

        self.run_name_edit.setText("test_run")
        self.out_edit.setText(os.path.join(os.path.expanduser("~"), "hrebsd_results"))

        self.up2_edit.textChanged.connect(self._update_preview)
        self.ang_edit.textChanged.connect(self._on_ang_path_changed)
        self._ang_loader = None   # keep reference so the thread isn't GC'd

    # ── ANG background loading ────────────────────────────────────────────────

    def _get_patshape(self):
        """Return patshape from the UP2 header, or (512, 512) as fallback."""
        up2_path = self.up2_edit.text()
        if up2_path and os.path.isfile(up2_path):
            try:
                import Data
                return Data.UP2(up2_path).patshape
            except Exception:
                pass
        return (512, 512)

    def _on_ang_path_changed(self, path):
        wiz = self.wizard()
        if not path or not os.path.isfile(path):
            if wiz:
                wiz.ang_data        = None
                wiz.ang_loaded_path = ""
            self._ang_load_status.setText("")
            return

        patshape = self._get_patshape()

        # Skip if already cached for this exact path + patshape
        if (wiz and wiz.ang_data is not None
                and wiz.ang_loaded_path == path
                and wiz.ang_loaded_patshape == patshape):
            self._ang_load_status.setText("ANG loaded.")
            return

        if wiz:
            wiz.ang_data        = None
            wiz.ang_loaded_path = ""

        self._ang_load_status.setText("Loading ANG…")

        if self._ang_loader is not None and self._ang_loader.isRunning():
            self._ang_loader.done_signal.disconnect()
            self._ang_loader.error_signal.disconnect()

        self._ang_loader = AngLoaderWorker(path, patshape)
        self._ang_loader.done_signal.connect(
            lambda d: self._on_ang_loaded(d, path, patshape)
        )
        self._ang_loader.error_signal.connect(self._on_ang_error)
        self._ang_loader.start()

    def _on_ang_loaded(self, ang_data, path, patshape):
        wiz = self.wizard()
        if wiz:
            wiz.ang_data            = ang_data
            wiz.ang_loaded_path     = path
            wiz.ang_loaded_patshape = patshape
        rows, cols = ang_data.shape
        self._ang_load_status.setText(
            f"ANG loaded — Scan Shape: {rows}×{cols}"
        )
        self._ang_load_status.setStyleSheet("color: #88cc88; font-style: italic;")

    def _on_ang_error(self, msg: str):
        self._ang_load_status.setText("ANG load failed — check console.")
        self._ang_load_status.setStyleSheet("color: #cc4444; font-style: italic;")
        print("\n--- ANG load error ---\n" + msg)

    def _apply_material_preset(self, index: int):
        preset = self._preset_combo.itemData(index)
        if preset is None:
            # Placeholder selected — wipe the elastic constants back to the
            # "not yet populated" state so the user can't accidentally run
            # the pipeline with stale numbers from a previous selection.
            self._C11.setValue(0)
            self._C12.setValue(0)
            self._C44.setValue(0)
            self._struct_lbl.setText("—")
            return
        ec = preset.get("elastic_constants_GPa", {})
        self._C11.setValue(ec.get("C11", 0))
        self._C12.setValue(ec.get("C12", 0))
        self._C44.setValue(ec.get("C44", 0))
        self._struct_lbl.setText(preset.get("structure", "—"))

    def _open_new_material_dialog(self):
        dlg = NewMaterialDialog(parent=self)
        dlg.saved.connect(self._on_new_material_saved)
        dlg.exec()

    def _on_new_material_saved(self, preset: dict):
        """Add the new preset to the dropdown and select it."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.addItem(preset["name"], userData=preset)
        # Re-sort by name (skip index 0 — the placeholder)
        items = [(self._preset_combo.itemText(i), self._preset_combo.itemData(i))
                 for i in range(1, self._preset_combo.count())]
        items.sort(key=lambda t: t[0])
        for i, (text, data) in enumerate(items, start=1):
            self._preset_combo.setItemText(i, text)
            self._preset_combo.setItemData(i, data)
        new_idx = next(
            (i for i in range(1, self._preset_combo.count())
             if self._preset_combo.itemText(i) == preset["name"]),
            self._preset_combo.count() - 1,
        )
        self._preset_combo.blockSignals(False)
        self._preset_combo.setCurrentIndex(new_idx)
        self._struct_lbl.setText(preset.get("structure", "cubic"))

    def get_material_params(self) -> dict:
        return {
            "crystal_C11":       self._C11.value(),
            "crystal_C12":       self._C12.value(),
            "crystal_C44":       self._C44.value(),
            "crystal_structure": self._struct_lbl.text(),
        }

    def _update_output_dir(self):
        base = self._base_dir_edit.text().strip()
        name = self.run_name_edit.text().strip()
        if base and name:
            full = os.path.join(base, name)
            self._path_preview.setText(full)
            self._path_preview.setStyleSheet("color: white; font-style: italic;")
        elif base:
            full = base
            self._path_preview.setText("(enter a run name above)")
            self._path_preview.setStyleSheet("color: gray; font-style: italic;")
        else:
            full = ""
            self._path_preview.setText("(select a parent folder above)")
            self._path_preview.setStyleSheet("color: gray; font-style: italic;")
        self.out_edit.setText(full)

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

    # NOTE: _launch_vis_dialog used to live here.  It's been moved to
    # HREBSDWizard in Run_GUI.py so the button can sit next to Help on
    # every page of the wizard, not just on Step 1.


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Scan Geometry
# ─────────────────────────────────────────────────────────────────────────────

class ScanGeometryPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 2 of 6 — Scan Geometry")
        self.setSubTitle(
            "Fields marked with  ✦  are auto-populated from your files "
            "Check them and adjust if needed."
        )

        # ── Two-column outer layout ───────────────────────────────────────────
        columns   = QHBoxLayout()
        left_col  = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(10)
        right_col.setSpacing(10)

        # ══ LEFT COLUMN ═══════════════════════════════════════════════════════

        # ── Tilts ─────────────────────────────────────────────────────────────
        tilt_group  = QGroupBox("Tilts")
        tilt_layout = QFormLayout()

        self.tilt = QDoubleSpinBox()
        self.tilt.setRange(0, 90)
        self.tilt.setValue(70.0)
        self.tilt.setSuffix(" °")

        self.det_tilt = QDoubleSpinBox()
        self.det_tilt.setRange(0, 30)
        self.det_tilt.setValue(10.0)
        self.det_tilt.setSuffix(" °")

        self.identity_rotation = QCheckBox("Use identity R (skip sample↔detector rotation)")
        self.identity_rotation.setChecked(False)
        self.identity_rotation.setToolTip(
            "When checked, R = I instead of the sample→detector chain from "
            "rotation_matrix_passive_version2(det_tilt, sample_tilt).\n"
            "Use this to leave strain/rotation tensors in whatever frame "
            "F2strain produces (no frame change applied)."
        )

        tilt_layout.addRow("Sample tilt:", self.tilt)
        tilt_layout.addRow("Detector tilt:", self.det_tilt)
        tilt_layout.addRow(self.identity_rotation)
        tilt_group.setLayout(tilt_layout)
        left_col.addWidget(tilt_group)

        # ── EDAX Pattern Center ───────────────────────────────────────────────
        pc_group  = QGroupBox("EDAX Pattern Center  ✦  (auto-populated from ANG)")
        pc_layout = QFormLayout()

        self.pc_x = QDoubleSpinBox()
        self.pc_x.setRange(0.0, 1.0)
        self.pc_x.setDecimals(5)
        self.pc_x.setSingleStep(0.001)
        self.pc_x.setValue(0.5)

        self.pc_y = QDoubleSpinBox()
        self.pc_y.setRange(0.0, 1.0)
        self.pc_y.setDecimals(5)
        self.pc_y.setSingleStep(0.001)
        self.pc_y.setValue(0.5)

        self.pc_z = QDoubleSpinBox()
        self.pc_z.setRange(0.01, 3.0)
        self.pc_z.setDecimals(5)
        self.pc_z.setSingleStep(0.001)
        self.pc_z.setValue(0.65)

        pc_layout.addRow("x*  (xstar):", self.pc_x)
        pc_layout.addRow("y*  (ystar):", self.pc_y)
        pc_layout.addRow("z*  (zstar):", self.pc_z)
        pc_layout.addRow(
            _note("EDAX fractional coordinates as reported by OIM / TSL. "
                  "y* is measured from the bottom of the detector.")
        )
        pc_group.setLayout(pc_layout)
        left_col.addWidget(pc_group)

        left_col.addStretch()

        # ══ RIGHT COLUMN ══════════════════════════════════════════════════════

        # ── Detector ──────────────────────────────────────────────────────────
        det_group  = QGroupBox("Detector")
        det_layout = QFormLayout()

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(1.0, 500.0)
        self.pixel_size.setDecimals(2)
        self.pixel_size.setValue(30.0)
        self.pixel_size.setSuffix(" µm")

        self.pat_h = QSpinBox()
        self.pat_h.setRange(1, 4096)
        self.pat_h.setValue(512)

        self.pat_w = QSpinBox()
        self.pat_w.setRange(1, 4096)
        self.pat_w.setValue(512)

        det_layout.addRow("Detector pixel size:", self.pixel_size)
        det_layout.addRow("Pattern height  ✦:", self.pat_h)
        det_layout.addRow("Pattern width   ✦:", self.pat_w)
        det_group.setLayout(det_layout)
        right_col.addWidget(det_group)

        # ── Scan ──────────────────────────────────────────────────────────────
        scan_group  = QGroupBox("Scan  ✦  (auto-populated from ANG)")
        scan_layout = QFormLayout()

        self.rows = QSpinBox()
        self.rows.setRange(1, 99999)
        self.rows.setValue(1)

        self.cols = QSpinBox()
        self.cols.setRange(1, 99999)
        self.cols.setValue(1)

        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.001, 10000.0)
        self.step_size.setDecimals(4)
        self.step_size.setValue(1.0)
        self.step_size.setSuffix(" µm")

        scan_layout.addRow("Rows:", self.rows)
        scan_layout.addRow("Columns:", self.cols)
        scan_layout.addRow("Step size:", self.step_size)
        scan_group.setLayout(scan_layout)
        right_col.addWidget(scan_group)

        # ── Scan Strategy ─────────────────────────────────────────────────────
        strategy_group  = QGroupBox("Scan Strategy")
        strategy_layout = QVBoxLayout()

        # Dropdown — itemData carries the canonical key the pipeline uses
        # (matches the keys consumed by pc_homography_correction.make_scan_grid).
        self._strategy_combo = QComboBox()
        self._strategy_combo.addItem("Standard (origin: upper-left, x →, y ↓)",      userData="standard")
        self._strategy_combo.addItem("Lower Left  (origin: lower-left, x →, y ↑)",    userData="lower_left")
        self._strategy_combo.addItem("Lower Right (origin: lower-right, x ←, y ↑)",   userData="lower_right")
        self._strategy_combo.addItem("Direct Electron  (origin: upper-right, x ←, y ↓)", userData="direct_electron")
        # Lower-left is the GUI default (most common scan-acquisition convention).
        self._strategy_combo.setCurrentIndex(1)
        strategy_layout.addWidget(self._strategy_combo)

        self._apply_pc_correction = QCheckBox("Apply pattern centre drift correction")
        self._apply_pc_correction.setChecked(True)
        strategy_layout.addSpacing(4)
        strategy_layout.addWidget(self._apply_pc_correction)

        self._pc_warning = QLabel(
            "\u26a0\ufe0f  Warning: for large scans, not accounting for PC shifts "
            "can result in errors in the strain of significant magnitude."
        )
        self._pc_warning.setWordWrap(True)
        self._pc_warning.setStyleSheet("color: #fab387; font-size: 11px;")
        self._pc_warning.setVisible(False)
        strategy_layout.addWidget(self._pc_warning)
        self._apply_pc_correction.stateChanged.connect(
            lambda state: self._pc_warning.setVisible(state == 0)
        )

        strategy_group.setLayout(strategy_layout)
        right_col.addWidget(strategy_group)

        # ── Strain formulation ────────────────────────────────────────────────
        strain_group  = QGroupBox("Strain Formulation")
        strain_layout = QVBoxLayout()
        self._small_strain = QCheckBox("Use small-strain formulation")
        self._small_strain.setChecked(False)
        strain_layout.addWidget(self._small_strain)
        strain_group.setLayout(strain_layout)
        right_col.addWidget(strain_group)

        # ── Auto-populate status ──────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._status_label.setWordWrap(True)
        right_col.addWidget(self._status_label)

        right_col.addStretch()

        # ── Assemble columns ──────────────────────────────────────────────────
        columns.addLayout(left_col,  stretch=1)
        columns.addLayout(right_col, stretch=1)
        self.setLayout(columns)

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
                try:
                    pat_obj  = Data.UP2(up2_path)
                    patshape = pat_obj.patshape
                except Exception:
                    patshape = (self.pat_h.value(), self.pat_w.value())

                wiz      = self.wizard()
                ang_data = (
                    wiz.ang_data
                    if wiz and wiz.ang_data is not None
                       and wiz.ang_loaded_path == ang_path
                    else utilities.read_ang(ang_path, patshape, segment_grain_threshold=None)
                )

                pc = ang_data.pc
                self.pc_x.setValue(float(pc[0]))
                self.pc_y.setValue(float(pc[1]))
                self.pc_z.setValue(float(pc[2]))
                populated.append("pattern center")

                r, c = ang_data.shape
                self.rows.setValue(int(r))
                self.cols.setValue(int(c))
                populated.append("rows / columns")

            except Exception as exc:
                errors.append(f"ANG: {exc}")

            # Step size — parse header directly
            try:
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
            "tilt":                self.tilt.value(),
            "det_tilt":            self.det_tilt.value(),
            "identity_rotation":   self.identity_rotation.isChecked(),
            "pc_edax":             (self.pc_x.value(), self.pc_y.value(), self.pc_z.value()),
            "pixel_size":          self.pixel_size.value(),
            "pat_h":               self.pat_h.value(),
            "pat_w":               self.pat_w.value(),
            "rows":                self.rows.value(),
            "cols":                self.cols.value(),
            "step_size":           self.step_size.value(),
            "scan_strategy":       (self._strategy_combo.currentData() or "standard"),
            "apply_pc_correction": self._apply_pc_correction.isChecked(),
            "small_strain":        self._small_strain.isChecked(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Grain Segmentation + Region of Interest
# ─────────────────────────────────────────────────────────────────────────────

class ROISelectionPage(QWizardPage):

    _DIRECTIONS = {
        "ND  [001]": np.array([0.0, 0.0, 1.0]),
        "RD  [100]": np.array([1.0, 0.0, 0.0]),
        "TD  [010]": np.array([0.0, 1.0, 0.0]),
    }

    def __init__(self):
        super().__init__()
        self.setTitle("Step 4 of 6 — Grain Segmentation & Region of Interest")
        self.setSubTitle(
            "Optionally segment grains, then define a region of interest. "
            "A yellow box marks the ROI on both maps."
        )
        self._ipf_worker  = None
        self._seg_worker  = None
        self._rgb_map     = None
        self._grain_ids             = None
        self._kam                   = None
        self._grain_remapped        = None
        self._grain_remap           = None
        self._grain_lut             = None
        self._grain_norm            = None
        self._grain_discrete_colors = None
        self._roi_rects             = [None, None]

        # ── Grain segmentation group ──────────────────────────────────────────
        seg_group  = QGroupBox("Grain Segmentation  (optional)")
        seg_layout = QFormLayout()

        self.seg_threshold = QDoubleSpinBox()
        self.seg_threshold.setRange(0.1, 30.0)
        self.seg_threshold.setValue(2.0)
        self.seg_threshold.setSuffix(" °")

        self._misor_warning = QLabel(
            "\u26a0\ufe0f  Warning: Remapping is not currently implemented, so increased error may occur" \
            "when the misorientation threshold is above ~5°. Use with caution and check results carefully if adjusting above this value."
        )
        self._misor_warning.setWordWrap(True)
        self._misor_warning.setStyleSheet("color: #fab387; font-size: 11px;")
        self._misor_warning.setVisible(False)
        self.seg_threshold.valueChanged.connect(
            lambda v: self._misor_warning.setVisible(v > 5.0)
        )

        self.min_grain_size = QSpinBox()
        self.min_grain_size.setRange(1, 99999)
        self.min_grain_size.setValue(1)

        self._seg_btn = QPushButton("Run Segmentation")
        self._seg_btn.clicked.connect(self._start_segmentation)

        self._seg_status = QLabel(" ")
        self._seg_status.setStyleSheet("color: gray;")
        self._seg_status.setWordWrap(True)

        seg_btn_row = QWidget()
        sbh = QHBoxLayout(seg_btn_row)
        sbh.setContentsMargins(0, 0, 0, 0)
        sbh.addWidget(self._seg_btn)
        sbh.addStretch()

        self._dir_combo = QComboBox()
        self._dir_combo.addItems(list(self._DIRECTIONS.keys()))
        self._dir_combo.currentIndexChanged.connect(self._recompute_ipf)

        dir_w = QWidget(); dh = QHBoxLayout(dir_w); dh.setContentsMargins(0, 0, 0, 0)
        dh.addWidget(self._dir_combo); dh.addStretch()

        seg_layout.addRow("Threshold:", self.seg_threshold)
        seg_layout.addRow("", self._misor_warning)
        seg_layout.addRow("Min grain size:", self.min_grain_size)
        seg_layout.addRow("", seg_btn_row)
        seg_layout.addRow(self._seg_status)
        seg_group.setLayout(seg_layout)

        # ── ROI + IPF direction group ─────────────────────────────────────────
        roi_group  = QGroupBox("Select a Region of Interest")
        roi_layout = QFormLayout()
        self._roi_form_layout = roi_layout

        # Mode: rectangle vs grain
        self._rect_radio  = QRadioButton("Rectangle")
        self._grain_radio = QRadioButton("Grain")
        self._rect_radio.setChecked(True)
        self._grain_radio.setEnabled(False)   # enabled after segmentation runs

        self._roi_mode_grp = QButtonGroup(self)
        self._roi_mode_grp.addButton(self._rect_radio,  0)
        self._roi_mode_grp.addButton(self._grain_radio, 1)
        self._roi_mode_grp.idToggled.connect(self._on_roi_mode_changed)

        mode_w = QWidget(); mh = QHBoxLayout(mode_w); mh.setContentsMargins(0, 0, 0, 0)
        mh.addWidget(self._rect_radio)
        mh.addWidget(self._grain_radio)
        mh.addStretch()
        roi_layout.addRow("ROI mode:", mode_w)

        # Grain selector (hidden until segmentation done)
        self._grain_roi_combo = QComboBox()
        self._grain_roi_combo.setVisible(False)
        self._grain_roi_combo.currentIndexChanged.connect(self._on_grain_roi_changed)
        roi_layout.addRow("Grain:", self._grain_roi_combo)

        self.roi_row_start = QSpinBox(); self.roi_row_start.setRange(0, 9999); self.roi_row_start.setValue(0)
        self.roi_row_stop  = QSpinBox(); self.roi_row_stop.setRange(1, 10000); self.roi_row_stop.setValue(10)
        self.roi_col_start = QSpinBox(); self.roi_col_start.setRange(0, 9999); self.roi_col_start.setValue(0)
        self.roi_col_stop  = QSpinBox(); self.roi_col_stop.setRange(1, 10000); self.roi_col_stop.setValue(10)

        for sb in (self.roi_row_start, self.roi_row_stop,
                   self.roi_col_start, self.roi_col_stop):
            sb.valueChanged.connect(self._update_roi_rects)

        self._roi_row_w = QWidget(); rh = QHBoxLayout(self._roi_row_w); rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(QLabel("start")); rh.addWidget(self.roi_row_start)
        rh.addWidget(QLabel("  stop")); rh.addWidget(self.roi_row_stop); rh.addStretch()

        self._roi_col_w = QWidget(); ch = QHBoxLayout(self._roi_col_w); ch.setContentsMargins(0, 0, 0, 0)
        ch.addWidget(QLabel("start")); ch.addWidget(self.roi_col_start)
        ch.addWidget(QLabel("  stop")); ch.addWidget(self.roi_col_stop); ch.addStretch()

        self._roi_note = _note("Indices are 0-based — top-left corner is (row 0, col 0).")

        roi_layout.addRow("Rows  (y):", self._roi_row_w)
        roi_layout.addRow("Columns  (x):", self._roi_col_w)
        roi_layout.addRow(self._roi_note)
        roi_group.setLayout(roi_layout)

        # ── IPF direction group (standalone, sits between top row and maps) ────
        ipf_dir_group  = QGroupBox("Select IPF Direction")
        ipf_dir_layout = QFormLayout()
        ipf_dir_layout.addRow("IPF direction:", dir_w)
        ipf_dir_group.setLayout(ipf_dir_layout)

        # ── KAM map button (in its own group, sibling of IPF-direction) ──────
        kam_group = QGroupBox("KAM Map")
        kam_layout = QVBoxLayout()
        self._kam_btn = QPushButton("Show KAM Map…")
        self._kam_btn.setToolTip(
            "Open a dialog with the per-pixel kernel-average misorientation "
            "(KAM) computed during segmentation."
        )
        self._kam_btn.clicked.connect(self._show_kam_map)
        self._kam_btn.setEnabled(False)   # enabled after segmentation finishes
        kam_layout.addWidget(self._kam_btn)
        kam_group.setLayout(kam_layout)

        # ── Map display (IPF left, grain map right) ───────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        ipf_panel  = QWidget()
        ipf_vbox   = QVBoxLayout(ipf_panel)
        ipf_vbox.setContentsMargins(0, 0, 4, 0)

        self._ipf_fig = Figure()
        self._ipf_fig.subplots_adjust(left=0.04, right=0.70, top=0.93, bottom=0.04)
        self._ipf_ax  = self._ipf_fig.add_subplot(111)
        self._ipf_ax.set_visible(False)
        self._key_ax  = self._ipf_fig.add_axes([0.73, 0.10, 0.24, 0.78])
        self._key_ax.set_visible(False)
        self._ipf_canvas = FigureCanvas(self._ipf_fig)
        self._ipf_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ipf_status = QLabel("Computing IPF map…")
        self._ipf_status.setStyleSheet("color: gray;")
        self._ipf_status.setWordWrap(True)

        ipf_vbox.addWidget(QLabel("IPF Map"))
        ipf_vbox.addWidget(self._ipf_canvas)
        ipf_vbox.addWidget(self._ipf_status)

        _bg = THEME["surface_bg"]
        self._grain_dialog = QDialog(self)
        self._grain_dialog.setWindowTitle("Grain ID Map")
        self._grain_dialog.resize(700, 600)
        grain_vbox = QVBoxLayout(self._grain_dialog)
        grain_vbox.setContentsMargins(4, 4, 4, 4)

        self._grain_fig = Figure(tight_layout=True, facecolor=_bg)
        self._grain_ax  = self._grain_fig.add_subplot(111)
        self._grain_ax.set_facecolor(_bg)
        self._grain_ax.set_visible(False)
        self._grain_canvas = FigureCanvas(self._grain_fig)
        self._grain_canvas.setStyleSheet(f"background-color: {_bg};")
        self._grain_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._grain_status = QLabel("")
        self._grain_status.setStyleSheet("color: gray;")
        self._grain_status.setWordWrap(True)

        grain_vbox.addWidget(self._grain_canvas)
        grain_vbox.addWidget(self._grain_status)

        self._splitter.addWidget(ipf_panel)

        # ── Assemble page ─────────────────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.addWidget(roi_group)
        top_row.addWidget(seg_group)

        outer = QVBoxLayout()
        outer.addLayout(top_row)
        ipf_dir_row = QHBoxLayout()
        ipf_dir_row.addWidget(ipf_dir_group, stretch=1)
        ipf_dir_row.addWidget(kam_group)
        outer.addLayout(ipf_dir_row)
        outer.addWidget(self._splitter, stretch=1)
        self.setLayout(outer)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initializePage(self):
        wiz      = self.wizard()
        ang_path = wiz.field("ang_path")
        geom     = wiz.geometry_page.get_params()
        rows, cols = geom["rows"], geom["cols"]
        patshape = (geom["pat_h"], geom["pat_w"])

        self.roi_row_start.setMaximum(max(0, rows - 1))
        self.roi_row_stop.setMaximum(rows);  self.roi_row_stop.setValue(rows)
        self.roi_col_start.setMaximum(max(0, cols - 1))
        self.roi_col_stop.setMaximum(cols);  self.roi_col_stop.setValue(cols)

        if not ang_path or not os.path.exists(ang_path):
            self._ipf_status.setText("No ANG file found")
            return

        self._ipf_status.setText("Computing IPF map… (this may take a few seconds)")
        direction = self._DIRECTIONS[self._dir_combo.currentText()]
        cached = wiz.ang_data if wiz.ang_data is not None and wiz.ang_loaded_path == ang_path else None
        self._ipf_worker = IPFWorker(ang_path, patshape, direction, ang_data=cached)
        self._ipf_worker.done_signal.connect(self._on_ipf_done)
        self._ipf_worker.start()

    # ── Grain segmentation ────────────────────────────────────────────────────

    def _start_segmentation(self):
        wiz      = self.wizard()
        ang_path = wiz.field("ang_path")
        geom     = wiz.geometry_page.get_params()
        patshape = (geom["pat_h"], geom["pat_w"])

        if not ang_path or not os.path.exists(ang_path):
            self._seg_status.setText("No ANG file found.")
            return

        self._seg_btn.setEnabled(False)
        self._seg_status.setText("Segmenting grains… (may take a while for large scans)")

        cached = wiz.ang_data if wiz.ang_data is not None and wiz.ang_loaded_path == ang_path else None
        self._seg_worker = SegmentWorker(
            ang_path, patshape,
            self.seg_threshold.value(),
            self.min_grain_size.value(),
            ang_data=cached,
        )
        self._seg_worker.done_signal.connect(self._on_seg_done)
        self._seg_worker.log_signal.connect(self._seg_status.setText)
        self._seg_worker.start()

    def _show_kam_map(self):
        """Open a dialog displaying the KAM map from the latest segmentation."""
        if self._kam is None:
            QMessageBox.information(self, "No KAM",
                                    "Run segmentation first to produce a KAM map.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Kernel Average Misorientation (KAM)")
        dlg.resize(780, 640)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        _bg = THEME["surface_bg"]
        fig = Figure(facecolor=_bg, tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(_bg)
        finite = self._kam[np.isfinite(self._kam)]
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        im = ax.imshow(self._kam, cmap="inferno", origin="upper", vmin=0, vmax=vmax)
        ax.set_title(
            f"KAM (deg)  —  max shown = {vmax:.3f}°",
            color="white", fontweight="bold",
        )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white")
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f"background-color: {_bg};")
        layout = QVBoxLayout(dlg)
        layout.addWidget(canvas)
        dlg.show()
        # Keep a reference so the dialog isn't garbage-collected immediately.
        self._kam_dialog = dlg

    def _on_seg_done(self, grain_ids, kam, error: str):
        self._seg_btn.setEnabled(True)
        if grain_ids is None:
            self._seg_status.setText(f"Error: {error}")
            return

        self._grain_ids = grain_ids
        self._kam       = kam
        self._kam_btn.setEnabled(True)
        # Count grains that survived the min-size filter (label > 0)
        sizes    = np.bincount(grain_ids.ravel())          # index 0 = grain-0 pixels
        n_grains = int(np.count_nonzero(sizes[1:]))        # valid grains only
        n_small  = int(grain_ids.max()) - n_grains         # discarded as too small

        self._grain_ax.set_visible(True)
        self._grain_ax.clear()

        import matplotlib.pyplot as _plt
        import matplotlib.colors as _mcolors
        import matplotlib.patches as mpatches

        # Surviving grain IDs (non-zero, non-discarded) — may be non-contiguous
        surviving_ids = [gid for gid in range(1, len(sizes)) if sizes[gid] > 0]
        n_surviving   = len(surviving_ids)

        # Remap surviving IDs to compact indices 1..n so colors are well-defined.
        # Discarded grains (sizes[gid]==0) stay as 0 after remapping → black.
        remap = np.zeros(len(sizes), dtype=np.int32)
        for compact_idx, gid in enumerate(surviving_ids, start=1):
            remap[gid] = compact_idx
        remapped = remap[grain_ids]   # (rows, cols), 0 = grain-0/discarded
        self._grain_remapped = remapped
        self._grain_remap    = remap

        # Build a ListedColormap with one color per grain.  Uses HSV with
        # golden-ratio hue stepping so adjacent grain IDs always get
        # noticeably different hues (φ⁻¹ guarantees the hue sequence never
        # clusters), and modulates saturation / value slightly so two grains
        # that happen to land near the same hue still differ in brightness.
        import colorsys
        GOLDEN = 0.6180339887498949
        discrete_colors = []
        for i in range(n_surviving):
            h = (i * GOLDEN) % 1.0
            s = 0.75 if (i % 2 == 0) else 0.95
            v = 0.95 if (i % 3 != 0) else 0.75
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            discrete_colors.append((r, g, b, 1.0))
        grain_lut  = _mcolors.ListedColormap(discrete_colors, name="grain_discrete")
        grain_lut.set_bad(color="black")
        grain_norm = _mcolors.BoundaryNorm(
            np.arange(0.5, n_surviving + 1.5), ncolors=n_surviving
        )
        self._grain_lut           = grain_lut
        self._grain_norm          = grain_norm
        self._grain_discrete_colors = discrete_colors

        _masked_remap = np.ma.masked_where(remapped == 0, remapped)
        self._grain_ax.imshow(
            _masked_remap, cmap=grain_lut, norm=grain_norm,
            interpolation="nearest", origin="upper",
        )
        title = f"Grain IDs  ({n_surviving} grains"
        if n_small:
            title += f", {n_small} discarded)"
        else:
            title += ")"
        self._grain_ax.set_title(title, fontsize=9)
        self._grain_ax.axis("off")


        # ── Grain legend ──────────────────────────────────────────────────────
        # Colors derived from the same LUT so they are guaranteed to match.
        # compact_idx is 1-based → grain_lut maps 1→color[0], 2→color[1], …
        legend_entries = [
            (gid, int(sizes[gid]), discrete_colors[(ci - 1) % n_surviving])
            for ci, gid in enumerate(surviving_ids, start=1)
        ]

        MAX_LEGEND = 30
        if len(legend_entries) > MAX_LEGEND:
            legend_entries.sort(key=lambda x: -x[1])
            legend_entries = legend_entries[:MAX_LEGEND]
            leg_title = f"Largest {MAX_LEGEND} grains"
        else:
            leg_title = "Grains"

        legend_patches = [
            mpatches.Patch(facecolor=col, edgecolor="white", linewidth=0.3,
                           label=f"G{gid}  ({sz} px)")
            for gid, sz, col in legend_entries
        ]
        # Cap at 4 columns so the legend never gets wider than the plot.
        ncols = max(1, min(4, (len(legend_patches) + 7) // 8))
        self._grain_ax.legend(
            handles=legend_patches,
            title=leg_title,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            fontsize=6,
            title_fontsize=7,
            ncol=ncols,
            framealpha=0.85,
            handlelength=1.2,
            handleheight=0.9,
        )

        self._grain_fig.tight_layout(pad=0.5)

        if self._rgb_map is not None:
            masked_rgb = self._rgb_map.copy()
            masked_rgb[grain_ids == 0] = 0.0
            self._ipf_ax.clear()
            self._ipf_ax.imshow(masked_rgb, origin="upper", interpolation="nearest")
            label = self._dir_combo.currentText().split()[0]
            self._ipf_ax.set_title(f"IPF Map  //  {label}", fontsize=9)
            self._ipf_ax.axis("off")
            self._roi_rects[0] = None
            self._ipf_canvas.draw()

        self._grain_dialog.show()

        self._roi_rects[1] = None
        self._update_roi_rects()

        self._grain_canvas.draw()
        self._grain_status.setText(
            f"{n_grains} grains  (threshold = {self.seg_threshold.value():.1f}°, "
            f"min size = {self.min_grain_size.value()})"
        )
        discard_note = f"  ({n_small} discarded — shown black)" if n_small else ""
        self._seg_status.setText(f"Done — {n_grains} grains segmented.{discard_note}")
        self._populate_grain_combo()

    # ── IPF map ───────────────────────────────────────────────────────────────

    def _recompute_ipf(self):
        wiz      = self.wizard()
        ang_path = wiz.field("ang_path")
        geom     = wiz.geometry_page.get_params()
        patshape = (geom["pat_h"], geom["pat_w"])
        if not ang_path or not os.path.exists(ang_path):
            return
        self._ipf_status.setText("Recomputing…")
        direction = self._DIRECTIONS[self._dir_combo.currentText()]
        cached = wiz.ang_data if wiz.ang_data is not None and wiz.ang_loaded_path == ang_path else None
        self._ipf_worker = IPFWorker(ang_path, patshape, direction, ang_data=cached)
        self._ipf_worker.done_signal.connect(self._on_ipf_done)
        self._ipf_worker.start()

    def _on_ipf_done(self, rgb_map, error: str):
        if rgb_map is None:
            self._ipf_status.setText(f"Error computing IPF map: {error}")
            return

        self._rgb_map = rgb_map
        label = self._dir_combo.currentText().split()[0]

        self._ipf_ax.set_visible(True)
        self._ipf_ax.clear()
        self._ipf_ax.imshow(rgb_map, origin="upper", interpolation="nearest")
        self._ipf_ax.set_title(f"IPF Map  //  {label}", fontsize=9)
        #self._ipf_ax.set_xlabel("Column", fontsize=8)
        #self._ipf_ax.set_ylabel("Row", fontsize=8)
        #self._ipf_ax.tick_params(labelsize=7)

        from ipf_map import plot_ipf_triangle
        self._key_ax.set_visible(True)
        self._key_ax.clear()
        plot_ipf_triangle(self._key_ax, n=150)

        self._roi_rects[0] = None
        self._update_roi_rects()

        self._ipf_canvas.draw()

        rows, cols, _ = rgb_map.shape
        self._ipf_status.setText(
            f"IPF map  ({rows} × {cols} patterns, direction = {label})"
        )

    # ── Grain ROI helpers ─────────────────────────────────────────────────────

    def _on_roi_mode_changed(self, btn_id: int, checked: bool):
        if not checked:
            return
        is_grain = (btn_id == 1)
        self._grain_roi_combo.setVisible(is_grain)
        show_rect = not is_grain
        for w in (self._roi_row_w, self._roi_col_w, self._roi_note):
            w.setVisible(show_rect)
            layout = self._roi_form_layout
            row, _ = layout.getWidgetPosition(w)
            if row >= 0:
                label_item = layout.itemAt(row, layout.ItemRole.LabelRole)
                if label_item and label_item.widget():
                    label_item.widget().setVisible(show_rect)
        if is_grain and self._grain_ids is not None:
            self._on_grain_roi_changed(self._grain_roi_combo.currentIndex())

    def _populate_grain_combo(self):
        """Rebuild the grain dropdown after segmentation finishes."""
        self._grain_roi_combo.blockSignals(True)
        self._grain_roi_combo.clear()
        if self._grain_ids is not None:
            sizes = np.bincount(self._grain_ids.ravel())
            for gid in range(1, len(sizes)):
                if sizes[gid] > 0:
                    self._grain_roi_combo.addItem(
                        f"Grain {gid}  ({sizes[gid]} px)", userData=gid
                    )
        self._grain_roi_combo.blockSignals(False)
        has_grains = self._grain_roi_combo.count() > 0
        self._grain_radio.setEnabled(has_grains)
        if self._grain_radio.isChecked() and has_grains:
            self._on_grain_roi_changed(self._grain_roi_combo.currentIndex())

    def _on_grain_roi_changed(self, index: int):
        if index < 0 or self._grain_ids is None:
            return
        grain_id = self._grain_roi_combo.itemData(index)
        if grain_id is not None:
            self._apply_grain_roi(grain_id)

    def _apply_grain_roi(self, grain_id: int):
        """Set ROI spinboxes to the bounding box of the given grain."""
        rows_idx, cols_idx = np.where(self._grain_ids == grain_id)
        if rows_idx.size == 0:
            return
        for sb in (self.roi_row_start, self.roi_row_stop,
                   self.roi_col_start, self.roi_col_stop):
            sb.blockSignals(True)
        self.roi_row_start.setValue(int(rows_idx.min()))
        self.roi_row_stop.setValue(int(rows_idx.max()) + 1)
        self.roi_col_start.setValue(int(cols_idx.min()))
        self.roi_col_stop.setValue(int(cols_idx.max()) + 1)
        for sb in (self.roi_row_start, self.roi_row_stop,
                   self.roi_col_start, self.roi_col_stop):
            sb.blockSignals(False)
        self._update_roi_rects()

        # Highlight the selected grain on the grain map (others → white)
        if self._grain_remapped is not None and self._grain_discrete_colors is not None:
            import matplotlib.colors as _mcolors
            highlight_lut = _mcolors.ListedColormap(self._grain_discrete_colors, name="grain_highlight")
            highlight_lut.set_bad(color="white")
            masked = np.ma.masked_where(self._grain_ids != grain_id, self._grain_remapped)
            self._grain_ax.clear()
            self._grain_ax.set_facecolor("white")
            self._grain_ax.imshow(
                masked, cmap=highlight_lut, norm=self._grain_norm,
                interpolation="nearest", origin="upper",
            )
            self._grain_ax.set_title(f"Grain {grain_id}  (selected)", fontsize=9)
            self._grain_ax.axis("off")
            self._grain_canvas.draw()

    # ── ROI rectangle ─────────────────────────────────────────────────────────

    def _update_roi_rects(self):
        import matplotlib.patches as mpatches

        axes_canvas = [
            (self._ipf_ax,   self._ipf_canvas,   0),
            (self._grain_ax, self._grain_canvas,  1),
        ]
        for ax, canvas, idx in axes_canvas:
            if not ax.get_visible():
                continue

            if self._roi_rects[idx] is not None:
                try:
                    self._roi_rects[idx].remove()
                except Exception:
                    pass
                self._roi_rects[idx] = None

            r0 = self.roi_row_start.value()
            r1 = self.roi_row_stop.value()
            c0 = self.roi_col_start.value()
            c1 = self.roi_col_stop.value()

            rect = mpatches.Rectangle(
                (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                linewidth=2, edgecolor="black", facecolor="none",
            )
            ax.add_patch(rect)
            self._roi_rects[idx] = rect
            canvas.draw_idle()

    # ── Params ────────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        result = {
            "roi_slice": [
                slice(self.roi_row_start.value(), self.roi_row_stop.value()),
                slice(self.roi_col_start.value(), self.roi_col_stop.value()),
            ]
        }
        if self._grain_radio.isChecked():
            gid = self._grain_roi_combo.currentData()
            if gid is not None:
                result["_roi_grain_id"] = gid
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Reference Pattern Selection
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Simulated-pattern live tuner dialog
# ─────────────────────────────────────────────────────────────────────────────

class SimTunerDialog(QDialog):
    """
    Interactive dialog for live adjustment of Euler angles and pattern center.

    Changing any spinbox triggers a 600 ms debounce timer; when it fires a new
    SimRefWorker is launched and the canvas updates automatically.

    applied_signal emits (euler_deg: tuple, pc_edax: tuple) when the user
    clicks "Apply to Reference".
    """
    applied_signal = pyqtSignal(tuple, tuple, float)  # euler_deg, pc_edax, sample_tilt_deg

    def __init__(self, master_path: str, euler_deg: tuple, pc_edax: tuple,
                 det_shape: tuple, det_tilt_deg: float, sample_tilt_deg: float = 70.0,
                 exp_pat: np.ndarray = None, exp_pat_proc: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulated Pattern Live Tuner")
        self.setMinimumSize(960, 620)
        self._master_path     = master_path
        self._det_shape       = det_shape
        self._det_tilt_deg    = det_tilt_deg     # held fixed at the Step 2 value
        self._sample_tilt_deg = sample_tilt_deg  # held fixed at the Step 2 value
        self._exp_pat_raw     = exp_pat       # unprocessed (may be None)
        self._exp_pat_proc    = exp_pat_proc  # processed   (may be None)
        # Default to the processed experimental pattern when available so the
        # checkerboard reflects the same preprocessing pipeline the optimizer
        # will see at run time (Step 3 hp / lp / γ / mask / flip).
        self._exp_pat         = exp_pat_proc if exp_pat_proc is not None else exp_pat
        self._worker          = None

        # ── Debounce timer ────────────────────────────────────────────────────
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(600)   # ms — increase if generation is slow
        self._timer.timeout.connect(self._generate)

        # ── Controls (left column) ────────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        # Euler angles
        eu_group  = QGroupBox("Euler Angles (Bunge ZXZ, degrees)")
        eu_layout = QFormLayout()

        eu_step_combo = QComboBox()
        eu_step_combo.addItems(["0.01", "0.1", "1.0", "5.0", "10.0"])
        eu_step_combo.setCurrentText("1.0")
        eu_layout.addRow("Arrow-key step (°):", eu_step_combo)

        self._phi1 = QDoubleSpinBox(); self._phi1.setRange(0, 360); self._phi1.setDecimals(3); self._phi1.setSuffix(" °"); self._phi1.setValue(euler_deg[0]); self._phi1.setSingleStep(1.0)
        self._Phi  = QDoubleSpinBox(); self._Phi.setRange(0, 180);  self._Phi.setDecimals(3);  self._Phi.setSuffix(" °");  self._Phi.setValue(euler_deg[1]);  self._Phi.setSingleStep(1.0)
        self._phi2 = QDoubleSpinBox(); self._phi2.setRange(0, 360); self._phi2.setDecimals(3); self._phi2.setSuffix(" °"); self._phi2.setValue(euler_deg[2]); self._phi2.setSingleStep(1.0)
        # Make the spinboxes (and their +/- arrows) larger for finger-friendly tuning.
        _BIG_SPIN_QSS = (
            "QDoubleSpinBox { font-size: 16px; min-height: 36px; padding: 2px 6px; }"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 28px; }"
        )
        for _w in (self._phi1, self._Phi, self._phi2):
            _w.setStyleSheet(_BIG_SPIN_QSS)
        eu_layout.addRow("φ₁ (phi1):", self._phi1)
        eu_layout.addRow("Φ  (Phi):",  self._Phi)
        eu_layout.addRow("φ₂ (phi2):", self._phi2)
        eu_group.setLayout(eu_layout)

        eu_step_combo.currentTextChanged.connect(
            lambda t: [w.setSingleStep(float(t)) for w in (self._phi1, self._Phi, self._phi2)]
        )

        # Pattern center
        pc_group  = QGroupBox("Pattern Center (EDAX/TSL convention)")
        pc_layout = QFormLayout()

        pc_step_combo = QComboBox()
        pc_step_combo.addItems(["0.0001", "0.001", "0.005", "0.01"])
        pc_step_combo.setCurrentText("0.001")
        pc_layout.addRow("Arrow-key step:", pc_step_combo)

        self._pcx = QDoubleSpinBox(); self._pcx.setRange(0, 2); self._pcx.setDecimals(5); self._pcx.setValue(pc_edax[0]); self._pcx.setSingleStep(0.001)
        self._pcy = QDoubleSpinBox(); self._pcy.setRange(0, 2); self._pcy.setDecimals(5); self._pcy.setValue(pc_edax[1]); self._pcy.setSingleStep(0.001)
        self._pcz = QDoubleSpinBox(); self._pcz.setRange(0, 2); self._pcz.setDecimals(5); self._pcz.setValue(pc_edax[2]); self._pcz.setSingleStep(0.001)
        for _w in (self._pcx, self._pcy, self._pcz):
            _w.setStyleSheet(_BIG_SPIN_QSS)
        pc_layout.addRow("x*:", self._pcx)
        pc_layout.addRow("y*:", self._pcy)
        pc_layout.addRow("z*:", self._pcz)
        pc_group.setLayout(pc_layout)

        pc_step_combo.currentTextChanged.connect(
            lambda t: [w.setSingleStep(float(t)) for w in (self._pcx, self._pcy, self._pcz)]
        )

        # Detector geometry (detector size, detector tilt, sample tilt) is
        # set on Step 2 and is not tunable here.

        # Wire all spinboxes to debounce
        for w in (self._phi1, self._Phi, self._phi2,
                  self._pcx, self._pcy, self._pcz):
            w.valueChanged.connect(self._on_value_changed)

        # ── View mode: Checkerboard vs Flicker ────────────────────────────────
        view_group  = QGroupBox("View Mode")
        view_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        self._view_cb_radio   = QRadioButton("Checkerboard")
        self._view_flick_radio = QRadioButton("Flicker")
        self._view_cb_radio.setChecked(True)
        mode_row.addWidget(self._view_cb_radio)
        mode_row.addWidget(self._view_flick_radio)
        mode_row.addStretch()
        view_layout.addLayout(mode_row)
        self._view_mode_grp = QButtonGroup(self)
        self._view_mode_grp.addButton(self._view_cb_radio, 0)
        self._view_mode_grp.addButton(self._view_flick_radio, 1)
        self._view_mode_grp.idToggled.connect(self._on_view_mode_changed)

        # Checkerboard sub-controls (shown only in checkerboard mode)
        self._cb_sub = QWidget()
        cb_sub_layout = QFormLayout(self._cb_sub)
        cb_sub_layout.setContentsMargins(0, 0, 0, 0)
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(4, 128)
        self._tile_spin.setValue(40)
        self._tile_spin.setSuffix(" px")
        self._tile_spin.setToolTip("Size of each checkerboard tile in pixels.")
        self._tile_spin.valueChanged.connect(self._on_value_changed)
        cb_sub_layout.addRow("Tile size:", self._tile_spin)
        view_layout.addWidget(self._cb_sub)

        # Flicker sub-controls (shown only in flicker mode)
        self._flick_sub = QWidget()
        flick_sub_layout = QHBoxLayout(self._flick_sub)
        flick_sub_layout.setContentsMargins(0, 0, 0, 0)
        self._flick_play_btn = QPushButton("▶ Play")
        self._flick_play_btn.setCheckable(True)
        self._flick_play_btn.toggled.connect(self._on_flick_toggle)
        flick_sub_layout.addWidget(self._flick_play_btn)
        flick_sub_layout.addWidget(QLabel("Speed:"))
        self._flick_speed = QSlider(Qt.Orientation.Horizontal)
        self._flick_speed.setMinimum(50)
        self._flick_speed.setMaximum(1000)
        self._flick_speed.setValue(250)
        self._flick_speed.setSingleStep(25)
        self._flick_speed.valueChanged.connect(self._on_flick_speed_changed)
        flick_sub_layout.addWidget(self._flick_speed, stretch=1)
        self._flick_speed_lbl = QLabel("250 ms")
        flick_sub_layout.addWidget(self._flick_speed_lbl)
        view_layout.addWidget(self._flick_sub)
        self._flick_sub.setVisible(False)

        # Experimental-processing toggle (applies to both view modes)
        self._proc_chk = QCheckBox("Apply processing to experimental")
        self._proc_chk.setChecked(exp_pat_proc is not None)
        self._proc_chk.setEnabled(exp_pat_proc is not None)
        self._proc_chk.setToolTip(
            "Use the processed (filtered) experimental pattern in the view."
            if exp_pat_proc is not None
            else "Not available — generate the pattern first so processing params are applied."
        )
        self._proc_chk.toggled.connect(self._on_proc_toggled)
        view_layout.addWidget(self._proc_chk)

        view_group.setLayout(view_layout)

        # Flicker timer + state — needed before _on_done can render flicker.
        self._flick_show_sim = True
        self._flick_im       = None
        self._flick_title    = None
        self._flick_timer    = QTimer(self)
        self._flick_timer.timeout.connect(self._on_flick_tick)
        self._last_sim_pat   = None  # cache so the view-mode toggle can redraw

        # ── Stacked control layout ────────────────────────────────────────────
        # All controls in a single column: Euler, PC, view-mode.  Detector
        # geometry was removed — those values are fixed by Step 2.
        ctrl.addWidget(eu_group)
        ctrl.addWidget(pc_group)
        ctrl.addWidget(view_group)

        # Status + buttons
        self._status = QLabel("Generating initial pattern…")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {THEME['warning']}; font-size: 11px;")
        ctrl.addWidget(self._status)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Finish && Apply")
        apply_btn.setToolTip("Push current Euler angles and PC back to the reference page and close the tuner.")
        apply_btn.clicked.connect(self._apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Discard any changes made in the tuner and close the window.")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        ctrl.addLayout(btn_row)
        ctrl.addStretch()

        # ── Canvas (right): checkerboard only ─────────────────────────────────
        # Single axis so the checkerboard has the entire right-hand area.
        # Current Euler / PC values are encoded in the title.
        self._fig = Figure(figsize=(8, 8))
        self._fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
        self._ax_overlay = self._fig.add_subplot(111)
        self._ax_overlay.axis("off")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumSize(480, 480)

        # ── Assemble ──────────────────────────────────────────────────────────
        outer = QHBoxLayout()
        outer.setSpacing(12)
        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl)
        ctrl_w.setFixedWidth(460)
        outer.addWidget(ctrl_w)
        outer.addWidget(self._canvas, stretch=1)
        self.setLayout(outer)

        # Fire first generation immediately
        self._generate()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_value_changed(self):
        self._timer.start()   # restart debounce each time a spinbox changes

    def _on_proc_toggled(self, checked: bool):
        self._exp_pat = self._exp_pat_proc if checked else self._exp_pat_raw
        # Redraw checkerboard with the newly active experimental pattern.
        self._timer.start()

    def _generate(self):
        # Silently drop the previous worker's result if it's still running
        if self._worker and self._worker.isRunning():
            try:
                self._worker.done_signal.disconnect()
                self._worker.error_signal.disconnect()
            except RuntimeError:
                pass

        euler_deg = (self._phi1.value(), self._Phi.value(), self._phi2.value())
        pc_edax   = (self._pcx.value(), self._pcy.value(), self._pcz.value())

        self._status.setText("Generating…")
        self._status.setStyleSheet(f"color: {THEME['warning']}; font-size: 11px;")

        self._worker = SimRefWorker(
            self._master_path, euler_deg, pc_edax,
            self._det_shape,
            det_tilt_deg    = self._det_tilt_deg,
            sample_tilt_deg = self._sample_tilt_deg,
        )
        self._worker.done_signal.connect(self._on_done)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_done(self, pat_np: np.ndarray):
        # Cache the latest sim so view-mode toggling and flicker ticks can
        # redraw without re-running the worker.
        self._last_sim_pat = pat_np
        self._redraw_view()
        self._status.setText(f"Done — {pat_np.shape[0]}×{pat_np.shape[1]} px.")
        self._status.setStyleSheet(f"color: {THEME['success']}; font-size: 11px;")

    def _redraw_view(self):
        """Dispatch to whichever view mode is selected (checkerboard /
        flicker), using the latest cached sim pattern."""
        if self._view_flick_radio.isChecked():
            self._draw_flicker_frame(force_resize=True)
        else:
            self._draw_checkerboard()

    def _draw_checkerboard(self):
        eu = (self._phi1.value(), self._Phi.value(), self._phi2.value())
        pc = (self._pcx.value(), self._pcy.value(), self._pcz.value())
        pat_np = self._last_sim_pat
        # Reset cached flicker handles so the next flicker pass starts fresh.
        self._flick_im = None
        self._flick_title = None

        self._ax_overlay.cla()
        if pat_np is not None and self._exp_pat is not None:
            exp_r = self._exp_pat
            sim_r = pat_np
            if exp_r.shape != sim_r.shape:
                from PIL import Image as _PIL
                sim_r = np.array(
                    _PIL.fromarray((sim_r * 255).astype(np.uint8)).resize(
                        (exp_r.shape[1], exp_r.shape[0]), resample=_PIL.BILINEAR
                    )
                ).astype(np.float32) / 255.0
            t = self._tile_spin.value()
            h, w = exp_r.shape
            rows = np.arange(h)[:, None]
            cols = np.arange(w)[None, :]
            mask = ((rows // t) + (cols // t)) % 2 == 0
            checkerboard = np.where(mask, exp_r, sim_r)
            self._ax_overlay.imshow(checkerboard, cmap="gray", vmin=0, vmax=1, origin="upper")
            self._ax_overlay.set_title(
                f"Checkerboard  ({t} px tiles, exp / sim)\n"
                f"φ₁={eu[0]:.2f}°  Φ={eu[1]:.2f}°  φ₂={eu[2]:.2f}°  |  "
                f"PC=({pc[0]:.4f}, {pc[1]:.4f}, {pc[2]:.4f})",
                fontsize=10,
            )
        else:
            self._ax_overlay.text(0.5, 0.5, "No experimental\npattern", ha="center",
                                  va="center", transform=self._ax_overlay.transAxes,
                                  fontsize=10, color="gray")
            self._ax_overlay.set_title("Checkerboard", fontsize=10)
        self._ax_overlay.axis("off")
        self._canvas.draw()

    def _draw_flicker_frame(self, force_resize: bool = False):
        """Render one frame of the flicker view onto the shared canvas.
        Mirrors the comparison-dialog flicker, but on the tuner's single
        axis — reuses one AxesImage handle for cheap per-tick updates."""
        if self._last_sim_pat is None or self._exp_pat is None:
            self._ax_overlay.cla()
            self._ax_overlay.text(0.5, 0.5, "No experimental\npattern", ha="center",
                                  va="center", transform=self._ax_overlay.transAxes,
                                  fontsize=10, color="gray")
            self._ax_overlay.axis("off")
            self._canvas.draw()
            return

        sim_r = self._last_sim_pat
        exp_r = self._exp_pat
        if exp_r.shape != sim_r.shape:
            from PIL import Image as _PIL
            sim_r = np.array(
                _PIL.fromarray((sim_r * 255).astype(np.uint8)).resize(
                    (exp_r.shape[1], exp_r.shape[0]), resample=_PIL.BILINEAR
                )
            ).astype(np.float32) / 255.0

        img   = sim_r if self._flick_show_sim else exp_r
        label = "SIMULATED" if self._flick_show_sim else "EXPERIMENTAL"
        color = "#4caf50" if self._flick_show_sim else "#ff9800"

        if self._flick_im is None or force_resize:
            self._ax_overlay.cla()
            self._flick_im = self._ax_overlay.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
            self._ax_overlay.axis("off")
            self._flick_title = self._ax_overlay.set_title(label, fontsize=14, color=color, fontweight="bold")
        else:
            self._flick_im.set_data(img)
            self._flick_title.set_text(label)
            self._flick_title.set_color(color)
        self._canvas.draw_idle()

    def _on_view_mode_changed(self, btn_id: int, checked: bool):
        if not checked:
            return
        flicker = (btn_id == 1)
        self._cb_sub.setVisible(not flicker)
        self._flick_sub.setVisible(flicker)
        if not flicker and self._flick_timer.isActive():
            self._flick_timer.stop()
            self._flick_play_btn.setChecked(False)
        self._redraw_view()

    def _on_flick_toggle(self, checked: bool):
        if checked and self._last_sim_pat is not None and self._exp_pat is not None:
            self._flick_timer.start(self._flick_speed.value())
            self._flick_play_btn.setText("⏸ Pause")
        else:
            self._flick_timer.stop()
            self._flick_play_btn.setText("▶ Play")

    def _on_flick_speed_changed(self, value: int):
        self._flick_speed_lbl.setText(f"{value} ms")
        if self._flick_timer.isActive():
            self._flick_timer.start(int(value))

    def _on_flick_tick(self):
        self._flick_show_sim = not self._flick_show_sim
        self._draw_flicker_frame(force_resize=False)

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg.splitlines()[-1]}")
        self._status.setStyleSheet(f"color: {THEME['error']}; font-size: 11px;")

    def _apply(self):
        euler_deg = (self._phi1.value(), self._Phi.value(), self._phi2.value())
        pc_edax   = (self._pcx.value(), self._pcy.value(), self._pcz.value())
        self.applied_signal.emit(euler_deg, pc_edax, float(self._sample_tilt_deg))
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reusable iOS-style toggle switch.
# ─────────────────────────────────────────────────────────────────────────────

class _ToggleSwitch(QWidget):
    """Two-position sliding toggle with optional labels on either side.

    Emits ``toggled(bool)`` when the user clicks (or clicks one of the
    labels).  The left position is ``False``; right is ``True``.  Clicking
    either label, the track, or the thumb flips the state.
    """
    toggled = pyqtSignal(bool)

    def __init__(self, left_label: str = "Off", right_label: str = "On",
                 parent=None):
        super().__init__(parent)
        self._left_label  = left_label
        self._right_label = right_label
        self._checked     = False
        self._thumb_pos   = 0.0   # 0.0 = full left, 1.0 = full right
        self._anim = QPropertyAnimation(self, b"thumb_pos", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)

    # ── Qt property used by the animation ──────────────────────────────────
    def get_thumb_pos(self) -> float:
        return self._thumb_pos
    def set_thumb_pos(self, v: float):
        self._thumb_pos = float(v)
        self.update()
    thumb_pos = pyqtProperty(float, fget=get_thumb_pos, fset=set_thumb_pos)

    # ── Public API ─────────────────────────────────────────────────────────
    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, on: bool, emit_signal: bool = True):
        on = bool(on)
        if on == self._checked:
            return
        self._checked = on
        self._anim.stop()
        self._anim.setStartValue(self._thumb_pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()
        if emit_signal:
            self.toggled.emit(on)

    def toggle(self):
        self.setChecked(not self._checked)

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        fm = self.fontMetrics()
        return QSize(
            fm.horizontalAdvance(self._left_label)
            + fm.horizontalAdvance(self._right_label)
            + 60,   # track + side gaps; tuned for the short "Real / Simulated" labels
            32,
        )

    # ── Interaction ────────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.toggle()
        super().mousePressEvent(ev)

    # ── Drawing ────────────────────────────────────────────────────────────
    def paintEvent(self, _ev):
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        fm = self.fontMetrics()
        left_w  = fm.horizontalAdvance(self._left_label)
        right_w = fm.horizontalAdvance(self._right_label)
        gap     = 10
        track_h = 26
        track_w = max(48, w - left_w - right_w - 2 * gap)
        track_x = left_w + gap
        track_y = (h - track_h) // 2

        # Track — accent colour on the right side state, neutral grey on left.
        accent_on  = QColor(THEME.get("accent",  "#4caf50"))
        accent_off = QColor("#555555")
        # Interpolate so the colour transitions during the animation.
        t = self._thumb_pos
        track_color = QColor(
            int(accent_off.red()   + t * (accent_on.red()   - accent_off.red())),
            int(accent_off.green() + t * (accent_on.green() - accent_off.green())),
            int(accent_off.blue()  + t * (accent_on.blue()  - accent_off.blue())),
        )
        p.setBrush(QBrush(track_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(track_x, track_y, track_w, track_h, track_h // 2, track_h // 2)

        # Thumb — circular knob, slides between the two ends.
        thumb_d  = track_h - 6
        thumb_x  = track_x + 3 + self._thumb_pos * (track_w - thumb_d - 6)
        thumb_y  = track_y + 3
        p.setBrush(QBrush(QColor("#fafafa")))
        p.setPen(QPen(QColor(0, 0, 0, 60), 1))
        p.drawEllipse(int(thumb_x), int(thumb_y), thumb_d, thumb_d)

        # Labels — bright when their side is selected, dimmed otherwise.
        dim    = QColor("#888888")
        bright = QColor("#ffffff")
        left_color  = bright if not self._checked else dim
        right_color = bright if self._checked     else dim
        p.setPen(QPen(left_color))
        p.drawText(
            0, 0, left_w, h,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self._left_label,
        )
        p.setPen(QPen(right_color))
        p.drawText(
            w - right_w, 0, right_w, h,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self._right_label,
        )


# ─────────────────────────────────────────────────────────────────────────────


class ReferencePatternPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 5 of 6 — Reference Pattern")
        self.setSubTitle(
            "Choose single-reference or per-grain mode, then click a point on "
            "the IPF map to set the reference pattern."
        )

        self._ref_marker    = None
        self._grain_markers = []   # per-grain mode: one marker per grain
        self._ipf_worker    = None
        self._sim_worker    = None
        self._refine_worker = None
        self._ref_pattern_set: ReferencePatternSet = None
        self._sim_pat_array = None   # last generated simulated pattern
        # Gate for the Sim-vs-Exp comparison dialog: only set True by the
        # explicit "Compare" button click.  Auto-regenerations (IPF clicks,
        # mode toggle, refinement) update the cached pattern + step-4 preview
        # silently without popping the comparison window.
        self._show_compare_dialog = False
        # Sim-vs-Exp dialog: lock the colour scale and colorbar after the
        # first click of Generate Pattern so subsequent clicks plot against
        # the same reference scale instead of stacking new colorbars.
        self._sim_pat_residual_vabs = None
        self._sim_pat_residual_cbar = None
        self._sim_exp_pat   = None   # experimental pattern at the clicked position
        # _sim_ref_row / _sim_ref_col were retired — sim mode now reads
        # the selected scan position directly from self.ref_row / self.ref_col
        # (the same spinboxes used by real-reference mode).
        self._ipf_rgb       = None   # cached RGB map for the main IPF

        # ── Reference mode switch ─────────────────────────────────────────────
        # iOS-style sliding toggle: left = real (experimental), right =
        # simulated.  Internally still backed by the existing QRadioButtons
        # (kept hidden) so all downstream code that checks
        # `self._single_radio.isChecked()` / `_sim_radio.isChecked()` and the
        # `_on_mode_changed(btn_id)` handler keep working unchanged.  The
        # per-grain (id=1) mode is no longer reachable from the UI but its
        # underlying widgets are preserved for easy re-enabling later.
        mode_group  = QGroupBox("Reference Pattern Type")
        mode_layout = QHBoxLayout()

        self._mode_switch = _ToggleSwitch(
            left_label="Real",
            right_label="Simulated",
            parent=self,
        )
        # Pin to its sizeHint instead of stretching — without this it
        # expands to fill the row and looks oversized.
        self._mode_switch.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        mode_layout.addWidget(self._mode_switch)
        mode_layout.addStretch()       # pushes the switch to the left
        mode_group.setLayout(mode_layout)

        # Hidden drivers: the QRadioButtons that the switch flips.  No
        # parent widget (None) and never added to a layout — they exist
        # only to fire the QButtonGroup idToggled signal that downstream
        # handlers listen to.
        self._single_radio   = QRadioButton(); self._single_radio.setVisible(False)
        self._sim_radio      = QRadioButton(); self._sim_radio.setVisible(False)
        self._pergrain_radio = QRadioButton(); self._pergrain_radio.setVisible(False)
        self._single_radio.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._single_radio,   0)
        self._mode_grp.addButton(self._sim_radio,      2)
        self._mode_grp.addButton(self._pergrain_radio, 1)
        self._mode_grp.idToggled.connect(self._on_mode_changed)

        # Connect the switch → radio drivers so the existing handler runs.
        def _switch_to_radio(checked: bool):
            if checked:
                self._sim_radio.setChecked(True)
            else:
                self._single_radio.setChecked(True)
        self._mode_switch.toggled.connect(_switch_to_radio)

        # ── Left panel (1/3): position controls + pattern preview ─────────────
        left        = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        # The Reference-Pattern-Type box used to live in a top row by itself.
        # Place it as the first widget in the left column instead — that way
        # it inherits the same width as the Reference Position group below.
        left_layout.addWidget(mode_group)

        # Single-reference position group
        self._pos_group  = QGroupBox("Reference Position")
        pos_layout = QFormLayout()

        self.ref_row = QSpinBox()
        self.ref_row.setRange(0, 9999)
        self.ref_row.setValue(0)

        self.ref_col = QSpinBox()
        self.ref_col.setRange(0, 9999)
        self.ref_col.setValue(0)

        self.ref_row.valueChanged.connect(self._update_ref_marker)
        self.ref_col.valueChanged.connect(self._update_ref_marker)
        # When in sim mode, changes to ref_row/ref_col should auto-reload the
        # Euler angles from the .ang file at the new scan pixel.  Wired here
        # so that BOTH IPF clicks (via _apply_ipf_click → setValue) and manual
        # spinbox edits trigger the reload.
        self.ref_row.valueChanged.connect(self._on_ref_position_changed)
        self.ref_col.valueChanged.connect(self._on_ref_position_changed)

        pos_layout.addRow("Row  (y):", self.ref_row)
        pos_layout.addRow("Col  (x):", self.ref_col)
        pos_layout.addRow(_note("(0, 0) is the top-left corner of the scan."))
        self._pos_group.setLayout(pos_layout)
        left_layout.addWidget(self._pos_group)

        # NOTE: the strain formulation toggle was moved to Step 2 (scan
        # geometry) so it sits alongside the other once-per-run geometry
        # choices instead of inside the per-reference settings.

        # Per-grain info group (hidden in single mode)
        self._grain_info_group  = QGroupBox("Per-Grain References")
        grain_info_layout = QVBoxLayout()
        self._grain_count_lbl = QLabel("No segmentation found — proceed to Step 4 first.")
        self._grain_count_lbl.setWordWrap(True)

        # Selection strategy dropdown — "closest to mean orientation" (default)
        # or "lowest KAM (most uniform)".  The KAM map comes from segmentation.
        _strategy_row = QHBoxLayout()
        _strategy_row.addWidget(QLabel("Pick reference by:"))
        self._ref_strategy_combo = QComboBox()
        self._ref_strategy_combo.addItems([
            "Closest to mean orientation",
            "Lowest KAM (most uniform)",
        ])
        self._ref_strategy_combo.setToolTip(
            "How to choose the representative pixel within each grain.\n"
            "• Closest to mean orientation — uses the arithmetic-mean quaternion.\n"
            "• Lowest KAM — uses the in-grain kernel-average misorientation\n"
            "  computed during segmentation.  Boundary pixels (high KAM) are\n"
            "  excluded by the interior filter.\n\n"
            "Both strategies erode each grain by 2 px before picking, to keep\n"
            "the reference away from grain boundaries / scan edges."
        )
        self._ref_strategy_combo.currentIndexChanged.connect(
            lambda _i: self._auto_select_references()
        )
        _strategy_row.addWidget(self._ref_strategy_combo, stretch=1)

        self._active_grain_lbl = QLabel("Active grain for override:")
        self._grain_combo = QComboBox()
        grain_info_layout.addWidget(self._grain_count_lbl)
        grain_info_layout.addLayout(_strategy_row)
        grain_info_layout.addWidget(self._active_grain_lbl)
        grain_info_layout.addWidget(self._grain_combo)
        grain_info_layout.addWidget(_note(
            "Click on the IPF map to move the selected grain's reference point."
        ))
        self._grain_info_group.setLayout(grain_info_layout)
        left_layout.addWidget(self._grain_info_group)

        # Simulated reference group
        self._sim_group  = QGroupBox("Simulated Reference")
        sim_layout = QFormLayout()

        # Master-pattern path is set in Step 1; surface it here read-only so
        # users have visual confirmation of what file will be used.
        self._sim_mp_status = QLabel("(set on Step 1)")
        self._sim_mp_status.setStyleSheet("color: gray; font-style: italic;")
        self._sim_mp_status.setWordWrap(True)
        sim_layout.addRow("Master pattern:", self._sim_mp_status)

        # Euler angles come from the .ang at the currently-selected ref
        # position — they are not user-tunable here.  Refinement / tuner
        # results overwrite the stored tuple and refresh the label.
        self._sim_euler_deg = (0.0, 0.0, 0.0)
        self._sim_euler_lbl = QLabel("φ₁ = —   Φ = —   φ₂ = —")
        self._sim_euler_lbl.setStyleSheet("font-family: monospace;")
        sim_layout.addRow("Euler (Bunge):", self._sim_euler_lbl)
        sim_layout.addRow(_note("Click the IPF map to update from scan orientation."))

        self._sim_gen_btn = QPushButton("Compare Simulated and Experimental Patterns")
        self._sim_gen_btn.clicked.connect(self._on_compare_clicked)
        sim_layout.addRow("", self._sim_gen_btn)

        self._sim_tuner_btn = QPushButton("Open Live Tuner…")
        self._sim_tuner_btn.setToolTip(
            "Open an interactive window where you can adjust Euler angles "
            "and PC values and see the simulated pattern update in real time."
        )
        self._sim_tuner_btn.clicked.connect(self._open_sim_tuner)
        sim_layout.addRow("", self._sim_tuner_btn)

        self._refine_btn = QPushButton("Refine PC && Euler…")
        self._refine_btn.setToolTip(
            "Open the PC / Euler refinement settings dialog.  From there, "
            "tune the Nelder-Mead controls (restarts, σ, symmetry) and "
            "click Run Refinement."
        )
        self._refine_btn.clicked.connect(self._open_refine_settings)
        sim_layout.addRow("", self._refine_btn)

        # The Nelder-Mead controls used to live inline here; they now live
        # in a dedicated dialog built in _build_refine_settings_dialog().

        # ── Simulated-pattern preprocessing overrides ───────────────────────
        # Per-sim values that override the Step 3 real-pattern preprocessing
        # at refinement time.  Lives here (not in the refine dialog) so the
        # overrides apply to any sim generation path that reads them.
        sim_pre_group = QGroupBox("Simulated Pattern Preprocessing Overrides")
        sim_pre_form  = QFormLayout()
        sim_pre_form.addRow(_note(
            "Set each to 0 to inherit from the Step 3 real-pattern value.\n"
            "Raise high-pass to strip sim background; lower gamma to reduce contrast."
        ))

        self._refine_sim_hp_sigma = QDoubleSpinBox()
        self._refine_sim_hp_sigma.setRange(0.0, 200.0)
        self._refine_sim_hp_sigma.setDecimals(1)
        self._refine_sim_hp_sigma.setSingleStep(5.0)
        self._refine_sim_hp_sigma.setValue(0.0)
        self._refine_sim_hp_sigma.setSpecialValueText("Same as real")
        self._refine_sim_hp_sigma.setToolTip(
            "High-pass sigma for the simulated pattern only.\n"
            "0 = use the Step 3 high-pass sigma unchanged."
        )
        sim_pre_form.addRow("Sim high-pass σ:", self._refine_sim_hp_sigma)

        self._refine_sim_lp_sigma = QDoubleSpinBox()
        self._refine_sim_lp_sigma.setRange(0.0, 20.0)
        self._refine_sim_lp_sigma.setDecimals(2)
        self._refine_sim_lp_sigma.setSingleStep(0.5)
        self._refine_sim_lp_sigma.setValue(0.0)
        self._refine_sim_lp_sigma.setSpecialValueText("Same as real")
        self._refine_sim_lp_sigma.setToolTip(
            "Low-pass (smoothing) sigma for the simulated pattern only.\n"
            "0 = use the Step 3 low-pass sigma unchanged."
        )
        sim_pre_form.addRow("Sim low-pass σ:", self._refine_sim_lp_sigma)

        self._refine_sim_gamma = QDoubleSpinBox()
        self._refine_sim_gamma.setRange(0.0, 3.0)
        self._refine_sim_gamma.setDecimals(3)
        self._refine_sim_gamma.setSingleStep(0.05)
        self._refine_sim_gamma.setValue(0.0)
        self._refine_sim_gamma.setSpecialValueText("Same as real")
        self._refine_sim_gamma.setToolTip(
            "Gamma correction for the simulated pattern only.\n"
            "0 = use the Step 3 gamma unchanged."
        )
        sim_pre_form.addRow("Sim gamma:", self._refine_sim_gamma)

        sim_pre_group.setLayout(sim_pre_form)
        sim_layout.addRow(sim_pre_group)

        # ── Pipeline-side reference-pattern tweaks (moved from Step 3) ──────
        # These three controls used to live on the Step 3 "Reference Pattern
        # Preprocessing (override)" group.  They've been migrated here so all
        # simulated-reference-related settings sit in one place.  The
        # per-component overrides + the master-toggle + the preview checkbox
        # from that group were dropped as redundant.
        sim_layout.addRow(_note("── Advanced (applied during IC-GN run) ──"))

        self._spectral_match_ref = QCheckBox(
            "Spectral-match reference to experimental patterns"
        )
        self._spectral_match_ref.setChecked(False)
        sim_layout.addRow(self._spectral_match_ref)

        self._tikhonov_perspective = QCheckBox(
            "Tikhonov-regularize perspective (h_31, h_32) in IC-GN"
        )
        self._tikhonov_perspective.setChecked(False)

        self._tikhonov_lambda = QDoubleSpinBox()
        self._tikhonov_lambda.setRange(0.0, 1000.0)
        self._tikhonov_lambda.setSingleStep(0.1)
        self._tikhonov_lambda.setDecimals(3)
        self._tikhonov_lambda.setValue(0.5)

        def _toggle_tikhonov(enabled: bool):
            self._tikhonov_lambda.setEnabled(enabled)
        self._tikhonov_perspective.toggled.connect(_toggle_tikhonov)
        _toggle_tikhonov(self._tikhonov_perspective.isChecked())

        sim_layout.addRow(self._tikhonov_perspective)
        sim_layout.addRow("Tikhonov λ multiplier:", self._tikhonov_lambda)

        self._rotate_patterns_90 = QCheckBox(
            "Rotate patterns 90° CCW before optimization (diagnostic)"
        )
        self._rotate_patterns_90.setChecked(False)
        sim_layout.addRow(self._rotate_patterns_90)

        self._sim_status = QLabel("")
        self._sim_status.setStyleSheet("color: gray; font-size: 11px;")
        self._sim_status.setWordWrap(True)
        sim_layout.addRow(self._sim_status)

        self._sim_group.setLayout(sim_layout)

        # Simulated-reference settings live in their own pop-up dialog so the
        # main page stays compact when other reference modes are selected.
        self._sim_settings_dialog = QDialog(self)
        self._sim_settings_dialog.setWindowTitle("Simulated Reference — Parameters")
        self._sim_settings_dialog.setModal(False)
        self._sim_settings_dialog.resize(700, 720)
        _ssd_outer = QHBoxLayout(self._sim_settings_dialog)
        _ssd_outer.setContentsMargins(12, 12, 12, 12)
        _ssd_outer.addWidget(self._sim_group, stretch=1)

        # NOTE: the sim-settings dialog used to embed its own clickable IPF
        # map.  It was removed — the main GUI's IPF on Step 4 is now the
        # single source of truth (clicks there update ref_row/ref_col and,
        # in sim mode, auto-load the Euler angles).

        # NOTE: the "Simulated Reference" placeholder card used to live here
        # (with an "Open simulated reference settings…" button).  It's been
        # removed — flipping the mode switch to Simulated now auto-opens the
        # settings dialog directly, so the placeholder was redundant.

        self._build_refine_settings_dialog()

        # ── Reference Pattern Preview (lower-left) ────────────────────────────
        ref_pat_group  = QGroupBox("Reference Pattern")
        ref_pat_layout = QVBoxLayout(ref_pat_group)
        _bg = THEME["surface_bg"]
        self._ref_pat_fig = Figure(facecolor=_bg)
        self._ref_pat_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._ref_pat_ax  = self._ref_pat_fig.add_subplot(111)
        self._ref_pat_ax.set_facecolor(_bg)
        self._ref_pat_ax.set_visible(False)
        self._ref_pat_canvas = FigureCanvas(self._ref_pat_fig)
        self._ref_pat_canvas.setStyleSheet(f"background-color: {_bg};")
        self._ref_pat_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ref_pat_canvas.setMinimumSize(220, 220)
        self._ref_pat_status = QLabel("")
        self._ref_pat_status.setStyleSheet("color: gray; font-size: 11px;")
        self._ref_pat_status.setWordWrap(True)
        ref_pat_layout.addWidget(self._ref_pat_canvas, stretch=1)
        ref_pat_layout.addWidget(self._ref_pat_status)
        left_layout.addWidget(ref_pat_group, stretch=1)

        self._ref_pat_timer = QTimer(self)
        self._ref_pat_timer.setSingleShot(True)
        self._ref_pat_timer.setInterval(200)
        self._ref_pat_timer.timeout.connect(self._load_ref_pattern_preview)
        self.ref_row.valueChanged.connect(self._ref_pat_timer.start)
        self.ref_col.valueChanged.connect(self._ref_pat_timer.start)

        # ── Right panel (2/3): clickable IPF map ──────────────────────────────
        right        = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)

        ipf_group  = QGroupBox("IPF Map  (ND)  —  click to set reference position")
        ipf_layout = QVBoxLayout(ipf_group)

        self._ipf_fig = Figure()
        self._ipf_ax  = self._ipf_fig.add_subplot(111)
        self._ipf_ax.set_visible(False)
        self._ipf_canvas = FigureCanvas(self._ipf_fig)
        self._ipf_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ipf_canvas.setMinimumSize(320, 320)
        self._ipf_canvas.mpl_connect("button_press_event", self._on_ipf_click)

        self._ipf_status = QLabel("Loading IPF map…")
        self._ipf_status.setStyleSheet("color: gray;")
        self._ipf_status.setWordWrap(True)

        ipf_layout.addWidget(self._ipf_canvas, stretch=1)
        ipf_layout.addWidget(self._ipf_status)
        right_layout.addWidget(ipf_group)

        # ── Outer layout ──────────────────────────────────────────────────────
        # The Reference-Pattern-Type group is now the first widget in the
        # left column (above), so the outer layout is just the two-panel
        # split with no separate top row.
        outer = QVBoxLayout()

        panels = QHBoxLayout()
        panels.addWidget(left,  stretch=1)
        panels.addWidget(right, stretch=2)
        outer.addLayout(panels, stretch=1)
        self.setLayout(outer)

        # Simulated pattern viewer window
        _bg = THEME["surface_bg"]
        self._sim_dialog = QDialog(self)
        self._sim_dialog.setWindowTitle("Simulated vs Experimental Reference Pattern")
        self._sim_dialog.resize(1300, 950)
        _sdlg_layout = QVBoxLayout(self._sim_dialog)
        self._sim_pat_fig = Figure(tight_layout=True, facecolor=_bg)
        self._sim_pat_ax_exp  = self._sim_pat_fig.add_subplot(131)   # left:   experimental
        self._sim_pat_ax_sim  = self._sim_pat_fig.add_subplot(132)   # centre: simulated
        self._sim_pat_ax_diff = self._sim_pat_fig.add_subplot(133)   # right:  difference
        for ax in (self._sim_pat_ax_exp, self._sim_pat_ax_sim, self._sim_pat_ax_diff):
            ax.set_facecolor(_bg)
        self._sim_pat_canvas = FigureCanvas(self._sim_pat_fig)
        self._sim_pat_canvas.setStyleSheet(f"background-color: {_bg};")
        self._sim_pat_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _sdlg_layout.addWidget(self._sim_pat_canvas, stretch=2)

        # ── Flicker (alternating sim/exp) panel ──────────────────────────────
        # Classic EBSD pattern-matching trick — when the two patterns alternate
        # at ~5-10 Hz, the eye picks up structural mismatches as "motion"
        # (band edges shifting, intensity changes) far more sensitively than
        # by comparing them side-by-side.
        flicker_row = QHBoxLayout()

        self._sim_flicker_play_btn = QPushButton("▶ Play flicker")
        self._sim_flicker_play_btn.setCheckable(True)
        self._sim_flicker_play_btn.setStyleSheet(
            f"background-color: {THEME['accent']}; color: {THEME['accent_text']}; "
            f"font-weight: bold; padding: 4px 12px; border-radius: 4px;"
        )
        self._sim_flicker_play_btn.toggled.connect(self._on_flicker_toggle)
        flicker_row.addWidget(self._sim_flicker_play_btn)

        flicker_row.addWidget(QLabel("Speed:"))
        self._sim_flicker_speed = QSlider(Qt.Orientation.Horizontal)
        self._sim_flicker_speed.setMinimum(50)
        self._sim_flicker_speed.setMaximum(1000)
        self._sim_flicker_speed.setValue(250)      # ~4 Hz default
        self._sim_flicker_speed.setSingleStep(25)
        self._sim_flicker_speed.setFixedWidth(220)
        self._sim_flicker_speed.valueChanged.connect(self._on_flicker_speed_changed)
        flicker_row.addWidget(self._sim_flicker_speed)

        self._sim_flicker_speed_lbl = QLabel("250 ms / frame")
        self._sim_flicker_speed_lbl.setStyleSheet("color: white; font-size: 11px;")
        flicker_row.addWidget(self._sim_flicker_speed_lbl)

        flicker_row.addStretch()

        self._sim_flicker_frame_lbl = QLabel("(flicker stopped)")
        self._sim_flicker_frame_lbl.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        flicker_row.addWidget(self._sim_flicker_frame_lbl)

        _sdlg_layout.addLayout(flicker_row)

        # Big single-pattern canvas for the flicker view.
        self._sim_flicker_fig = Figure(tight_layout=True, facecolor=_bg)
        self._sim_flicker_ax  = self._sim_flicker_fig.add_subplot(111)
        self._sim_flicker_ax.set_facecolor(_bg)
        self._sim_flicker_ax.axis("off")
        self._sim_flicker_canvas = FigureCanvas(self._sim_flicker_fig)
        self._sim_flicker_canvas.setStyleSheet(f"background-color: {_bg};")
        self._sim_flicker_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _sdlg_layout.addWidget(self._sim_flicker_canvas, stretch=3)

        # Per-frame state for the flicker.
        self._sim_flicker_exp = None        # latest processed exp pattern
        self._sim_flicker_sim = None        # latest processed sim pattern
        self._sim_flicker_im  = None        # AxesImage handle, reused for speed
        self._sim_flicker_show_sim = True   # which one is on screen RIGHT NOW
        self._sim_flicker_timer = QTimer(self)
        self._sim_flicker_timer.timeout.connect(self._on_flicker_tick)

        self._sim_dlg_status = QLabel("")
        self._sim_dlg_status.setStyleSheet("color: white; font-size: 13px;")
        self._sim_dlg_status.setWordWrap(True)
        _sdlg_layout.addWidget(self._sim_dlg_status)

        # Initial visibility
        self._grain_info_group.setVisible(False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initializePage(self):
        wiz      = self.wizard()
        geom     = wiz.geometry_page.get_params()
        ang_path = wiz.field("ang_path")
        patshape = (geom["pat_h"], geom["pat_w"])

        self.ref_row.setMaximum(geom["rows"] - 1)
        self.ref_col.setMaximum(geom["cols"] - 1)

        # Mirror the master-pattern path from Step 1 into the read-only label.
        mp_path = (wiz.field("master_pattern_path") or "").strip()
        if mp_path:
            self._sim_mp_status.setText(mp_path)
            self._sim_mp_status.setStyleSheet("color: gray;")
        else:
            self._sim_mp_status.setText("(not set — go to Step 1 to choose an .h5 master pattern)")
            self._sim_mp_status.setStyleSheet("color: gray; font-style: italic;")

        if not ang_path or not os.path.exists(ang_path):
            self._ipf_status.setText("No ANG file — cannot draw IPF map.")
            return

        self._ipf_status.setText("Computing IPF map… (this may take a few seconds)")
        cached = wiz.ang_data if wiz.ang_data is not None and wiz.ang_loaded_path == ang_path else None
        # Defer the worker start until the next event-loop tick so that
        # showEvent / resize have run and the canvas has a non-zero size.
        # Otherwise the worker can finish (especially with cached ang_data)
        # before the figure has been laid out, and matplotlib renders into
        # a 0×0 region that never gets repainted.
        def _spawn():
            print(f"[Step 4 IPF] starting worker, ang={ang_path}, cached={'yes' if cached is not None else 'no'}")
            self._ipf_worker = IPFWorker(ang_path, patshape, np.array([0.0, 0.0, 1.0]), ang_data=cached)
            self._ipf_worker.done_signal.connect(self._on_ipf_done)
            self._ipf_worker.start()
        QTimer.singleShot(0, _spawn)

    def showEvent(self, event):
        """If the IPF was rendered before the canvas had its real size
        (worker emitted faster than the page laid out), force a redraw
        once the page is actually visible."""
        super().showEvent(event)
        if self._ipf_rgb is not None and self._ipf_ax.get_visible():
            self._ipf_canvas.draw_idle()

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_mode_changed(self, btn_id: int, checked: bool):
        if not checked:
            return
        # Reference Position is always shown — the row/col spinboxes are
        # the single source of truth for "where the reference is selected",
        # regardless of mode.  In sim mode, the spinboxes additionally
        # drive the Euler-angle auto-load via _on_ref_position_changed.
        self._pos_group.setVisible(True)
        self._grain_info_group.setVisible(btn_id == 1)
        if btn_id == 2:
            # Pull Euler from .ang at the currently-selected ref position
            # BEFORE opening the dialog — _open_sim_settings auto-triggers
            # pattern generation and we don't want it to run on stale
            # (0,0,0) angles.
            self._reload_sim_euler_from_ref_position()
            self._open_sim_settings()
        else:
            self._sim_settings_dialog.hide()
        if btn_id == 1:
            self._auto_select_references()
        elif btn_id == 0:
            self._clear_grain_markers()
            self._update_ref_marker()
        else:
            self._clear_grain_markers()
            self._update_ref_marker()
        # Reference-pattern preview is shared between modes — refresh so
        # it shows experimental (real mode) or the cached sim (sim mode).
        self._load_ref_pattern_preview()

    def _auto_select_references(self):
        """Build a ReferencePatternSet from the current segmentation result."""
        wiz = self.wizard()
        grain_ids = getattr(wiz.roi_page, "_grain_ids", None)
        ang_data  = wiz.ang_data

        if grain_ids is None:
            self._grain_count_lbl.setText(
                "No segmentation found — proceed to Step 4 and run segmentation first."
            )
            self._ref_pattern_set = None
            self._grain_combo.clear()
            self._clear_grain_markers()
            return

        if ang_data is None:
            self._grain_count_lbl.setText("ANG data not loaded yet — please wait.")
            return

        geom = wiz.geometry_page.get_params()
        strategy = ("kam_min" if self._ref_strategy_combo.currentIndex() == 1
                    else "mean")
        kam = getattr(wiz.roi_page, "_kam", None)
        if strategy == "kam_min" and kam is None:
            # Fall back to mean if segmentation didn't store a KAM (e.g. older
            # cached run).  The UI hint also tells the user to re-segment.
            strategy = "mean"
        self._ref_pattern_set = select_references(
            grain_ids, ang_data, geom["cols"],
            strategy=strategy,
            kam=kam,
            interior_erode=2,
        )

        n = len(self._ref_pattern_set)
        _strat_label = ("lowest KAM" if strategy == "kam_min"
                        else "closest to mean orientation")
        self._grain_count_lbl.setText(
            f"{n} grain{'s' if n != 1 else ''} found — one reference "
            f"auto-selected per grain ({_strat_label}, interior-filtered)."
        )

        self._grain_combo.blockSignals(True)
        self._grain_combo.clear()
        for entry in self._ref_pattern_set:
            self._grain_combo.addItem(
                f"Grain {entry.grain_id}  (row={entry.ref_row}, col={entry.ref_col})",
                userData=entry.grain_id,
            )
        self._grain_combo.blockSignals(False)

        self._draw_grain_markers()

    def _draw_grain_markers(self):
        """Place one '+' marker per grain on the IPF map."""
        self._clear_grain_markers()
        if not self._ipf_ax.get_visible() or self._ref_pattern_set is None:
            return
        colors = ["#fdca40", "#f38ba8", "#a6e3a1", "#89dceb", "#cba6f7",
                  "#fab387", "#89b4fa", "#eba0ac", "#94e2d5", "#b4befe"]
        for i, entry in enumerate(self._ref_pattern_set):
            color = colors[i % len(colors)]
            marker, = self._ipf_ax.plot(
                entry.ref_col, entry.ref_row,
                marker="+", color=color,
                markersize=14, markeredgewidth=2.5, zorder=10, linestyle="none",
            )
            self._grain_markers.append(marker)
        self._ipf_canvas.draw_idle()

    def _clear_grain_markers(self):
        for m in self._grain_markers:
            try:
                m.remove()
            except Exception:
                pass
        self._grain_markers.clear()
        self._ipf_canvas.draw_idle()

    # ── IPF map ───────────────────────────────────────────────────────────────
    # NOTE: _draw_dialog_ipf used to mirror the page IPF onto the sim
    # dialog's embedded IPF; that dialog IPF is gone now.

    def _on_ipf_done(self, rgb_map, error: str):
        print(f"[Step 4 IPF] _on_ipf_done fired — rgb_map shape="
              f"{None if rgb_map is None else rgb_map.shape}  error={'<none>' if not error else error[:120]}")
        if rgb_map is None:
            self._ipf_status.setText(f"Error computing IPF map: {error}")
            return
        self._ipf_rgb = rgb_map
        self._ipf_ax.set_visible(True)
        self._ipf_ax.clear()
        self._ipf_ax.imshow(rgb_map, origin="upper", interpolation="nearest")
        self._ipf_ax.axis("off")
        try:
            self._ipf_fig.tight_layout(pad=0.5)
        except Exception as exc:
            print(f"[Step 4 IPF] tight_layout warning: {exc}")
        self._ref_marker = None
        self._grain_markers.clear()
        if self._single_radio.isChecked():
            self._update_ref_marker()
        else:
            self._draw_grain_markers()
        self._ipf_canvas.draw()
        # Fallback: re-issue draw_idle on the next event-loop tick in case
        # the canvas size changed between the worker's emit and now.
        QTimer.singleShot(0, self._ipf_canvas.draw_idle)
        self._ipf_status.setText("Click any point to set the reference pattern.")

    # NOTE: _update_sim_ref_marker was retired with the sim-dialog IPF.
    # Callers used to invoke it after every ref-position change; those
    # call sites have been removed below.

    def _apply_ipf_click(self, row: int, col: int):
        """Mode-aware action when the user clicks a row/col on either IPF axis."""
        wiz  = self.wizard()
        geom = wiz.geometry_page.get_params()
        col  = max(0, min(col, geom["cols"] - 1))
        row  = max(0, min(row, geom["rows"] - 1))

        if self._single_radio.isChecked() or self._sim_radio.isChecked():
            # Both real and simulated modes now write to the SAME spinboxes —
            # ref_row / ref_col are the single source of truth for "selected
            # scan pixel".  The valueChanged signals fire _update_ref_marker
            # (always) and _on_ref_position_changed (which reloads the Euler
            # in sim mode), so we don't need to call those directly.
            self.ref_row.setValue(row)
            self.ref_col.setValue(col)
        else:
            # Per-grain: update the active grain's reference
            gid = self._grain_combo.currentData()
            if gid is None or self._ref_pattern_set is None:
                return
            pat_idx = row * geom["cols"] + col
            ang_data = wiz.ang_data
            euler = None
            if ang_data is not None:
                euler = tuple(ang_data.eulers[row, col])
            self._ref_pattern_set.update_ref(gid, row, col, pat_idx, euler)
            # Update combo label
            idx = self._grain_combo.currentIndex()
            self._grain_combo.setItemText(idx, f"Grain {gid}  (row={row}, col={col})")
            self._draw_grain_markers()

    def _on_ipf_click(self, event):
        if event.inaxes is not self._ipf_ax or not self._ipf_ax.get_visible():
            return
        self._apply_ipf_click(int(round(event.ydata)), int(round(event.xdata)))

    # _on_sim_ipf_click was retired with the sim-dialog IPF.

    def _update_ref_marker(self):
        """Reference-position marker on the main IPF.  Now used in both
        real-pattern and simulated-pattern modes (the spinboxes drive the
        marker regardless of which radio is active)."""
        if not self._ipf_ax.get_visible():
            return
        if self._ref_marker is not None:
            try:
                self._ref_marker.remove()
            except Exception:
                pass
            self._ref_marker = None
        row = self.ref_row.value()
        col = self.ref_col.value()
        self._ref_marker, = self._ipf_ax.plot(
            col, row, marker="+", color="white",
            markersize=14, markeredgewidth=2.5, zorder=10, linestyle="none"
        )
        self._ipf_canvas.draw_idle()
        self._ref_pat_timer.start()

    def _on_ref_position_changed(self, _value: int = 0):
        """Slot fired when ref_row or ref_col changes (from either an IPF
        click or a manual spinbox edit).  In sim mode, reload the Euler
        angles from .ang at the new scan position AND re-generate the
        simulated pattern so the step-4 preview tracks the click."""
        if self._sim_radio.isChecked():
            self._reload_sim_euler_from_ref_position()
            mp_path = (self.wizard().field("master_pattern_path") or "").strip()
            if mp_path and os.path.exists(mp_path):
                self._generate_sim_pattern()

    def _reload_sim_euler_from_ref_position(self):
        """Pull (φ₁, Φ, φ₂) from the .ang at the currently-selected
        ref position into the sim-Euler label / cached tuple."""
        wiz = self.wizard()
        if wiz is None or wiz.ang_data is None:
            return
        row = self.ref_row.value()
        col = self.ref_col.value()
        try:
            eu = wiz.ang_data.eulers[row, col]    # radians
        except Exception:
            return
        self._set_sim_euler_deg((float(np.degrees(eu[0])),
                                 float(np.degrees(eu[1])),
                                 float(np.degrees(eu[2]))))
        self._sim_status.setText(
            f"Euler angles auto-loaded from row {row}, col {col}."
        )

    def _set_sim_euler_deg(self, euler_deg: tuple):
        """Cache the current (φ₁, Φ, φ₂) in degrees and refresh the label."""
        self._sim_euler_deg = (float(euler_deg[0]),
                               float(euler_deg[1]),
                               float(euler_deg[2]))
        self._sim_euler_lbl.setText(
            f"φ₁ = {self._sim_euler_deg[0]:7.3f}°   "
            f"Φ = {self._sim_euler_deg[1]:7.3f}°   "
            f"φ₂ = {self._sim_euler_deg[2]:7.3f}°"
        )

    def _load_ref_pattern_preview(self):
        """Render the reference-pattern preview.

        Real modes (single / per-grain) → raw experimental pattern at
        (ref_row, ref_col).  Sim mode → last generated simulated pattern,
        or a hint to generate one if none is cached yet."""
        wiz = self.wizard()
        if wiz is None:
            return
        row = self.ref_row.value()
        col = self.ref_col.value()

        if self._sim_radio.isChecked():
            if self._sim_pat_array is not None:
                self._ref_pat_ax.clear()
                self._ref_pat_ax.imshow(self._sim_pat_array, cmap="gray", origin="upper")
                self._ref_pat_ax.axis("off")
                self._ref_pat_ax.set_visible(True)
                self._ref_pat_canvas.draw_idle()
                self._ref_pat_status.setText(
                    f"Simulated pattern  (φ₁={self._sim_euler_deg[0]:.2f}°  "
                    f"Φ={self._sim_euler_deg[1]:.2f}°  φ₂={self._sim_euler_deg[2]:.2f}°)."
                )
                return
            self._ref_pat_ax.clear()
            self._ref_pat_ax.set_visible(False)
            self._ref_pat_canvas.draw_idle()
            self._ref_pat_status.setText(
                "Simulated pattern not ready yet — generation runs automatically "
                "when the simulated-reference settings open."
            )
            return

        up2_path = wiz.field("up2_path")
        if not up2_path or not os.path.exists(up2_path):
            self._ref_pat_status.setText("No UP2 file loaded.")
            return
        try:
            geom = wiz.geometry_page.get_params()
            if row >= geom["rows"] or col >= geom["cols"]:
                return
            pat_idx = int(np.ravel_multi_index((row, col), (geom["rows"], geom["cols"])))
            import Data
            pat_obj = Data.UP2(up2_path)
            pat = pat_obj.read_pattern(pat_idx, process=False)
            self._ref_pat_ax.clear()
            self._ref_pat_ax.imshow(pat, cmap="gray", origin="upper")
            self._ref_pat_ax.axis("off")
            self._ref_pat_ax.set_visible(True)
            self._ref_pat_canvas.draw_idle()
            self._ref_pat_status.setText(f"Pattern at row {row}, col {col}.")
        except Exception as exc:
            self._ref_pat_status.setText(f"Could not load pattern: {exc}")

    # ── Pattern preview ───────────────────────────────────────────────────────

    # ── Simulated reference ───────────────────────────────────────────────────

    def _apply_step3_processing(self, img, up2_path: str):
        """Run a numpy pattern through the Step 3 high-pass/low-pass/gamma
        pipeline so its intensity characteristics match what the optimizer
        will see at runtime.  Applies the user's Step 3 mask too (whatever
        mask_type they picked) — matches the convention now used in
        get_homography_cpu.simulate_reference_pattern and
        optimize_reference._simulate, so the dialog's display + ZNSSD
        residual reflect the same masked comparison the optimiser sees.

        Returns the processed pattern normalised to [0, 1].  If up2_path
        is missing/invalid (no Data.UP2 can be created), returns img
        normalised to [0, 1] with no filtering applied.
        """
        import numpy as np
        arr = np.asarray(img, dtype=np.float32)
        if not up2_path or not os.path.exists(up2_path):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-9)
        try:
            import Data
            wiz = self.wizard()
            p = wiz.processing_page.get_params() if wiz else {}
            pat_obj = Data.UP2(up2_path)
            mask_type = p.get("mask_type", "None")
            if mask_type == "None":
                mask_type = None
            pat_obj.set_processing(
                low_pass_sigma          = p.get("low_pass_sigma", 1.0),
                high_pass_sigma         = p.get("high_pass_sigma", 10.0),
                truncate_std_scale      = 3.0,
                mask_type               = mask_type,
                center_cross_half_width = 6,
                flip_x                  = p.get("flip_x", False),
                gamma                   = p.get("gamma", 0.8),
            )
            proc = pat_obj.process_pattern(arr.copy()).astype(np.float32)
            lo, hi = proc.min(), proc.max()
            return (proc - lo) / (hi - lo + 1e-9)
        except Exception:
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-9)

    def _open_sim_settings(self):
        """Show the simulated-reference settings dialog (non-modal) and
        kick off pattern generation immediately — the master pattern is
        loaded once on the first generation and cached by SimPatGen, so
        subsequent runs (re-position, refinement, manual re-compare) skip
        the disk hit."""
        self._sim_settings_dialog.show()
        self._sim_settings_dialog.raise_()
        self._sim_settings_dialog.activateWindow()
        mp_path = (self.wizard().field("master_pattern_path") or "").strip()
        if mp_path and os.path.exists(mp_path):
            self._generate_sim_pattern()

    def _on_compare_clicked(self):
        """Explicit user click on the Compare button — set the gate so
        _on_sim_done pops the sim-vs-exp comparison dialog, then trigger
        a fresh generation."""
        self._show_compare_dialog = True
        self._generate_sim_pattern()

    def _open_refine_settings(self):
        """Show the PC / Euler refinement settings dialog (non-modal)."""
        self._refine_settings_dialog.show()
        self._refine_settings_dialog.raise_()
        self._refine_settings_dialog.activateWindow()

    def _build_refine_settings_dialog(self):
        """Construct the dedicated PC / Euler refinement settings dialog.
        All Nelder-Mead controls (max iters, restarts, σ, RNG seed, symmetry
        toggle, Laue group, sim-pattern preprocessing overrides) live here
        so the sim-settings dialog stays focused on pattern generation."""
        self._refine_settings_dialog = QDialog(self)
        self._refine_settings_dialog.setWindowTitle("PC / Euler Refinement — Settings")
        self._refine_settings_dialog.setModal(False)
        self._refine_settings_dialog.resize(560, 720)
        outer = QVBoxLayout(self._refine_settings_dialog)
        outer.setContentsMargins(12, 12, 12, 12)
        form = QFormLayout()

        self._refine_max_iter = QSpinBox()
        self._refine_max_iter.setRange(50, 2000)
        self._refine_max_iter.setValue(300)
        self._refine_max_iter.setSingleStep(50)
        self._refine_max_iter.setToolTip(
            "Maximum Nelder-Mead function evaluations per restart.\n"
            "Total cost ≈ (Refine max iters) × (Restarts)."
        )
        form.addRow("Refine max iters:", self._refine_max_iter)

        form.addRow(_note("── Multi-start Nelder-Mead ──"))

        self._refine_n_restarts = QSpinBox()
        self._refine_n_restarts.setRange(1, 32)
        self._refine_n_restarts.setValue(4)
        self._refine_n_restarts.setToolTip(
            "Number of Nelder-Mead runs.  Restart 1 starts at delta=0 "
            "(identical to single-start behaviour); restarts 2..N start at "
            "random Gaussian perturbations of zero.  Defends against local "
            "minima from symmetry-equivalent basins and PC/Euler degeneracy.\n"
            "Set to 1 to disable multi-start.  Total cost ≈ N × max_iter."
        )
        form.addRow("Restarts (N):", self._refine_n_restarts)

        self._refine_restart_rotvec_std = QDoubleSpinBox()
        self._refine_restart_rotvec_std.setRange(0.0, 30.0)
        self._refine_restart_rotvec_std.setDecimals(2)
        self._refine_restart_rotvec_std.setSingleStep(0.5)
        self._refine_restart_rotvec_std.setValue(3.0)
        self._refine_restart_rotvec_std.setSuffix(" °")
        self._refine_restart_rotvec_std.setToolTip(
            "Standard deviation of the random rotation-vector seed used for "
            "restarts 2..N (in degrees).  Typical .ang orientations are "
            "accurate to a few degrees, so 3° is a good default."
        )
        form.addRow("Restart rotvec σ:", self._refine_restart_rotvec_std)

        self._refine_restart_pc_std = QDoubleSpinBox()
        self._refine_restart_pc_std.setRange(0.0, 0.5)
        self._refine_restart_pc_std.setDecimals(4)
        self._refine_restart_pc_std.setSingleStep(0.005)
        self._refine_restart_pc_std.setValue(0.01)
        self._refine_restart_pc_std.setToolTip(
            "Standard deviation of the random PC seed used for restarts 2..N "
            "(Bruker units).  0.01 ≈ 1% of the detector size."
        )
        form.addRow("Restart PC σ:", self._refine_restart_pc_std)

        self._refine_restart_seed = QSpinBox()
        self._refine_restart_seed.setRange(0, 999999)
        self._refine_restart_seed.setValue(0)
        self._refine_restart_seed.setToolTip(
            "RNG seed for the random restart perturbations — same seed gives "
            "the same restart sequence across runs (useful for debugging)."
        )
        form.addRow("Restart RNG seed:", self._refine_restart_seed)

        self._refine_symmetry_restarts = QCheckBox(
            "Symmetry-informed restarts (one per Laue group element)"
        )
        self._refine_symmetry_restarts.setChecked(False)
        self._refine_symmetry_restarts.setToolTip(
            "When ticked, after the N random restarts run one additional "
            "Nelder-Mead from each non-identity element of the crystal's "
            "Laue group.  Defends against converging to the wrong symmetry-"
            "equivalent of the true orientation.\n\n"
            "Cost: adds (|Laue group| − 1) restarts on top of the N random "
            "ones (e.g. +23 runs for cubic m-3m)."
        )
        form.addRow(self._refine_symmetry_restarts)

        self._refine_laue_group = QComboBox()
        self._refine_laue_group.addItems([
            "1: C1 (triclinic)",
            "2: C2 (monoclinic)",
            "3: C3 (trigonal-low)",
            "4: C4 (tetragonal-low)",
            "5: C6 (hexagonal-low)",
            "6: D2 (orthorhombic)",
            "7: D3 (trigonal-high)",
            "8: D4 (tetragonal-high)",
            "9: D6 (hexagonal-high)",
            "10: T (cubic-low, m-3)",
            "11: O (cubic-high, m-3m)",
        ])
        self._refine_laue_group.setCurrentIndex(10)
        self._refine_laue_group.setToolTip(
            "Crystal Laue group used to enumerate symmetry-equivalent "
            "orientations.  Cubic m-3m (11) covers most metals."
        )
        form.addRow("Laue group:", self._refine_laue_group)

        # Simulated-pattern preprocessing overrides used to live here; they
        # now sit in their own sub-group in the Simulated-Reference dialog.

        self._refine_status = QLabel("")
        self._refine_status.setStyleSheet("color: gray; font-size: 11px;")
        self._refine_status.setWordWrap(True)
        form.addRow(self._refine_status)

        outer.addLayout(form)

        self._refine_run_btn = QPushButton("Run Refinement")
        self._refine_run_btn.setToolTip(
            "Start the multi-start Nelder-Mead refinement of PC and Euler "
            "angles using the settings above."
        )
        self._refine_run_btn.clicked.connect(self._run_pc_euler_refine)
        outer.addWidget(self._refine_run_btn)

        # Embedded log: replaces the standalone progress dialog so the user
        # can watch refinement output without juggling two windows.
        self._refine_log_box = QTextEdit()
        self._refine_log_box.setReadOnly(True)
        self._refine_log_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._refine_log_box.setMinimumHeight(180)
        outer.addWidget(self._refine_log_box, stretch=1)

        # Pending refinement results — populated by _on_done, applied by
        # Finish & Apply, discarded by Cancel.
        self._refine_pending_euler_deg = None
        self._refine_pending_pc        = None

        btn_row = QHBoxLayout()
        self._refine_apply_btn = QPushButton("Finish && Apply")
        self._refine_apply_btn.setToolTip(
            "Apply the refined Euler angles and PC to the reference page "
            "and close this dialog."
        )
        self._refine_apply_btn.clicked.connect(self._on_refine_apply)
        self._refine_cancel_btn = QPushButton("Cancel")
        self._refine_cancel_btn.setToolTip(
            "Discard any refinement results and close this dialog."
        )
        self._refine_cancel_btn.clicked.connect(self._on_refine_cancel)
        btn_row.addWidget(self._refine_apply_btn)
        btn_row.addWidget(self._refine_cancel_btn)
        outer.addLayout(btn_row)

    def _on_refine_apply(self):
        """Finish & Apply — push pending refinement results to the
        reference page and the geometry-page PC fields, then close."""
        if self._refine_pending_euler_deg is not None:
            self._set_sim_euler_deg(self._refine_pending_euler_deg)
            pc = self._refine_pending_pc
            geom_page = self.wizard().geometry_page
            geom_page.pc_x.setValue(float(pc[0]))
            geom_page.pc_y.setValue(float(pc[1]))
            geom_page.pc_z.setValue(float(pc[2]))
            # Auto-regenerate the sim at the freshly-applied values so the
            # step-4 preview reflects the refined orientation immediately.
            self._generate_sim_pattern()
        self._refine_pending_euler_deg = None
        self._refine_pending_pc        = None
        self._refine_settings_dialog.hide()

    def _on_refine_cancel(self):
        """Discard pending results and close the dialog."""
        self._refine_pending_euler_deg = None
        self._refine_pending_pc        = None
        self._refine_settings_dialog.hide()

    def _generate_sim_pattern(self):
        mp_path = (self.wizard().field("master_pattern_path") or "").strip()
        if not mp_path or not os.path.exists(mp_path):
            self._sim_status.setText(
                "Please select a valid master pattern file on Step 1 first."
            )
            return

        wiz  = self.wizard()
        geom = wiz.geometry_page.get_params()
        pc   = geom["pc_edax"]
        det_shape = (geom["pat_h"], geom["pat_w"])
        euler_deg = tuple(self._sim_euler_deg)

        # Load the experimental pattern at the clicked position for comparison.
        # self._sim_exp_pat is kept as the RAW normalized pattern (SimTuner
        # depends on this contract); _sim_exp_pat_proc holds the Step-3-processed
        # version used for display and the ZNSSD residual.
        self._sim_exp_pat      = None
        self._sim_exp_pat_proc = None
        up2_path = wiz.field("up2_path")
        if up2_path and os.path.exists(up2_path):
            try:
                import Data
                pat_obj = Data.UP2(up2_path)
                idx = self.ref_row.value() * geom["cols"] + self.ref_col.value()
                if idx < pat_obj.nPatterns:
                    raw = pat_obj.read_pattern(idx, process=False).astype(np.float32)
                    lo, hi = raw.min(), raw.max()
                    self._sim_exp_pat      = (raw - lo) / (hi - lo + 1e-9)
                    # Step-3-processed version for display / ZNSSD comparison
                    self._sim_exp_pat_proc = self._apply_step3_processing(
                        raw, up2_path
                    )
            except Exception:
                pass

        self._sim_gen_btn.setEnabled(False)
        self._sim_status.setText("Generating simulated pattern…")

        self._sim_worker = SimRefWorker(
            mp_path, euler_deg, pc, det_shape,
            det_tilt_deg    = geom["det_tilt"],
            sample_tilt_deg = geom["tilt"],
        )
        self._sim_worker.done_signal.connect(self._on_sim_done)
        self._sim_worker.error_signal.connect(self._on_sim_error)
        self._sim_worker.start()

    def _on_sim_done(self, pat_np):
        # Keep the RAW sim around — downstream pipeline / Step 3 preview
        # apply their own processing.
        self._sim_pat_array = pat_np
        self._sim_gen_btn.setEnabled(True)
        eu = tuple(self._sim_euler_deg)
        self._sim_status.setText(f"Done — {pat_np.shape[0]}×{pat_np.shape[1]} px.")
        # Push the new sim into the main step-4 reference-pattern preview
        # (shared widget between real & simulated modes).
        self._load_ref_pattern_preview()

        # Run the sim through the Step 3 pipeline (high-pass / low-pass / gamma,
        # mask disabled) so display & ZNSSD compare like-for-like against the
        # processed experimental pattern.
        wiz      = self.wizard()
        up2_path = wiz.field("up2_path") if wiz else ""
        sim_proc = self._apply_step3_processing(pat_np, up2_path)
        exp_proc = self._sim_exp_pat_proc if self._sim_exp_pat_proc is not None else self._sim_exp_pat

        # White, slightly larger titles so they're readable on the dark dialog
        # background — the matplotlib default is black.
        _title_kw = dict(fontsize=12, color="white", fontweight="bold")

        # Left: experimental (or placeholder if no UP2 loaded)
        self._sim_pat_ax_exp.clear()
        if exp_proc is not None:
            self._sim_pat_ax_exp.imshow(exp_proc, cmap="gray", origin="upper")
            self._sim_pat_ax_exp.set_title(
                f"Experimental (Step 3 processed)\n"
                f"row={self.ref_row.value()}, col={self.ref_col.value()}", **_title_kw
            )
        else:
            self._sim_pat_ax_exp.text(0.5, 0.5, "No UP2 file\navailable",
                                      ha="center", va="center", transform=self._sim_pat_ax_exp.transAxes,
                                      fontsize=11, color="white")
            self._sim_pat_ax_exp.set_title("Experimental", **_title_kw)
        self._sim_pat_ax_exp.axis("off")

        # Centre: simulated (Step 3 processed)
        self._sim_pat_ax_sim.clear()
        self._sim_pat_ax_sim.imshow(sim_proc, cmap="gray", origin="upper")
        self._sim_pat_ax_sim.set_title(
            f"Simulated (Step 3 processed)\n"
            f"φ₁={eu[0]:.2f}°  Φ={eu[1]:.2f}°  φ₂={eu[2]:.2f}°", **_title_kw
        )
        self._sim_pat_ax_sim.axis("off")

        # Right: ZNSSD residual + similarity metrics
        # Both patterns went through Step 3 first, so global brightness/contrast
        # offsets are already removed by the filter pipeline.  The z-normalisation
        # below removes any residual scale/offset, so the residual map shows
        # only structural mismatches.  This matches the IC-GN criterion exactly.
        self._sim_pat_ax_diff.clear()
        metrics_str = None
        if exp_proc is not None:
            exp_r = exp_proc
            sim_r = sim_proc
            # Resize sim to match exp if shapes differ
            if exp_r.shape != sim_r.shape:
                from PIL import Image as _PIL
                sim_r = np.array(
                    _PIL.fromarray(sim_r).resize((exp_r.shape[1], exp_r.shape[0]),
                                                 resample=_PIL.BILINEAR)
                )

            def _zn(x):
                x = x.astype(np.float64)
                x = x - x.mean()
                n = np.linalg.norm(x)
                return x / (n + 1e-12)
            exp_zn = _zn(exp_r)
            sim_zn = _zn(sim_r)
            residual = sim_zn - exp_zn

            # SSIM — locally brightness/contrast invariant; range [−1, 1].
            try:
                from skimage.metrics import structural_similarity as _ssim
                ssim_val = float(_ssim(exp_r, sim_r, data_range=1.0))
            except ImportError:
                ssim_val = None

            # ZNCC — global cosine similarity of the normalized patterns.
            zncc_val = float(np.dot(exp_zn.ravel(), sim_zn.ravel()))
            # ZNSSD scalar — Σ residual² ∈ [0, 4]; matches IC-GN's loss.
            znssd_val = float(np.dot(residual.ravel(), residual.ravel()))

            ssim_str = f"SSIM = {ssim_val:.4f}" if ssim_val is not None else "SSIM unavailable"
            metrics_str = f"{ssim_str}   ZNCC = {zncc_val:.4f}   ZNSSD = {znssd_val:.4f}"

            # FROZEN colour scale: vabs is computed only on the FIRST click and
            # reused on every subsequent click.  Same for the colorbar object
            # itself — matplotlib otherwise stacks a new colorbar axis on each
            # call, so by the 4th click you'd have 4 colorbars side-by-side.
            if getattr(self, "_sim_pat_residual_vabs", None) is None:
                self._sim_pat_residual_vabs = max(
                    float(np.percentile(np.abs(residual), 98)), 1e-9
                )
            vabs = self._sim_pat_residual_vabs
            im = self._sim_pat_ax_diff.imshow(
                residual, cmap="RdBu_r", origin="upper", vmin=-vabs, vmax=+vabs,
            )
            if getattr(self, "_sim_pat_residual_cbar", None) is None:
                cbar = self._sim_pat_fig.colorbar(
                    im, ax=self._sim_pat_ax_diff, fraction=0.046, pad=0.04
                )
                cbar.ax.tick_params(labelcolor="white", labelsize=10)
                self._sim_pat_residual_cbar = cbar
            self._sim_pat_ax_diff.set_title(
                f"ZNSSD residual  (Sim − Exp, z-normalized)\n{metrics_str}", **_title_kw
            )
        else:
            self._sim_pat_ax_diff.text(0.5, 0.5, "No experimental\npattern available",
                                       ha="center", va="center",
                                       transform=self._sim_pat_ax_diff.transAxes,
                                       fontsize=11, color="white")
            self._sim_pat_ax_diff.set_title("Difference", **_title_kw)
        self._sim_pat_ax_diff.axis("off")

        self._sim_pat_fig.tight_layout(pad=0.5)
        self._sim_pat_canvas.draw()
        status_metrics = f"  |  {metrics_str}" if metrics_str else ""
        self._sim_dlg_status.setText(
            f"Experimental: row={self.ref_row.value()}, col={self.ref_col.value()}  |  "
            f"Simulated: {pat_np.shape[0]}×{pat_np.shape[1]} px{status_metrics}"
        )

        # ── Update the flicker view with the latest pair ───────────────────
        # _sim_flicker_exp/sim store the up-to-date processed patterns; the
        # timer pulls from these on each tick.  We also force one frame
        # immediately so the user sees something even before they press Play.
        if exp_proc is not None:
            self._sim_flicker_exp = exp_proc
            self._sim_flicker_sim = sim_proc
            self._render_flicker_frame(force_resize=True)
        else:
            self._sim_flicker_exp = None
            self._sim_flicker_sim = None
            self._sim_flicker_ax.clear()
            self._sim_flicker_ax.text(
                0.5, 0.5, "No experimental pattern\navailable",
                ha="center", va="center",
                transform=self._sim_flicker_ax.transAxes,
                fontsize=12, color="white",
            )
            self._sim_flicker_ax.axis("off")
            self._sim_flicker_canvas.draw_idle()

        if self._show_compare_dialog:
            self._sim_dialog.show()
            self._show_compare_dialog = False

    # ── Flicker animation ─────────────────────────────────────────────────────

    def _render_flicker_frame(self, force_resize: bool = False):
        """Draw whichever pattern (sim or exp) is currently selected by
        self._sim_flicker_show_sim onto the flicker canvas.

        We avoid clearing the axis on every tick — instead we reuse a single
        AxesImage handle and just swap its data, which is dramatically
        cheaper than a full clear+imshow per frame and lets the flicker
        run smoothly even at 50 ms / frame.
        """
        if self._sim_flicker_exp is None or self._sim_flicker_sim is None:
            return
        img = self._sim_flicker_sim if self._sim_flicker_show_sim else self._sim_flicker_exp
        label = "SIMULATED" if self._sim_flicker_show_sim else "EXPERIMENTAL"
        color = "#4caf50" if self._sim_flicker_show_sim else "#ff9800"

        if self._sim_flicker_im is None or force_resize:
            self._sim_flicker_ax.clear()
            self._sim_flicker_im = self._sim_flicker_ax.imshow(
                img, cmap="gray", origin="upper",
            )
            self._sim_flicker_ax.axis("off")
            self._sim_flicker_title = self._sim_flicker_ax.set_title(
                label, fontsize=16, color=color, fontweight="bold",
            )
        else:
            self._sim_flicker_im.set_data(img)
            self._sim_flicker_title.set_text(label)
            self._sim_flicker_title.set_color(color)

        self._sim_flicker_canvas.draw_idle()

    def _on_flicker_toggle(self, checked: bool):
        """Play/Pause button callback."""
        if checked and self._sim_flicker_exp is not None:
            self._sim_flicker_timer.start(self._sim_flicker_speed.value())
            self._sim_flicker_play_btn.setText("⏸ Pause flicker")
            self._sim_flicker_frame_lbl.setStyleSheet(
                "color: #4caf50; font-weight: bold; font-size: 11px;"
            )
        else:
            self._sim_flicker_timer.stop()
            self._sim_flicker_play_btn.setText("▶ Play flicker")
            self._sim_flicker_frame_lbl.setText("(flicker stopped)")
            self._sim_flicker_frame_lbl.setStyleSheet(
                "color: gray; font-style: italic; font-size: 11px;"
            )

    def _on_flicker_speed_changed(self, value: int):
        """Speed slider callback — also re-arms the QTimer if currently running."""
        self._sim_flicker_speed_lbl.setText(f"{value} ms / frame")
        if self._sim_flicker_timer.isActive():
            self._sim_flicker_timer.start(int(value))

    def _on_flicker_tick(self):
        """One frame of the flicker — flip which pattern is shown, redraw."""
        self._sim_flicker_show_sim = not self._sim_flicker_show_sim
        self._render_flicker_frame(force_resize=False)
        self._sim_flicker_frame_lbl.setText(
            "showing: SIMULATED" if self._sim_flicker_show_sim
            else "showing: EXPERIMENTAL"
        )

    def _on_sim_error(self, msg: str):
        self._sim_gen_btn.setEnabled(True)
        self._sim_status.setText(f"Error: {msg.splitlines()[-1]}")

    def _open_sim_tuner(self):
        mp_path = (self.wizard().field("master_pattern_path") or "").strip()
        if not mp_path or not os.path.exists(mp_path):
            self._sim_status.setText("Select a master pattern file before opening the tuner.")
            return

        wiz      = self.wizard()
        geom     = wiz.geometry_page.get_params()
        pc_edax  = geom["pc_edax"]
        det_shape = (geom["pat_h"], geom["pat_w"])
        euler_deg = tuple(self._sim_euler_deg)

        # Load raw and processed experimental patterns if not already cached
        exp_pat_proc = None
        up2_path = wiz.field("up2_path")
        if up2_path and os.path.exists(up2_path):
            try:
                import Data
                pat_obj = Data.UP2(up2_path)
                idx = self.ref_row.value() * geom["cols"] + self.ref_col.value()
                if idx < pat_obj.nPatterns:
                    if self._sim_exp_pat is None:
                        raw = pat_obj.read_pattern(idx, process=False).astype(np.float32)
                        lo, hi = raw.min(), raw.max()
                        self._sim_exp_pat = (raw - lo) / (hi - lo + 1e-9)
                    # Always load a freshly processed version using current Step 3 params
                    p = wiz.processing_page.get_params()
                    mask_type = p["mask_type"] if p["mask_type"] != "None" else None
                    pat_obj.set_processing(
                        low_pass_sigma          = p["low_pass_sigma"],
                        high_pass_sigma         = p["high_pass_sigma"],
                        truncate_std_scale      = 3.0,
                        mask_type               = mask_type,
                        center_cross_half_width = 6,
                        flip_x                  = p["flip_x"],
                        gamma                   = p.get("gamma", 0.8),
                    )
                    proc = pat_obj.read_pattern(idx, process=True).astype(np.float32)
                    lo, hi = proc.min(), proc.max()
                    exp_pat_proc = (proc - lo) / (hi - lo + 1e-9)
            except Exception:
                pass

        dlg = SimTunerDialog(
            master_path     = mp_path,
            euler_deg       = euler_deg,
            pc_edax         = pc_edax,
            det_shape       = det_shape,
            det_tilt_deg    = geom["det_tilt"],
            sample_tilt_deg = geom["tilt"],
            exp_pat         = self._sim_exp_pat,
            exp_pat_proc    = exp_pat_proc,
            parent          = self,
        )
        dlg.applied_signal.connect(self._on_tuner_applied)
        dlg.show()

    def _on_tuner_applied(self, euler_deg: tuple, pc_edax: tuple, tilt_deg: float):
        """Push values from the tuner back into the reference page."""
        self._set_sim_euler_deg(euler_deg)
        geom_page = self.wizard().geometry_page
        geom_page.pc_x.setValue(pc_edax[0])
        geom_page.pc_y.setValue(pc_edax[1])
        geom_page.pc_z.setValue(pc_edax[2])
        geom_page.tilt.setValue(float(tilt_deg))
        self._sim_status.setText(
            f"Tuner applied — φ₁={euler_deg[0]:.2f}°  "
            f"Φ={euler_deg[1]:.2f}°  φ₂={euler_deg[2]:.2f}°  |  "
            f"PC: ({pc_edax[0]:.4f}, {pc_edax[1]:.4f}, {pc_edax[2]:.4f})  |  "
            f"tilt: {tilt_deg:.2f}°"
        )

    # ── PC / Euler refinement ─────────────────────────────────────────────────

    def _run_pc_euler_refine(self):
        wiz = self.wizard()
        up2_path = wiz.field("up2_path")
        if not up2_path or not os.path.exists(up2_path):
            QMessageBox.warning(self, "Missing UP2",
                                "Load a UP2 file first (Step 1).")
            return

        mp_path = (self.wizard().field("master_pattern_path") or "").strip()
        if not mp_path or not os.path.exists(mp_path):
            QMessageBox.warning(self, "Missing master pattern",
                                "Set a valid master pattern path on Step 1 first.")
            return

        geom = wiz.geometry_page.get_params()
        p    = wiz.processing_page.get_params()

        pat_idx    = self.ref_row.value() * geom["cols"] + self.ref_col.value()
        euler_init = np.deg2rad(list(self._sim_euler_deg))
        pc_init    = tuple(geom["pc_edax"])

        save_dir = os.path.join(os.path.dirname(up2_path), "pc_euler_refine")

        # Log output goes directly into the embedded box on the refine
        # settings dialog — no standalone progress window any more.
        log_box = self._refine_log_box
        log_box.clear()

        _sim_hp = self._refine_sim_hp_sigma.value()
        _sim_lp = self._refine_sim_lp_sigma.value()
        _sim_g  = self._refine_sim_gamma.value()
        self._refine_worker = PcEulerRefineWorker(
            up2_path               = up2_path,
            pat_idx                = pat_idx,
            processing_params      = p,
            master_pattern_path    = mp_path,
            euler_angles_init      = euler_init,
            pc_init                = pc_init,
            sample_tilt_deg        = geom["tilt"],
            detector_tilt_deg      = geom["det_tilt"],
            max_iter               = self._refine_max_iter.value(),
            n_restarts             = self._refine_n_restarts.value(),
            restart_rotvec_std_deg = self._refine_restart_rotvec_std.value(),
            restart_pc_std         = self._refine_restart_pc_std.value(),
            restart_seed           = self._refine_restart_seed.value(),
            symmetry_restarts      = self._refine_symmetry_restarts.isChecked(),
            laue_group_id          = self._refine_laue_group.currentIndex() + 1,
            save_dir               = save_dir,
            sim_high_pass_sigma    = _sim_hp if _sim_hp > 0.0 else None,
            sim_low_pass_sigma     = _sim_lp if _sim_lp > 0.0 else None,
            sim_gamma              = _sim_g  if _sim_g  > 0.0 else None,
        )

        def _on_log(msg):
            log_box.append(msg)
            sb = log_box.verticalScrollBar()
            sb.setValue(sb.maximum())

        def _on_done(euler_opt, pc_opt):
            euler_deg = np.degrees(euler_opt)
            # Stash the result as pending — actual apply happens when the
            # user clicks Finish & Apply on the refine dialog.
            self._refine_pending_euler_deg = (float(euler_deg[0]),
                                              float(euler_deg[1]),
                                              float(euler_deg[2]))
            self._refine_pending_pc = (float(pc_opt[0]),
                                       float(pc_opt[1]),
                                       float(pc_opt[2]))
            self._refine_status.setText(
                f"Refined — φ₁={euler_deg[0]:.3f}°  Φ={euler_deg[1]:.3f}°  "
                f"φ₂={euler_deg[2]:.3f}°  |  "
                f"PC: ({pc_opt[0]:.5f}, {pc_opt[1]:.5f}, {pc_opt[2]:.5f})"
            )
            self._refine_status.setStyleSheet("color: green; font-size: 11px;")
            log_box.append("\nRefinement complete — click Finish && Apply to keep the result, or Cancel to discard.")
            self._refine_run_btn.setEnabled(True)

        def _on_error(tb):
            log_box.append(f"\n[ERROR]\n{tb}")
            self._refine_run_btn.setEnabled(True)

        self._refine_worker.log_signal.connect(_on_log)
        self._refine_worker.done_signal.connect(_on_done)
        self._refine_worker.error_signal.connect(_on_error)
        self._refine_worker.start()
        self._refine_run_btn.setEnabled(False)

    # NOTE: The PC plane fit UI (button, modal, status label) used to live
    # here.  It was removed from the GUI per user request — the standalone
    # tool is still available in pc_plane_fit.py and gui_workers.PcPlaneFitWorker
    # if you want to re-wire it later.

    def get_params(self) -> dict:
        # small_strain now lives on Step 2 (ScanGeometryPage); nothing to add here.
        # Advanced sim-reference tweaks (spectral match, Tikhonov, rotate-90)
        # moved from Step 3 into Step 4's simulated section — surface them
        # for the pipeline regardless of which ref_mode is active so the
        # toggles still take effect for the experimental-reference path too.
        common = {
            "spectral_match_ref":         self._spectral_match_ref.isChecked(),
            "perspective_regularization": (
                float(self._tikhonov_lambda.value())
                if self._tikhonov_perspective.isChecked() else 0.0
            ),
            "rotate_patterns_90":         self._rotate_patterns_90.isChecked(),
        }
        if self._single_radio.isChecked():
            return {
                "ref_mode":     "single",
                "ref_position": (self.ref_row.value(), self.ref_col.value()),
                **common,
            }
        if self._sim_radio.isChecked():
            return {
                "ref_mode":            "simulated",
                "master_pattern_path": (self.wizard().field("master_pattern_path") or "").strip(),
                "euler_deg":           tuple(self._sim_euler_deg),
                "sim_pat_array":       self._sim_pat_array,
                **common,
            }
        return {
            "ref_mode":         "per_grain",
            "ref_pattern_set":  self._ref_pattern_set,
            **common,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Pattern Processing
# ─────────────────────────────────────────────────────────────────────────────

class _SquareCanvas(FigureCanvas):
    """FigureCanvas that always requests a square bounding box."""
    def hasHeightForWidth(self) -> bool:
        return True
    def heightForWidth(self, width: int) -> int:
        return width

class PatternProcessingPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 3 of 6 — Pattern Processing")
        self.setSubTitle(
            "Adjust the pre-processing parameters. The reference pattern updates "
            "live so you can see the effect before running."
        )

        self._preview_worker = None
        self._preview_timer  = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._run_preview)
        self._crop_rect    = None
        self._preview_shape = None

        # ── Controls (left panel, fixed width) ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        inner  = QWidget()
        ctrl   = QVBoxLayout(inner)
        ctrl.setSpacing(12)

        # Mask
        mask_group  = QGroupBox("Mask")
        mask_layout = QFormLayout()

        self.mask_type = QComboBox()
        self.mask_type.addItems(["None", "circular", "center_cross"])

        mask_layout.addRow("Mask type:", self.mask_type)
        mask_group.setLayout(mask_layout)
        ctrl.addWidget(mask_group)

        # Frequency filtering
        filt_group  = QGroupBox("Frequency Filtering")
        filt_layout = QFormLayout()

        self.high_pass = QDoubleSpinBox()
        self.high_pass.setRange(0, 200)
        self.high_pass.setValue(10.0)
        self.high_pass.setToolTip(
            "Sigma of the log-domain Gaussian high-pass background removal."
        )

        self.low_pass = QDoubleSpinBox()
        self.low_pass.setRange(0, 100)
        self.low_pass.setValue(1.0)

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 3.0)
        self.gamma.setValue(0.8)
        self.gamma.setSingleStep(0.05)
        self.gamma.setToolTip("Gamma < 1 brightens dark regions. Set to 1.0 to disable.")

        filt_layout.addRow("High-pass sigma:",      self.high_pass)
        filt_layout.addRow("Low-pass sigma:",       self.low_pass)
        filt_layout.addRow("Gamma:",                self.gamma)
        filt_group.setLayout(filt_layout)
        ctrl.addWidget(filt_group)

        self.flip_x = QCheckBox("Flip patterns vertically")

        # IC-GN region
        crop_group  = QGroupBox("IC-GN Region")
        crop_layout = QFormLayout()

        self.crop_fraction = QDoubleSpinBox()
        self.crop_fraction.setRange(0.1, 0.99)
        self.crop_fraction.setValue(0.9)
        self.crop_fraction.setSingleStep(0.05)
        self.crop_fraction.valueChanged.connect(self._update_crop_rect)

        crop_layout.addRow("Crop fraction:", self.crop_fraction)
        crop_layout.addRow(_note("Green outline shows the region the optimizer sees."))
        crop_group.setLayout(crop_layout)
        ctrl.addWidget(crop_group)

        # NOTE: the entire "Reference Pattern Preprocessing (override)" group
        # used to live here.  Its useful controls (spectral matching, Tikhonov
        # regularization, rotate-90 diagnostic) have been moved to the
        # Simulated-Reference settings on Step 4.  The per-component overrides
        # (high-pass / low-pass / gamma / mask) were redundant with the
        # existing "Sim …" overrides on Step 4 and have been dropped.

        # Advanced options group
        adv_group  = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()

        self._show_gradients = QCheckBox("Show gradients")
        self._show_gradients.setChecked(False)
        adv_layout.addWidget(self._show_gradients)

        adv_layout.addWidget(self.flip_x)

        adv_group.setLayout(adv_layout)
        ctrl.addWidget(adv_group)

        ctrl.addStretch()
        scroll.setWidget(inner)

        # ── Preview (right panel) ─────────────────────────────────────────────
        preview_widget = QWidget()
        preview_vbox   = QVBoxLayout(preview_widget)
        preview_vbox.setContentsMargins(6, 0, 0, 0)

        panels_row = QHBoxLayout()

        _fig_bg = THEME["surface_bg"]

        raw_box    = QGroupBox("Raw pattern")
        raw_vbox   = QVBoxLayout(raw_box)
        self._fig_raw  = Figure(facecolor=_fig_bg)
        self._fig_raw.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._ax_raw   = self._fig_raw.add_subplot(111)
        self._ax_raw.set_facecolor(_fig_bg)
        self._ax_raw.set_visible(False)
        self._canvas_raw = _SquareCanvas(self._fig_raw)
        self._canvas_raw.setStyleSheet(f"background-color: {_fig_bg};")
        self._canvas_raw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        raw_vbox.addWidget(self._canvas_raw)

        proc_box   = QGroupBox("Filtered pattern")
        proc_vbox  = QVBoxLayout(proc_box)
        self._fig_proc = Figure(facecolor=_fig_bg)
        self._fig_proc.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._ax_proc  = self._fig_proc.add_subplot(111)
        self._ax_proc.set_facecolor(_fig_bg)
        self._ax_proc.set_visible(False)
        self._canvas_proc = _SquareCanvas(self._fig_proc)
        self._canvas_proc.setStyleSheet(f"background-color: {_fig_bg};")
        self._canvas_proc.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        proc_vbox.addWidget(self._canvas_proc)

        panels_row.addWidget(raw_box)
        panels_row.addWidget(proc_box)

        # Gradient row (hidden until checkbox is ticked)
        self._grad_widget = QGroupBox("Gradients  (Gx · Gy · |G|)")
        grad_vbox = QVBoxLayout(self._grad_widget)
        self._fig_grad = Figure(tight_layout=True, facecolor=_fig_bg)
        self._ax_gx  = self._fig_grad.add_subplot(131)
        self._ax_gy  = self._fig_grad.add_subplot(132)
        self._ax_gm  = self._fig_grad.add_subplot(133)
        for ax in (self._ax_gx, self._ax_gy, self._ax_gm):
            ax.set_facecolor(_fig_bg)
            ax.axis("off")
        self._canvas_grad = FigureCanvas(self._fig_grad)
        self._canvas_grad.setStyleSheet(f"background-color: {_fig_bg};")
        self._canvas_grad.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._canvas_grad.setMaximumHeight(220)
        grad_vbox.addWidget(self._canvas_grad)
        self._grad_widget.setVisible(False)

        self._prev_status = QLabel("Preview loads when the page opens.")
        self._prev_status.setStyleSheet("color: gray;")
        self._prev_status.setWordWrap(True)

        preview_vbox.addLayout(panels_row, stretch=1)
        preview_vbox.addWidget(self._grad_widget)
        preview_vbox.addWidget(self._prev_status)

        def _toggle_grad_panel(checked):
            self._grad_widget.setVisible(checked)
            if checked and self._ax_gx.get_images():
                self._canvas_grad.draw()

        self._show_gradients.toggled.connect(_toggle_grad_panel)

        # ── Assemble ──────────────────────────────────────────────────────────
        outer = QHBoxLayout()
        outer.addWidget(scroll)
        outer.addWidget(preview_widget, stretch=1)
        self.setLayout(outer)

        # Connect all controls → debounced preview
        for w in (self.high_pass, self.low_pass, self.gamma):
            w.valueChanged.connect(self._schedule_preview)
        self.mask_type.currentIndexChanged.connect(self._schedule_preview)
        self.mask_type.currentIndexChanged.connect(self._update_crop_rect)
        # Circle mode allows crop_fraction = 1.0 (full inscribed disc); rect
        # mode must stay strictly below 1.  Swap the spinbox max each time
        # the mask type changes.
        self.mask_type.currentIndexChanged.connect(self._update_crop_fraction_max)
        self.flip_x.toggled.connect(self._schedule_preview)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initializePage(self):
        self._schedule_preview()

    # ── Preview ───────────────────────────────────────────────────────────────

    def _schedule_preview(self):
        self._preview_timer.start()

    def _run_preview(self):
        wiz      = self.wizard()
        up2_path = wiz.field("up2_path")

        ref_params = wiz.reference_page.get_params()

        # ── Simulated reference: apply filters to the generated array directly ─
        if ref_params.get("ref_mode") == "simulated":
            sim_pat = ref_params.get("sim_pat_array")
            if sim_pat is None:
                self._prev_status.setText(
                    "No simulated pattern yet — go back to Step 4 and click Generate."
                )
                return
            self._prev_status.setText("Processing simulated pattern…")
            try:
                import Data
                p = self.get_params()
                pat_obj = Data.UP2(up2_path) if up2_path and os.path.exists(up2_path) else None
                if pat_obj is not None:
                    mask_type = p["mask_type"] if p["mask_type"] != "None" else None
                    pat_obj.set_processing(
                        low_pass_sigma          = p["low_pass_sigma"],
                        high_pass_sigma         = p["high_pass_sigma"],
                        truncate_std_scale      = 3.0,
                        mask_type               = mask_type,
                        center_cross_half_width = 6,
                        flip_x                  = p["flip_x"],
                        gamma                   = p.get("gamma", 0.8),
                    )
                    processed = pat_obj.process_pattern(sim_pat.copy())
                else:
                    processed = sim_pat
                self._on_preview_done(sim_pat, processed)
                self._prev_status.setText("Simulated reference pattern (Step 4).")
            except Exception as exc:
                self._prev_status.setText(f"Error processing simulated pattern: {exc}")
            return

        # ── Experimental reference ────────────────────────────────────────────
        if not up2_path or not os.path.exists(up2_path):
            self._prev_status.setText("No UP2 file loaded — go back to Step 1.")
            return

        try:
            geom = wiz.geometry_page.get_params()
            if ref_params.get("ref_mode") == "per_grain":
                rps = ref_params.get("ref_pattern_set")
                pat_idx = rps[0].ref_pat_idx if rps and len(rps) > 0 else 0
            else:
                ref_pos = ref_params["ref_position"]
                pat_idx = int(np.ravel_multi_index(ref_pos, (geom["rows"], geom["cols"])))
        except Exception:
            pat_idx = 0

        # Cancel any in-flight worker before starting a new one
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.done_signal.disconnect()
            self._preview_worker.error_signal.disconnect()

        # The previous override-swap logic (for the now-removed reference
        # preprocessing override block) has been deleted.  The preview now
        # always reflects the main Step 3 preprocessing settings.
        preview_params = self.get_params()
        self._prev_status.setText("Processing…")

        self._preview_worker = PatternPreviewWorker(up2_path, pat_idx, preview_params)
        self._preview_worker.done_signal.connect(self._on_preview_done)
        self._preview_worker.error_signal.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_done(self, raw, processed):
        self._ax_raw.set_visible(True)
        self._ax_raw.clear()
        self._ax_raw.imshow(raw, cmap="gray", origin="upper")
        self._ax_raw.axis("off")
        self._canvas_raw.draw()

        self._ax_proc.set_visible(True)
        self._ax_proc.clear()
        self._ax_proc.imshow(processed, cmap="gray", origin="upper")
        self._ax_proc.axis("off")

        # Store shape and draw crop-fraction outline
        self._preview_shape = processed.shape
        self._crop_rect = None
        self._update_crop_rect()

        self._canvas_proc.draw()

        # Compute and cache gradients (cheap — always done so the panel is
        # ready instantly when the checkbox is toggled)
        pat = processed.astype(np.float32)
        gy, gx = np.gradient(pat)
        gmag   = np.sqrt(gx**2 + gy**2)
        vmax_x = np.percentile(np.abs(gx), 99) or 1e-9
        vmax_y = np.percentile(np.abs(gy), 99) or 1e-9
        vmax_m = np.percentile(gmag, 99)        or 1e-9

        for ax, data, cmap, vmin, vmax, title in [
            (self._ax_gx, gx,   "RdBu",   -vmax_x, vmax_x, "Gx"),
            (self._ax_gy, gy,   "RdBu",   -vmax_y, vmax_y, "Gy"),
            (self._ax_gm, gmag, "inferno", 0,       vmax_m, "|G|"),
        ]:
            ax.clear()
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(title, color="white", fontsize=9)
            ax.axis("off")

        self._fig_grad.tight_layout()
        if self._grad_widget.isVisible():
            self._canvas_grad.draw()

        self._prev_status.setText("")

    def _on_preview_error(self, msg: str):
        self._prev_status.setText("Preview error — check console.")
        print("\n--- Preview error ---\n" + msg)

    def _update_crop_fraction_max(self):
        """Allow crop_fraction up to 1.0 in circle mode; clamp to 0.99 in
        rect mode (the IC-GN solver rejects rect with crop_fraction >= 1)."""
        if self.mask_type.currentText() == "circular":
            self.crop_fraction.setRange(0.1, 1.0)
        else:
            self.crop_fraction.setRange(0.1, 0.99)

    def _update_crop_rect(self):
        """Draw / redraw the green crop-fraction outline on the filtered
        pattern.  Switches to a circle when mask_type is 'circular' so the
        overlay matches the actual IC-GN evaluation region."""
        if not self._ax_proc.get_visible() or self._preview_shape is None:
            return
        import matplotlib.patches as _mp
        if self._crop_rect is not None:
            try:
                self._crop_rect.remove()
            except Exception:
                pass
            self._crop_rect = None
        H, W = self._preview_shape
        cf = self.crop_fraction.value()
        if self.mask_type.currentText() == "circular":
            # Disc inscribed in the pattern, radius = cf * min(H, W) / 2,
            # centered on the geometric image center — matches the IC-GN
            # subset_shape_kind="circle" path in get_homography_cpu.optimize.
            radius = cf * min(H, W) / 2.0
            cx = W / 2.0 - 0.5
            cy = H / 2.0 - 0.5
            self._crop_rect = _mp.Circle(
                (cx, cy), radius,
                linewidth=1.5, edgecolor="#a6e3a1", facecolor="none", zorder=5,
            )
        else:
            margin_x = (1.0 - cf) / 2.0 * W
            margin_y = (1.0 - cf) / 2.0 * H
            self._crop_rect = _mp.Rectangle(
                (margin_x - 0.5, margin_y - 0.5), cf * W, cf * H,
                linewidth=1.5, edgecolor="#a6e3a1", facecolor="none", zorder=5,
            )
        self._ax_proc.add_patch(self._crop_rect)
        self._canvas_proc.draw_idle()

    # ── Params ────────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        # ref_preprocessing / spectral_match_ref / perspective_regularization /
        # rotate_patterns_90 used to be set here.  They've moved to
        # ReferencePatternPage.get_params (Step 4 → Simulated Reference settings).
        return {
            "high_pass_sigma": self.high_pass.value(),
            "low_pass_sigma":  self.low_pass.value(),
            "gamma":           self.gamma.value(),
            "flip_x":          self.flip_x.isChecked(),
            "mask_type":       self.mask_type.currentText(),
            "crop_fraction":   self.crop_fraction.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Optimization + Run
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationRunPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 6 of 6 — Optimization & Run")
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

        # Detect the number of logical CPUs (threads) on this machine and
        # cap the spinbox so the user can't oversubscribe.
        _n_cpu = os.cpu_count() or 1
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, _n_cpu)
        self.n_jobs.setValue(min(8, _n_cpu))
        self.n_jobs.setToolTip(
            f"Number of CPU threads the pipeline can use in parallel.\n"
            f"This machine reports {_n_cpu} logical CPU{'s' if _n_cpu != 1 else ''}."
        )

        self.init_type = QComboBox()
        self.init_type.addItems(["none", "partial", "full"])

        # Optimizer selection: IC-GN homography (default) or reversed-role IC-GN
        self.optimizer_method = QComboBox()
        self.optimizer_method.addItems([
            "IC-GN (homography, default)",
            "IC-GN reversed roles (experimental)",
        ])
        self.optimizer_method.setToolTip(
            "IC-GN runs the iterative inverse-compositional Gauss-Newton solver "
            "(get_homography_cpu).\n"
            "IC-GN reversed roles swaps which pattern provides the gradients / "
            "Jacobian / Hessian: each per-pattern image is treated as the "
            "reference (recomputed in every worker), and the designated "
            "reference pattern is treated as the target "
            "(get_homography_cpu_reversed).  The returned homographies are "
            "auto-inverted before downstream strain so the convention matches "
            "standard IC-GN.  Slower than standard IC-GN because the per-"
            "pattern Cholesky factor is no longer amortised."
        )

        opt_layout.addRow("Max iterations:", self.max_iter)
        opt_layout.addRow("CPU threads:", self.n_jobs)
        opt_layout.addRow("Init type:", self.init_type)
        opt_layout.addRow("Optimizer:", self.optimizer_method)
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
            f"font-size: 14px; font-weight: bold; "
            f"background-color: {THEME['run_btn_bg']}; "
            f"color: {THEME['run_btn_text']}; border-radius: 6px;"
        )
        self._run_btn.clicked.connect(self._start_run)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._status_label)

        # ── Log ───────────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Log:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._log)

        # ── Visualize button (shown after a successful run) ────────────────────
        self._vis_btn = QPushButton("Visualize Results")
        self._vis_btn.setFixedHeight(38)
        self._vis_btn.setStyleSheet(
            f"font-size: 13px; font-weight: bold; "
            f"background-color: {THEME['accent']}; color: #000; border-radius: 5px;"
        )
        self._vis_btn.setVisible(False)
        self._vis_btn.clicked.connect(self._open_vis_dialog)
        layout.addWidget(self._vis_btn)

        self._run_params = {}
        self.setLayout(layout)

    def initializePage(self):
        wiz  = self.wizard()
        geom = wiz.geometry_page.get_params()
        roi  = wiz.roi_page.get_params()
        ref  = wiz.reference_page.get_params()
        proc = wiz.processing_page.get_params()

        roi_str = (
            f"row {roi['roi_slice'][0].start}:{roi['roi_slice'][0].stop}, "
            f"col {roi['roi_slice'][1].start}:{roi['roi_slice'][1].stop}"
            if roi["roi_slice"] is not None else "full scan"
        )

        lines = [
            f"Run name         : {wiz.field('run_name')}",
            f"UP2 file         : {wiz.field('up2_path')}",
            f"ANG file         : {wiz.field('ang_path')}",
            f"Output folder    : {wiz.field('output_dir')}",
            f"",
            f"ROI              : {roi_str}",
            f"Reference mode   : {ref.get('ref_mode', 'single')}" + (
                f"  (row={ref['ref_position'][0]}, col={ref['ref_position'][1]})"
                if ref.get("ref_mode") == "single"
                else f"  (φ₁={ref['euler_deg'][0]:.2f}° Φ={ref['euler_deg'][1]:.2f}° φ₂={ref['euler_deg'][2]:.2f}°)"
                if ref.get("ref_mode") == "simulated"
                else f"  ({len(ref.get('ref_pattern_set') or [])} grains)"
            ),
            f"Sample tilt      : {geom['tilt']} °",
            f"Detector tilt    : {geom['det_tilt']} °",
            f"Pattern center   : x*={geom['pc_edax'][0]:.5f}  y*={geom['pc_edax'][1]:.5f}  z*={geom['pc_edax'][2]:.5f}",
            f"Rows × Columns   : {geom['rows']} × {geom['cols']}",
            f"Step size        : {geom['step_size']} µm",
            f"",
            f"High-pass σ      : {proc['high_pass_sigma']}",
            f"Flip patterns    : {proc['flip_x']}",
            f"Mask type        : {proc['mask_type']}",
            f"",
            f"Scan strategy    : {geom['scan_strategy']}",
            f"PC correction    : {'yes' if geom['apply_pc_correction'] else 'no'}",
        ]
        self._summary.setPlainText("\n".join(lines))

        # Reset state if the user went Back and returned
        self._done = False
        self._run_btn.setEnabled(True)
        self._status_label.setText("")
        self._vis_btn.setVisible(False)

    def _start_run(self):
        wiz    = self.wizard()
        params = {}
        params.update(wiz.geometry_page.get_params())
        params.update(wiz.reference_page.get_params())
        params.update(wiz.processing_page.get_params())
        # Optimizer selection: "icgn" (default) or "icgn_reversed"
        _opt_text = self.optimizer_method.currentText()
        if "reversed" in _opt_text:
            opt_choice = "icgn_reversed"
        else:
            opt_choice = "icgn"

        params.update({
            "max_iter":      self.max_iter.value(),
            "n_jobs":        self.n_jobs.value(),
            "init_type":     self.init_type.currentText(),
            "optimizer":     opt_choice,
            "up2":           wiz.field("up2_path"),
            "ang":           wiz.field("ang_path"),
            "component":     wiz.field("run_name"),
            "output_dir":    wiz.field("output_dir"),
            "date":          datetime.date.today().strftime("%B_%d_%Y"),
        })

        params.update(wiz.roi_page.get_params())

        # Per-grain mode needs the grain_ids array from the ROI page
        if params.get("ref_mode") == "per_grain":
            params["_grain_ids"] = getattr(wiz.roi_page, "_grain_ids", None)

        # Single-reference grain ROI: build exact pixel mask so non-grain pixels
        # inside the bounding box are NaN'd out after optimization
        if params.get("_roi_grain_id") is not None:
            grain_ids = getattr(wiz.roi_page, "_grain_ids", None)
            if grain_ids is not None:
                params["_roi_grain_mask"] = (grain_ids == params["_roi_grain_id"])

        self._run_params = params   # keep for the vis dialog

        self._run_btn.setEnabled(False)
        self._vis_btn.setVisible(False)
        self._log.clear()
        self._status_label.setText("Running…")
        self._status_label.setStyleSheet(f"color: {THEME['warning']}; font-weight: bold;")

        wiz = self.wizard()
        if wiz and wiz.ang_data is not None and wiz.ang_loaded_path == params.get("ang"):
            params["_ang_data"] = wiz.ang_data

        self._worker = PipelineWorker(params)
        self._worker.log_signal.connect(self._append_log)
        self._worker.done_signal.connect(self._on_done)
        self._worker.start()

    def _append_log(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_done(self, success: bool, msg: str):
        if success:
            self._status_label.setText("Done — results saved successfully.")
            self._status_label.setStyleSheet(f"color: {THEME['success']}; font-weight: bold;")
            self._vis_btn.setVisible(True)
            self._done = True
            self.completeChanged.emit()
        else:
            self._status_label.setText("Error — see log for details.")
            self._status_label.setStyleSheet(f"color: {THEME['error']}; font-weight: bold;")
            self._log.append("\n--- TRACEBACK ---\n" + msg)
            self._run_btn.setEnabled(True)

    def _open_vis_dialog(self):
        p   = self._run_params
        roi = p.get("roi_slice", None)

        if roi is not None:
            eff_rows = roi[0].stop - roi[0].start
            eff_cols = roi[1].stop - roi[1].start
        else:
            eff_rows = p.get("rows", 1)
            eff_cols = p.get("cols", 1)

        comp = p.get("component", "run")
        date = p.get("date", "")
        npy_path = os.path.join(
            p.get("output_dir", ""),
            f"{comp}_homographies_{date}.npy",
        )
        npz_path = os.path.join(
            p.get("output_dir", ""),
            f"{comp}_results_{date}.npy",
        )

        vis_params = {
            "npz_path":            npz_path,
            "npy_path":            npy_path,
            "ang":                 p.get("ang", ""),        # pre-fill ang path (already loaded in step 1)
            "save_folder":         p.get("output_dir", ""),
            "rows":                eff_rows,
            "cols":                eff_cols,
            "roi_slice":           p.get("roi_slice", None), # needed so VisWorker slices ang quats correctly
            "full_rows":           p.get("rows", eff_rows),  # full scan shape (before ROI)
            "full_cols":           p.get("cols", eff_cols),
            "pat_h":               p.get("pat_h", 512),
            "pat_w":               p.get("pat_w", 512),
            "pc_edax":             p.get("pc_edax", (0.5, 0.5, 0.5)),
            "tilt":                p.get("tilt", 70.0),
            "det_tilt":            p.get("det_tilt", 10.0),
            "identity_rotation":   p.get("identity_rotation", False),
            "step_size":           p.get("step_size", 1.0),
            "pixel_size":          p.get("pixel_size", 1.0),
            "scan_strategy":       p.get("scan_strategy", "lower_left"),
            "apply_pc_correction": p.get("apply_pc_correction", False),
            "tfbc_use_single_euler": p.get("tfbc_use_single_euler", False),
            "tfbc_euler_deg":        p.get("tfbc_euler_deg", (0.0, 0.0, 0.0)),
            "small_strain":          p.get("small_strain", False),
        }

        dlg = VisualizationDialog(vis_params, parent=self)
        dlg.show()

    def isComplete(self) -> bool:
        return self._done

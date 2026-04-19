"""
gui_pages.py — all QWizardPage subclasses for the DIC-HREBSD wizard.

  LoadFilesPage         — Step 1: select UP2, ANG, output folder
  ScanGeometryPage      — Step 2: tilts, PC, detector / scan geometry
  ROISelectionPage      — Step 3: grain segmentation + region of interest
  ReferencePatternPage  — Step 4: pick and preview the reference pattern
  PatternProcessingPage — Step 5: frequency filters, mask, CLAHE, flip
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
    QRadioButton, QButtonGroup,
    QTextEdit, QWidget, QScrollArea,
    QSizePolicy, QSplitter, QDialog,
)
from PyQt6.QtCore import Qt, QTimer

from gui_theme import THEME, _make_browse_row, _make_browse_dir, _note
from gui_workers import PipelineWorker, IPFWorker, SegmentWorker, PatternPreviewWorker, AngLoaderWorker
from gui_visualization import VisualizationDialog
from gui_sweep import ParameterSweepDialog


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

        self._struct_lbl = QLabel("cubic")
        mat_layout.addRow("Structure:", self._struct_lbl)

        self._C11 = QDoubleSpinBox()
        self._C11.setRange(0, 9999); self._C11.setDecimals(1)
        self._C11.setSuffix(" GPa"); self._C11.setValue(165.7)
        self._C11.setReadOnly(True); self._C11.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)

        self._C12 = QDoubleSpinBox()
        self._C12.setRange(0, 9999); self._C12.setDecimals(1)
        self._C12.setSuffix(" GPa"); self._C12.setValue(63.9)
        self._C12.setReadOnly(True); self._C12.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)

        self._C44 = QDoubleSpinBox()
        self._C44.setRange(0, 9999); self._C44.setDecimals(1)
        self._C44.setSuffix(" GPa"); self._C44.setValue(79.6)
        self._C44.setReadOnly(True); self._C44.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)

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

        # ── Visualize existing results ─────────────────────────────────────────
        vis_group  = QGroupBox("Visualize Existing Results")
        vis_layout = QHBoxLayout()
        vis_label  = QLabel(
            "Already have a homographies .npy file from a previous run? "
            "Open the results viewer directly without running the full pipeline. NOTE: Currently only supports files for the full field of view (no ROI)."
        )
        vis_label.setWordWrap(True)
        vis_label.setStyleSheet("color: gray; font-size: 11px;")
        self._open_vis_btn = QPushButton("Open Results Viewer")
        self._open_vis_btn.setFixedHeight(34)
        self._open_vis_btn.setStyleSheet(
            f"font-size: 12px; font-weight: bold; "
            f"background-color: {THEME['accent']}; color: #000; border-radius: 5px;"
        )
        self._open_vis_btn.clicked.connect(self._launch_vis_dialog)
        vis_layout.addWidget(vis_label, stretch=1)
        vis_layout.addWidget(self._open_vis_btn)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        self.setLayout(layout)

        # No mandatory (*) so Next is never greyed out during development.
        # Default values let you click through all steps without real files.
        self.registerField("up2_path",  self.up2_edit)
        self.registerField("ang_path",  self.ang_edit)
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
            return
        ec = preset.get("elastic_constants_GPa", {})
        self._C11.setValue(ec.get("C11", self._C11.value()))
        self._C12.setValue(ec.get("C12", self._C12.value()))
        self._C44.setValue(ec.get("C44", self._C44.value()))
        self._struct_lbl.setText(preset.get("structure", self._struct_lbl.text()))

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

    def _launch_vis_dialog(self):
        dlg = VisualizationDialog({}, parent=self)
        dlg.show()


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

        tilt_layout.addRow("Sample tilt:", self.tilt)
        tilt_layout.addRow("Detector tilt:", self.det_tilt)
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
        
        self._strategy_standard  = QRadioButton("Standard (origin: upper-left, x →, y ↓)")
        self._strategy_lr = QRadioButton("Lower Right (origin: lower-right, x ←, y ↑)")
        self._strategy_de  = QRadioButton("Direct Electron  (origin: upper-right, x ←, y ↓)")

        self._strategy_standard.setChecked(True)

        strategy_layout.addWidget(self._strategy_standard)
        strategy_layout.addWidget(self._strategy_lr)
        strategy_layout.addWidget(self._strategy_de)


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
            "pc_edax":             (self.pc_x.value(), self.pc_y.value(), self.pc_z.value()),
            "pixel_size":          self.pixel_size.value(),
            "pat_h":               self.pat_h.value(),
            "pat_w":               self.pat_w.value(),
            "rows":                self.rows.value(),
            "cols":                self.cols.value(),
            "step_size":           self.step_size.value(),
            "scan_strategy":       ("direct_electron" if self._strategy_de.isChecked()
                                   else "lower_right" if self._strategy_lr.isChecked()
                                   else "standard"),
            "apply_pc_correction": self._apply_pc_correction.isChecked(),
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
        self.setTitle("Step 3 of 6 — Grain Segmentation & Region of Interest")
        self.setSubTitle(
            "Optionally segment grains, then define a region of interest. "
            "A yellow box marks the ROI on both maps."
        )
        self._ipf_worker  = None
        self._seg_worker  = None
        self._rgb_map     = None
        self._grain_ids             = None
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
        outer.addWidget(ipf_dir_group)
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
        self._seg_worker.start()

    def _on_seg_done(self, grain_ids, _kam, error: str):
        self._seg_btn.setEnabled(True)
        if grain_ids is None:
            self._seg_status.setText(f"Error: {error}")
            return

        self._grain_ids = grain_ids
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

        # Build a ListedColormap cycling tab20b for exactly n_surviving grains.
        # BoundaryNorm gives each integer compact index a unique discrete color
        # with no interpolation.
        _base_colors = list(_plt.cm.tab20b.colors)   # 20 RGBA tuples
        discrete_colors = [_base_colors[i % 20] for i in range(n_surviving)]
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
            (gid, int(sizes[gid]), discrete_colors[(ci - 1) % 20])
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
        ncols = max(1, min(4, (len(legend_patches) + 7) // 8))
        self._grain_ax.legend(
            handles=legend_patches,
            title=leg_title,
            loc="lower left",
            bbox_to_anchor=(0.0, -0.01),
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

class ReferencePatternPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle("Step 4 of 6 — Reference Pattern")
        self.setSubTitle(
            "Choose single-reference or per-grain mode, then click a point on "
            "the IPF map to set the reference pattern."
        )

        self._ref_marker    = None
        self._grain_markers = []   # per-grain mode: one marker per grain
        self._ipf_worker    = None
        self._ref_pattern_set: ReferencePatternSet = None

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_group  = QGroupBox("Reference Mode")
        mode_layout = QHBoxLayout()
        self._single_radio   = QRadioButton("Single reference")
        self._pergrain_radio = QRadioButton("Per-grain (auto)")
        self._single_radio.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._single_radio,   0)
        self._mode_grp.addButton(self._pergrain_radio, 1)
        mode_layout.addWidget(self._single_radio)
        mode_layout.addWidget(self._pergrain_radio)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)

        self._mode_grp.idToggled.connect(self._on_mode_changed)

        # ── Left panel (1/3): position controls + pattern preview ─────────────
        left        = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

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

        pos_layout.addRow("Row  (y):", self.ref_row)
        pos_layout.addRow("Col  (x):", self.ref_col)
        pos_layout.addRow(_note("(0, 0) is the top-left corner of the scan."))
        self._pos_group.setLayout(pos_layout)
        left_layout.addWidget(self._pos_group)

        # Per-grain info group (hidden in single mode)
        self._grain_info_group  = QGroupBox("Per-Grain References")
        grain_info_layout = QVBoxLayout()
        self._grain_count_lbl = QLabel("No segmentation found — run Step 3 first.")
        self._grain_count_lbl.setWordWrap(True)
        self._active_grain_lbl = QLabel("Active grain for override:")
        self._grain_combo = QComboBox()
        grain_info_layout.addWidget(self._grain_count_lbl)
        grain_info_layout.addWidget(self._active_grain_lbl)
        grain_info_layout.addWidget(self._grain_combo)
        grain_info_layout.addWidget(_note(
            "Click on the IPF map to move the selected grain's reference point."
        ))
        self._grain_info_group.setLayout(grain_info_layout)
        left_layout.addWidget(self._grain_info_group)

        # Pattern preview
        pat_group  = QGroupBox("Pattern Preview")
        pat_layout = QVBoxLayout(pat_group)

        self._pat_btn = QPushButton("Load Pattern Preview")
        self._pat_btn.clicked.connect(self._load_preview)
        pat_layout.addWidget(self._pat_btn)

        _bg = THEME["surface_bg"]
        self._pat_fig = Figure(tight_layout=True, facecolor=_bg)
        self._pat_ax  = self._pat_fig.add_subplot(111)
        self._pat_ax.set_facecolor(_bg)
        self._pat_ax.set_visible(False)
        self._pat_canvas = FigureCanvas(self._pat_fig)
        self._pat_canvas.setStyleSheet(f"background-color: {_bg};")
        self._pat_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._pat_status = QLabel("Click the IPF map or enter row/col, then load preview.")
        self._pat_status.setStyleSheet("color: gray;")
        self._pat_status.setWordWrap(True)

        pat_layout.addWidget(self._pat_canvas, stretch=1)
        pat_layout.addWidget(self._pat_status)
        left_layout.addWidget(pat_group, stretch=1)

        # ── Right panel (2/3): clickable IPF map ──────────────────────────────
        right        = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)

        ipf_group  = QGroupBox("IPF Map  (ND)  —  click to set reference position")
        ipf_layout = QVBoxLayout(ipf_group)

        self._ipf_fig = Figure(tight_layout=True)
        self._ipf_ax  = self._ipf_fig.add_subplot(111)
        self._ipf_ax.set_visible(False)
        self._ipf_canvas = FigureCanvas(self._ipf_fig)
        self._ipf_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ipf_canvas.mpl_connect("button_press_event", self._on_ipf_click)

        self._ipf_status = QLabel("Loading IPF map…")
        self._ipf_status.setStyleSheet("color: gray;")
        self._ipf_status.setWordWrap(True)

        ipf_layout.addWidget(self._ipf_canvas, stretch=1)
        ipf_layout.addWidget(self._ipf_status)
        right_layout.addWidget(ipf_group)

        # ── Outer layout ──────────────────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.addWidget(mode_group)

        outer = QVBoxLayout()
        outer.addLayout(top_row)

        panels = QHBoxLayout()
        panels.addWidget(left,  stretch=1)
        panels.addWidget(right, stretch=2)
        outer.addLayout(panels, stretch=1)
        self.setLayout(outer)

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

        if not ang_path or not os.path.exists(ang_path):
            self._ipf_status.setText("No ANG file — cannot draw IPF map.")
            return

        self._ipf_status.setText("Computing IPF map… (this may take a few seconds)")
        cached = wiz.ang_data if wiz.ang_data is not None and wiz.ang_loaded_path == ang_path else None
        self._ipf_worker = IPFWorker(ang_path, patshape, np.array([0, 0, 1]), ang_data=cached)
        self._ipf_worker.done_signal.connect(self._on_ipf_done)
        self._ipf_worker.start()

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_mode_changed(self, btn_id: int, checked: bool):
        if not checked:
            return
        is_single = (btn_id == 0)
        self._pos_group.setVisible(is_single)
        self._grain_info_group.setVisible(not is_single)
        if not is_single:
            self._auto_select_references()
        else:
            self._clear_grain_markers()
            self._update_ref_marker()

    def _auto_select_references(self):
        """Build a ReferencePatternSet from the current segmentation result."""
        wiz = self.wizard()
        grain_ids = getattr(wiz.roi_page, "_grain_ids", None)
        ang_data  = wiz.ang_data

        if grain_ids is None:
            self._grain_count_lbl.setText(
                "No segmentation found — go back to Step 3 and run segmentation first."
            )
            self._ref_pattern_set = None
            self._grain_combo.clear()
            self._clear_grain_markers()
            return

        if ang_data is None:
            self._grain_count_lbl.setText("ANG data not loaded yet — please wait.")
            return

        geom = wiz.geometry_page.get_params()
        self._ref_pattern_set = select_references(grain_ids, ang_data, geom["cols"])

        n = len(self._ref_pattern_set)
        self._grain_count_lbl.setText(f"{n} grain{'s' if n != 1 else ''} found — one reference auto-selected per grain.")

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

    def _on_ipf_done(self, rgb_map, error: str):
        if rgb_map is None:
            self._ipf_status.setText(f"Error computing IPF map: {error}")
            return
        self._ipf_ax.set_visible(True)
        self._ipf_ax.clear()
        self._ipf_ax.imshow(rgb_map, origin="upper", interpolation="nearest")
        self._ipf_ax.axis("off")
        self._ipf_fig.tight_layout(pad=0.5)
        self._ref_marker = None
        self._grain_markers.clear()
        if self._single_radio.isChecked():
            self._update_ref_marker()
        else:
            self._draw_grain_markers()
        self._ipf_canvas.draw()
        self._ipf_status.setText("Click any point to set the reference pattern.")

    def _on_ipf_click(self, event):
        if event.inaxes is not self._ipf_ax or not self._ipf_ax.get_visible():
            return
        wiz  = self.wizard()
        geom = wiz.geometry_page.get_params()
        col  = int(round(event.xdata))
        row  = int(round(event.ydata))
        col  = max(0, min(col, geom["cols"] - 1))
        row  = max(0, min(row, geom["rows"] - 1))

        if self._single_radio.isChecked():
            self.ref_row.blockSignals(True)
            self.ref_col.blockSignals(True)
            self.ref_row.setValue(row)
            self.ref_col.setValue(col)
            self.ref_row.blockSignals(False)
            self.ref_col.blockSignals(False)
            self._update_ref_marker()
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

    def _update_ref_marker(self):
        """Single-reference mode marker."""
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

    # ── Pattern preview ───────────────────────────────────────────────────────

    def _load_preview(self):
        wiz  = self.wizard()
        path = wiz.field("up2_path")
        geom = wiz.geometry_page.get_params()

        if self._single_radio.isChecked():
            row = self.ref_row.value()
            col = self.ref_col.value()
        else:
            gid = self._grain_combo.currentData()
            if gid is None or self._ref_pattern_set is None:
                self._pat_status.setText("No grain reference selected.")
                return
            entry = self._ref_pattern_set.by_grain(gid)
            row, col = entry.ref_row, entry.ref_col

        idx = row * geom["cols"] + col

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
            self._pat_ax.set_title(f"Row {row},  Col {col}", fontsize=9)
            self._pat_ax.axis("off")
            self._pat_fig.tight_layout(pad=0.3)
            self._pat_canvas.draw()
            self._pat_status.setText(
                f"{pat_obj.patshape[0]} × {pat_obj.patshape[1]} px  |  index {idx}"
            )
        except Exception as exc:
            self._pat_status.setText(f"Error: {exc}")

    def get_params(self) -> dict:
        if self._single_radio.isChecked():
            return {
                "ref_mode":     "single",
                "ref_position": (self.ref_row.value(), self.ref_col.value()),
            }
        return {
            "ref_mode":         "per_grain",
            "ref_pattern_set":  self._ref_pattern_set,
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
        self.setTitle("Step 5 of 6 — Pattern Processing")
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

        # Frequency filtering
        filt_group  = QGroupBox("Frequency Filtering")
        filt_layout = QFormLayout()

        self.high_pass = QDoubleSpinBox()
        self.high_pass.setRange(0, 200)
        self.high_pass.setValue(10.0)

        self.low_pass = QDoubleSpinBox()
        self.low_pass.setRange(0, 100)
        self.low_pass.setValue(0.0)

        filt_layout.addRow("High-pass sigma:", self.high_pass)
        filt_layout.addRow("Low-pass sigma:",  self.low_pass)
        filt_group.setLayout(filt_layout)
        ctrl.addWidget(filt_group)

        # Mask
        mask_group  = QGroupBox("Mask")
        mask_layout = QFormLayout()

        self.mask_type = QComboBox()
        self.mask_type.addItems(["None", "circular", "center_cross"])

        mask_layout.addRow("Mask type:", self.mask_type)
        mask_group.setLayout(mask_layout)
        ctrl.addWidget(mask_group)

        # CLAHE
        clahe_group  = QGroupBox("CLAHE")
        clahe_layout = QFormLayout()

        self.use_clahe = QCheckBox("Enable CLAHE")
        self.use_clahe.setChecked(False)

        self.clahe_kernel = QSpinBox()
        self.clahe_kernel.setRange(1, 32)
        self.clahe_kernel.setValue(6)

        self.clahe_clip = QDoubleSpinBox()
        self.clahe_clip.setRange(0.001, 1.0)
        self.clahe_clip.setValue(0.01)
        self.clahe_clip.setSingleStep(0.005)

        def _toggle_clahe(enabled):
            self.clahe_kernel.setEnabled(enabled)
            self.clahe_clip.setEnabled(enabled)
            self._schedule_preview()

        self.use_clahe.toggled.connect(_toggle_clahe)
        _toggle_clahe(self.use_clahe.isChecked())  # set initial enabled state

        clahe_layout.addRow("",             self.use_clahe)
        clahe_layout.addRow("Kernel size:", self.clahe_kernel)
        clahe_layout.addRow("Clip limit:",  self.clahe_clip)
        clahe_group.setLayout(clahe_layout)
        ctrl.addWidget(clahe_group)

        # Orientation
        orient_group  = QGroupBox("Orientation")
        orient_layout = QFormLayout()

        self.flip_x = QCheckBox("Flip patterns vertically")

        orient_layout.addRow("", self.flip_x)
        orient_group.setLayout(orient_layout)
        ctrl.addWidget(orient_group)

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

        # Advanced options group
        adv_group  = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()

        self._show_gradients = QCheckBox("Show gradients")
        self._show_gradients.setChecked(False)
        adv_layout.addWidget(self._show_gradients)

        self._sweep_btn = QPushButton("Parameter Sweep…")
        self._sweep_btn.setStyleSheet(
            f"background-color: {THEME['surface_bg']}; "
            f"color: {THEME['accent']}; "
            f"border: 1px solid {THEME['accent']}; border-radius: 4px; padding: 4px;"
        )
        self._sweep_btn.clicked.connect(self._open_sweep_dialog)
        adv_layout.addWidget(self._sweep_btn)

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
        for w in (self.high_pass, self.low_pass, self.clahe_clip):
            w.valueChanged.connect(self._schedule_preview)
        self.clahe_kernel.valueChanged.connect(self._schedule_preview)
        self.use_clahe.toggled.connect(self._schedule_preview)
        self.mask_type.currentIndexChanged.connect(self._schedule_preview)
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
        if not up2_path or not os.path.exists(up2_path):
            self._prev_status.setText("No UP2 file loaded — go back to Step 1.")
            return

        # Use the reference pattern index if available, otherwise 0
        try:
            ref_params = wiz.reference_page.get_params()
            geom       = wiz.geometry_page.get_params()
            if ref_params.get("ref_mode", "single") == "per_grain":
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

        self._prev_status.setText("Processing…")
        self._preview_worker = PatternPreviewWorker(up2_path, pat_idx, self.get_params())
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

    def _update_crop_rect(self):
        """Draw / redraw the green crop-fraction outline on the filtered pattern."""
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
        cf       = self.crop_fraction.value()
        margin_x = (1.0 - cf) / 2.0 * W
        margin_y = (1.0 - cf) / 2.0 * H
        self._crop_rect = _mp.Rectangle(
            (margin_x - 0.5, margin_y - 0.5), cf * W, cf * H,
            linewidth=1.5, edgecolor="#a6e3a1", facecolor="none", zorder=5,
        )
        self._ax_proc.add_patch(self._crop_rect)
        self._canvas_proc.draw_idle()

    # ── Parameter sweep ───────────────────────────────────────────────────────

    def _open_sweep_dialog(self):
        wiz      = self.wizard()
        up2_path = wiz.field("up2_path") if wiz else ""
        if not up2_path or not os.path.exists(up2_path):
            self._prev_status.setText("No UP2 file loaded — go back to Step 1.")
            return

        try:
            ref_params = wiz.reference_page.get_params()
            geom       = wiz.geometry_page.get_params()
            if ref_params.get("ref_mode", "single") == "per_grain":
                rps = ref_params.get("ref_pattern_set")
                pat_idx = rps[0].ref_pat_idx if rps and len(rps) > 0 else 0
            else:
                ref_pos = ref_params["ref_position"]
                pat_idx = int(np.ravel_multi_index(ref_pos, (geom["rows"], geom["cols"])))
        except Exception:
            pat_idx = 0

        dlg = ParameterSweepDialog(up2_path, pat_idx, self.get_params(), parent=self)
        dlg.params_selected.connect(self._apply_sweep_params)
        dlg.exec()

    def _apply_sweep_params(self, params: dict):
        """Apply the selected sweep result back to the page spinboxes."""
        self.high_pass.blockSignals(True)
        self.low_pass.blockSignals(True)
        self.clahe_kernel.blockSignals(True)
        self.clahe_clip.blockSignals(True)

        self.high_pass.setValue(params.get("high_pass_sigma", self.high_pass.value()))
        self.low_pass.setValue(params.get("low_pass_sigma",   self.low_pass.value()))
        self.clahe_kernel.setValue(params.get("clahe_kernel", self.clahe_kernel.value()))
        self.clahe_clip.setValue(params.get("clahe_clip",     self.clahe_clip.value()))

        self.high_pass.blockSignals(False)
        self.low_pass.blockSignals(False)
        self.clahe_kernel.blockSignals(False)
        self.clahe_clip.blockSignals(False)

        # Trigger a single live preview with the new values
        self._schedule_preview()

    # ── Params ────────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "high_pass_sigma": self.high_pass.value(),
            "low_pass_sigma":  self.low_pass.value(),
            "flip_x":          self.flip_x.isChecked(),
            "mask_type":       self.mask_type.currentText(),
            "use_clahe":       self.use_clahe.isChecked(),
            "clahe_kernel":    self.clahe_kernel.value(),
            "clahe_clip":      self.clahe_clip.value(),
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

        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 64)
        self.n_jobs.setValue(8)

        self.init_type = QComboBox()
        self.init_type.addItems(["none", "partial", "full"])

        opt_layout.addRow("Max iterations:", self.max_iter)
        opt_layout.addRow("n_jobs:", self.n_jobs)
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
                if ref.get("ref_mode", "single") == "single"
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
            f"CLAHE            : {'on (kernel=' + str(proc['clahe_kernel']) + ')' if proc['use_clahe'] else 'off'}",
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
        params.update({
            "max_iter":      self.max_iter.value(),
            "n_jobs":        self.n_jobs.value(),
            "init_type":     self.init_type.currentText(),
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
            "step_size":           p.get("step_size", 1.0),
            "pixel_size":          p.get("pixel_size", 1.0),
            "scan_strategy":       p.get("scan_strategy", "standard"),
            "apply_pc_correction": p.get("apply_pc_correction", False),
        }

        dlg = VisualizationDialog(vis_params, parent=self)
        dlg.show()

    def isComplete(self) -> bool:
        return self._done

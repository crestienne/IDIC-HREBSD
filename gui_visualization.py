"""
gui_visualization.py — stand-alone results viewer.

  VisualizationDialog  — QDialog that runs VisWorker and calls
                         Results_plotting.plot_all_results to render figures
"""

import os
import traceback

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QWidget,
)

from gui_theme import THEME, _make_browse_row, _make_browse_dir
from gui_workers import VisWorker
from gui_materials import _load_material_presets
from Results_plotting import plot_all_results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization dialog
# ─────────────────────────────────────────────────────────────────────────────

class VisualizationDialog(QDialog):
    """
    Stand-alone dialog that loads a homographies .npy file and produces
    the same strain/rotation figures as VisualizingResults_SiGe.py,
    but without PC drift correction.
    """

    def __init__(self, run_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualize Results")
        self.setMinimumSize(640, 820)
        self._worker   = None
        self._presets  = _load_material_presets()
        # ROI / scan-shape context — stored silently, not exposed in the form
        self._roi_slice  = run_params.get("roi_slice", None)
        self._full_rows  = run_params.get("full_rows", run_params.get("rows", 1))
        self._full_cols  = run_params.get("full_cols", run_params.get("cols", 1))

        layout = QVBoxLayout()
        layout.setSpacing(10)

        # ── Data paths ────────────────────────────────────────────────────────
        data_group  = QGroupBox("Data")
        data_layout = QFormLayout()

        self._npy_edit = QLineEdit(run_params.get("npy_path", ""))
        data_layout.addRow(
            "Homographies .npy:",
            _make_browse_row(self, self._npy_edit,
                             "NPY files (*.npy);;All files (*)", "Select homographies file"),
        )

        self._save_edit = QLineEdit(run_params.get("save_folder", ""))
        data_layout.addRow("Save figures to:", _make_browse_dir(self, self._save_edit))

        self._ang_edit = QLineEdit(run_params.get("ang", ""))
        data_layout.addRow(
            ".ang file (for TFBC):",
            _make_browse_row(self, self._ang_edit,
                             "ANG files (*.ang);;All files (*)", "Select .ang file"),
        )

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # ── Scan geometry ─────────────────────────────────────────────────────
        scan_group  = QGroupBox("Scan Geometry")
        scan_layout = QFormLayout()

        self._rows = QSpinBox()
        self._rows.setRange(1, 99999)
        self._rows.setValue(run_params.get("rows", 1))

        self._cols = QSpinBox()
        self._cols.setRange(1, 99999)
        self._cols.setValue(run_params.get("cols", 100))

        self._pat_h = QSpinBox()
        self._pat_h.setRange(1, 4096)
        self._pat_h.setValue(run_params.get("pat_h", 512))

        self._pat_w = QSpinBox()
        self._pat_w.setRange(1, 4096)
        self._pat_w.setValue(run_params.get("pat_w", 512))

        scan_layout.addRow("Rows:", self._rows)
        scan_layout.addRow("Columns:", self._cols)
        scan_layout.addRow("Pattern height (px):", self._pat_h)
        scan_layout.addRow("Pattern width (px):", self._pat_w)
        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)

        # ── Physical parameters ───────────────────────────────────────────────
        phys_group  = QGroupBox("Physical Parameters")
        phys_layout = QFormLayout()

        pc = run_params.get("pc_edax", (0.5, 0.5, 0.5))
        self._pc_x = QDoubleSpinBox(); self._pc_x.setRange(0, 2); self._pc_x.setDecimals(5); self._pc_x.setValue(pc[0])
        self._pc_y = QDoubleSpinBox(); self._pc_y.setRange(0, 2); self._pc_y.setDecimals(5); self._pc_y.setValue(pc[1])
        self._pc_z = QDoubleSpinBox(); self._pc_z.setRange(0, 2); self._pc_z.setDecimals(5); self._pc_z.setValue(pc[2])

        pc_row = QHBoxLayout()
        for lbl, w in [("x*", self._pc_x), ("y*", self._pc_y), ("z*", self._pc_z)]:
            pc_row.addWidget(QLabel(lbl))
            pc_row.addWidget(w)

        phys_layout.addRow("Pattern center (EDAX):", pc_row)

        self._tilt = QDoubleSpinBox()
        self._tilt.setRange(0, 90)
        self._tilt.setValue(run_params.get("tilt", 70.0))

        self._det_tilt = QDoubleSpinBox()
        self._det_tilt.setRange(0, 90)
        self._det_tilt.setValue(run_params.get("det_tilt", 10.0))

        self._samp_frame = QCheckBox("Rotate to sample frame")
        self._samp_frame.setChecked(True)

        phys_layout.addRow("Sample tilt (°):", self._tilt)
        phys_layout.addRow("Detector tilt (°):", self._det_tilt)
        phys_layout.addRow("", self._samp_frame)
        phys_group.setLayout(phys_layout)
        layout.addWidget(phys_group)

        # ── Plot options ──────────────────────────────────────────────────────
        plot_group  = QGroupBox("Plot Options")
        plot_layout = QFormLayout()

        self._strain_lim = QDoubleSpinBox()
        self._strain_lim.setRange(1e-6, 1.0)
        self._strain_lim.setDecimals(4)
        self._strain_lim.setSingleStep(1e-3)
        self._strain_lim.setValue(1e-2)

        self._rot_lim = QDoubleSpinBox()
        self._rot_lim.setRange(0.001, 10.0)
        self._rot_lim.setDecimals(3)
        self._rot_lim.setValue(0.25)

        self._linemap_row = QSpinBox()
        self._linemap_row.setRange(0, 99999)
        self._linemap_row.setValue(0)

        plot_layout.addRow("Strain colour limit (±):", self._strain_lim)
        plot_layout.addRow("Rotation colour limit (± °):", self._rot_lim)
        plot_layout.addRow("Line map row:", self._linemap_row)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # ── Traction-free BC ──────────────────────────────────────────────────
        tfbc_group  = QGroupBox("Traction-Free Boundary Condition")
        tfbc_layout = QFormLayout()

        self._tfbc_chk = QCheckBox("Apply traction-free BC (requires .ang file above)")
        self._tfbc_chk.setChecked(False)
        tfbc_layout.addRow("", self._tfbc_chk)

        # Crystal preset dropdown
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("— select preset —", userData=None)
        for preset in self._presets:
            self._preset_combo.addItem(preset["name"], userData=preset)
        self._preset_combo.currentIndexChanged.connect(self._apply_preset)
        tfbc_layout.addRow("Crystal preset:", self._preset_combo)

        self._struct_lbl = QLabel("cubic")
        tfbc_layout.addRow("Structure:", self._struct_lbl)

        self._C11 = QDoubleSpinBox(); self._C11.setRange(0, 9999); self._C11.setDecimals(1); self._C11.setSuffix(" GPa"); self._C11.setValue(165.7)
        self._C12 = QDoubleSpinBox(); self._C12.setRange(0, 9999); self._C12.setDecimals(1); self._C12.setSuffix(" GPa"); self._C12.setValue(63.9)
        self._C44 = QDoubleSpinBox(); self._C44.setRange(0, 9999); self._C44.setDecimals(1); self._C44.setSuffix(" GPa"); self._C44.setValue(79.6)

        tfbc_layout.addRow("C₁₁:", self._C11)
        tfbc_layout.addRow("C₁₂:", self._C12)
        tfbc_layout.addRow("C₄₄:", self._C44)

        tfbc_group.setLayout(tfbc_layout)
        layout.addWidget(tfbc_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._plot_btn = QPushButton("Compute & Plot All")
        self._plot_btn.setFixedHeight(40)
        self._plot_btn.setStyleSheet(
            f"font-size: 13px; font-weight: bold; "
            f"background-color: {THEME['accent']}; color: #000; border-radius: 5px;"
        )
        self._plot_btn.clicked.connect(self._run)
        btn_row.addWidget(self._plot_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("font-weight: bold;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self.setLayout(layout)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_preset(self, index: int):
        """Auto-fill elastic constant fields when a crystal preset is selected."""
        preset = self._preset_combo.itemData(index)
        if preset is None:
            return
        ec = preset.get("elastic_constants_GPa", {})
        self._C11.setValue(ec.get("C11", self._C11.value()))
        self._C12.setValue(ec.get("C12", self._C12.value()))
        self._C44.setValue(ec.get("C44", self._C44.value()))
        self._struct_lbl.setText(preset.get("structure", "cubic"))

    def _gather(self) -> dict:
        return {
            "npy_path":          self._npy_edit.text(),
            "ang_path":          self._ang_edit.text(),
            "save_folder":       self._save_edit.text(),
            "rows":              self._rows.value(),
            "cols":              self._cols.value(),
            "roi_slice":         self._roi_slice,   # passed silently to VisWorker
            "full_rows":         self._full_rows,
            "full_cols":         self._full_cols,
            "pat_h":             self._pat_h.value(),
            "pat_w":             self._pat_w.value(),
            "pc_edax":           (self._pc_x.value(), self._pc_y.value(), self._pc_z.value()),
            "tilt":              self._tilt.value(),
            "det_tilt":          self._det_tilt.value(),
            "samp_frame":        self._samp_frame.isChecked(),
            "strain_lim":        self._strain_lim.value(),
            "rot_lim":           self._rot_lim.value(),
            "linemap_row":       self._linemap_row.value(),
            "tfbc_enabled":      self._tfbc_chk.isChecked(),
            "crystal_C11":       self._C11.value(),
            "crystal_C12":       self._C12.value(),
            "crystal_C44":       self._C44.value(),
            "crystal_structure": self._struct_lbl.text(),
        }

    def _run(self):
        params = self._gather()
        if not params["npy_path"] or not os.path.exists(params["npy_path"]):
            self._status.setText("Homographies .npy file not found.")
            self._status.setStyleSheet(f"color: {THEME['error']}; font-weight: bold;")
            return
        if params["tfbc_enabled"] and not os.path.isfile(params["ang_path"]):
            self._status.setText("TFBC requires a valid .ang file — please set it above.")
            self._status.setStyleSheet(f"color: {THEME['error']}; font-weight: bold;")
            return
        if params["tfbc_enabled"] and not params["samp_frame"]:
            self._status.setText("TFBC requires 'Rotate to sample frame' to be checked.")
            self._status.setStyleSheet(f"color: {THEME['error']}; font-weight: bold;")
            return
        save = params["save_folder"]
        if save:
            os.makedirs(save, exist_ok=True)

        self._plot_btn.setEnabled(False)
        self._status.setText("Computing…")
        self._status.setStyleSheet(f"color: {THEME['warning']}; font-weight: bold;")
        self._pending_params = params

        self._worker = VisWorker(params)
        self._worker.results_signal.connect(self._on_results)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_results(self, results: dict):
        self._plot_btn.setEnabled(True)
        self._status.setText("Plotting…")
        plot_all_results(results, self._pending_params)
        tfbc_note = " + TFBC" if self._pending_params.get("tfbc_enabled") and "base_quats" in results else ""
        save_note = f" — figures saved to folder." if self._pending_params["save_folder"] else ""
        self._status.setText(f"Done{tfbc_note}. Figures shown in separate windows.{save_note}")
        self._status.setStyleSheet(f"color: {THEME['success']}; font-weight: bold;")

    def _on_error(self, msg: str):
        self._plot_btn.setEnabled(True)
        self._status.setText("Error — see details below.")
        self._status.setStyleSheet(f"color: {THEME['error']}; font-weight: bold;")
        print("\n--- Visualization error ---\n" + msg)

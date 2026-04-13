"""
gui_sweep.py — Parameter sweep dialog for pattern processing.

  ParameterSweepDialog — lets the user define a grid of high-pass sigma ×
                          CLAHE kernel values (all other params fixed at the
                          current page values), runs the sweep in a background
                          thread, and displays the results as a clickable grid.
                          Clicking any cell emits params_selected(dict) and
                          closes the dialog.
"""

import traceback

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QWidget, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal

from gui_theme import THEME
from gui_workers import SweepWorker


class ParameterSweepDialog(QDialog):
    """
    Sweep high-pass sigma (rows) × CLAHE kernel (columns) and show every
    processed pattern result in an interactive grid.  Click any cell to apply
    those parameters and close.
    """

    params_selected = pyqtSignal(dict)

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, up2_path: str, pat_idx: int, base_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter Sweep")
        self.setMinimumSize(960, 700)

        self._up2_path   = up2_path
        self._pat_idx    = pat_idx
        self._base       = base_params   # current page params — used for fixed values
        self._worker     = None
        self._param_grid = []            # flat list of param dicts, row-major
        self._n_rows     = 0
        self._n_cols     = 0
        self._fig        = None
        self._canvas     = None
        self._axes       = None          # ndarray shape (n_rows, n_cols)
        self._hp_vals    = []
        self._ck_vals    = []

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Sweep configuration ───────────────────────────────────────────────
        cfg      = QGroupBox("Sweep Configuration")
        cfg_form = QFormLayout(cfg)

        self._hp_edit = QLineEdit("5, 10, 15, 20")
        self._hp_edit.setToolTip(
            "Comma-separated high-pass sigma values.\n"
            "These become the rows of the result grid."
        )
        cfg_form.addRow("High-pass \u03c3 values:", self._hp_edit)

        self._ck_edit = QLineEdit("4, 6, 8")
        self._ck_edit.setToolTip(
            "Comma-separated CLAHE kernel values (integers).\n"
            "These become the columns of the result grid."
        )
        cfg_form.addRow("CLAHE kernel values:", self._ck_edit)

        self._fixed_lbl = QLabel(self._fixed_str())
        self._fixed_lbl.setStyleSheet("color: gray; font-size: 11px;")
        cfg_form.addRow("Fixed params:", self._fixed_lbl)

        layout.addWidget(cfg)

        # ── Run button ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Sweep")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setStyleSheet(
            f"font-weight: bold; background-color: {THEME['accent']}; "
            f"color: {THEME['accent_text']}; border-radius: 4px;"
        )
        self._run_btn.clicked.connect(self._run_sweep)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Scrollable results grid ───────────────────────────────────────────
        self._scroll       = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_inner = QWidget()
        self._scroll_vbox  = QVBoxLayout(self._scroll_inner)
        self._scroll_vbox.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._scroll_inner)
        layout.addWidget(self._scroll, stretch=1)

        # ── Status label ──────────────────────────────────────────────────────
        self._status = QLabel("Configure values above, then click Run Sweep.")
        self._status.setStyleSheet("color: gray;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fixed_str(self) -> str:
        b = self._base
        return (
            f"low_pass={b.get('low_pass_sigma', 0.0)},  "
            f"clahe_clip={b.get('clahe_clip', 0.01)},  "
            f"mask={b.get('mask_type', 'None')},  "
            f"flip_x={b.get('flip_x', False)}"
        )

    @staticmethod
    def _parse_floats(text: str) -> list:
        """Parse comma-separated text into a deduplicated list of floats."""
        vals, seen = [], set()
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = float(tok)
            except ValueError:
                continue
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    # ── Sweep logic ───────────────────────────────────────────────────────────

    def _run_sweep(self):
        hp_vals = self._parse_floats(self._hp_edit.text())
        ck_vals = self._parse_floats(self._ck_edit.text())
        if not hp_vals or not ck_vals:
            self._status.setText("Enter at least one value for each parameter.")
            self._status.setStyleSheet(f"color: {THEME['error']};")
            return

        self._hp_vals = hp_vals
        self._ck_vals = ck_vals
        self._n_rows  = len(hp_vals)
        self._n_cols  = len(ck_vals)
        n_total       = self._n_rows * self._n_cols

        # Build param list: outer loop = hp (rows), inner = ck (cols)
        self._param_grid = []
        for hp in hp_vals:
            for ck in ck_vals:
                p = dict(self._base)
                p["high_pass_sigma"] = hp
                p["clahe_kernel"]    = int(round(ck))
                self._param_grid.append(p)

        # Create fresh matplotlib figure
        _bg      = THEME["surface_bg"]
        cell_in  = 1.8          # inches per cell
        label_in = 0.4          # extra space for row ylabel
        fig_w = cell_in * self._n_cols + label_in
        fig_h = cell_in * self._n_rows + 0.3
        self._fig = Figure(figsize=(fig_w, fig_h), facecolor=_bg)

        axes_flat = []
        for ri in range(self._n_rows):
            row = []
            for ci in range(self._n_cols):
                ax = self._fig.add_subplot(self._n_rows, self._n_cols,
                                           ri * self._n_cols + ci + 1)
                ax.set_facecolor(_bg)
                ax.set_visible(False)
                row.append(ax)
            axes_flat.append(row)
        self._axes = np.array(axes_flat)

        # Column labels (top row)
        for ci, ck in enumerate(ck_vals):
            self._axes[0, ci].set_title(
                f"kernel={int(ck)}", fontsize=7, pad=2, color="white"
            )
        # Row labels (left column)
        for ri, hp in enumerate(hp_vals):
            self._axes[ri, 0].set_ylabel(
                f"hp\u03c3={hp}", fontsize=7, labelpad=2, color="white"
            )

        self._fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)

        # Replace old canvas in the scroll widget
        while self._scroll_vbox.count():
            item = self._scroll_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_bg};")
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        self._canvas.setMinimumSize(int(fig_w * 100), int(fig_h * 100))
        self._canvas.mpl_connect("button_press_event", self._on_grid_click)
        self._scroll_vbox.addWidget(self._canvas)
        self._scroll_vbox.addStretch()

        # Abort any running worker and start fresh
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()

        self._run_btn.setEnabled(False)
        self._status.setText(f"Running 0 / {n_total}…")
        self._status.setStyleSheet(f"color: {THEME['warning']};")

        self._worker = SweepWorker(self._up2_path, self._pat_idx, self._param_grid)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, idx: int, total: int, img, params: dict):
        ri = idx // self._n_cols
        ci = idx %  self._n_cols
        ax = self._axes[ri, ci]
        ax.set_visible(True)
        ax.clear()
        ax.imshow(img, cmap="gray", origin="upper", aspect="auto")
        ax.axis("off")
        # Re-apply labels lost by clear()
        ax.set_title(
            f"kernel={params['clahe_kernel']}", fontsize=7, pad=2, color="white"
        )
        if ci == 0:
            ax.set_ylabel(
                f"hp\u03c3={params['high_pass_sigma']}", fontsize=7,
                labelpad=2, color="white"
            )
        self._canvas.draw_idle()
        self._status.setText(f"Running {idx + 1} / {total}…")

    def _on_done(self):
        self._run_btn.setEnabled(True)
        self._status.setText(
            "Done — click any pattern to apply those parameters and close."
        )
        self._status.setStyleSheet(f"color: {THEME['success']};")

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._status.setText("Error — see console for details.")
        self._status.setStyleSheet(f"color: {THEME['error']};")
        print("\n--- Sweep error ---\n" + msg)

    # ── Grid click ────────────────────────────────────────────────────────────

    def _on_grid_click(self, event):
        """Detect which subplot was clicked, emit params, and close."""
        if event.inaxes is None or self._axes is None:
            return
        for ri in range(self._n_rows):
            for ci in range(self._n_cols):
                if event.inaxes is self._axes[ri, ci]:
                    idx = ri * self._n_cols + ci
                    if idx < len(self._param_grid):
                        self._highlight(ri, ci)
                        self.params_selected.emit(self._param_grid[idx])
                        self.accept()
                    return

    def _highlight(self, ri: int, ci: int):
        """Draw a yellow border around the selected axes before closing."""
        ax = self._axes[ri, ci]
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["accent"])
            spine.set_linewidth(3)
            spine.set_visible(True)
        self._canvas.draw()

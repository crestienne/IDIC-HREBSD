"""
gui_materials.py — shared material preset loader + new-material dialog.

Kept in its own module so that both gui_pages.py and gui_visualization.py
can import from it without creating a circular dependency.
"""

import os
import json
import re

from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
    QLabel, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal

_MATERIALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Materials")


def _load_material_presets() -> list:
    """Return a list of material dicts from Materials/*.json, sorted by name."""
    presets = []
    if not os.path.isdir(_MATERIALS_DIR):
        return presets
    for fname in sorted(os.listdir(_MATERIALS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_MATERIALS_DIR, fname)) as f:
                presets.append(json.load(f))
        except Exception:
            pass
    presets.sort(key=lambda d: d.get("name", ""))
    return presets


# ─────────────────────────────────────────────────────────────────────────────
# New-material dialog
# ─────────────────────────────────────────────────────────────────────────────

class NewMaterialDialog(QDialog):
    """
    Dialog for creating a new material JSON file in the Materials/ directory.
    Emits `saved(dict)` with the new preset when the file is written.
    """
    saved = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Material Preset")
        self.setMinimumWidth(360)

        form = QFormLayout()

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Germanium (Ge)")
        form.addRow("Material name:", self._name)

        self._structure = QComboBox()
        self._structure.addItems(["cubic", "hexagonal", "tetragonal", "orthorhombic"])
        form.addRow("Crystal structure:", self._structure)

        def _spinbox(lo=0.0, hi=9999.0, val=0.0):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(1)
            sb.setSuffix(" GPa")
            sb.setValue(val)
            return sb

        self._C11 = _spinbox()
        self._C12 = _spinbox()
        self._C44 = _spinbox()
        form.addRow("C₁₁:", self._C11)
        form.addRow("C₁₂:", self._C12)
        form.addRow("C₄₄:", self._C44)

        self._reference = QLineEdit()
        self._reference.setPlaceholderText("e.g. Madelung 1982  (optional)")
        form.addRow("Reference:", self._reference)

        # Save location row
        self._save_dir = QLineEdit(_MATERIALS_DIR)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._pick_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._save_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("Save to folder:", dir_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color: gray; font-style: italic;")
        self._status.setWordWrap(True)

        # Buttons
        btn_row = QHBoxLayout()
        save_btn   = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(self._status)
        outer.addLayout(btn_row)

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder", self._save_dir.text())
        if path:
            self._save_dir.setText(path)

    def _save(self):
        name = self._name.text().strip()
        if not name:
            self._status.setText("Material name is required.")
            self._status.setStyleSheet("color: #cc4444;")
            return

        preset = {
            "name":      name,
            "structure": self._structure.currentText(),
            "elastic_constants_GPa": {
                "C11": self._C11.value(),
                "C12": self._C12.value(),
                "C44": self._C44.value(),
            },
        }
        ref = self._reference.text().strip()
        if ref:
            preset["reference"] = ref

        # Derive filename from the material name
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[\s]+", "_", slug).strip("_") or "material"
        save_dir = self._save_dir.text().strip()

        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{slug}.json")

        # Warn if file already exists
        if os.path.isfile(filepath):
            ans = QMessageBox.question(
                self, "Overwrite?",
                f"{os.path.basename(filepath)} already exists.\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        try:
            with open(filepath, "w") as f:
                json.dump(preset, f, indent=2)
        except Exception as exc:
            self._status.setText(f"Save failed: {exc}")
            self._status.setStyleSheet("color: #cc4444;")
            return

        self._status.setText(f"Saved → {filepath}")
        self._status.setStyleSheet("color: #88cc88;")
        self.saved.emit(preset)
        self.accept()

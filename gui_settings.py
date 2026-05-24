"""
gui_settings.py — user-facing settings dialog + QSettings persistence.

Persists two preferences across GUI launches:
    - "font_point_size" : int    (8..24, default 13)
    - "theme_mode"      : str    ("dark" or "light", default "dark")

Stored via QSettings (platform-native location — `~/Library/Preferences/...`
on macOS, registry on Windows, `~/.config/...` on Linux).
"""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QComboBox, QPushButton, QMessageBox, QWidget,
)


ORG_NAME       = "DIC-HREBSD"
APP_NAME       = "DIC-HREBSD GUI"

DEFAULT_FONT_PT = 13
DEFAULT_MODE    = "dark"
MIN_FONT_PT     = 9
MAX_FONT_PT     = 22


# ─────────────────────────────────────────────────────────────────────────────
# QSettings helpers
# ─────────────────────────────────────────────────────────────────────────────

def _settings() -> QSettings:
    return QSettings(ORG_NAME, APP_NAME)


def saved_font_pt() -> int:
    """Return the persisted font point size, falling back to the default."""
    try:
        v = int(_settings().value("font_point_size", DEFAULT_FONT_PT))
    except (TypeError, ValueError):
        v = DEFAULT_FONT_PT
    return max(MIN_FONT_PT, min(MAX_FONT_PT, v))


def saved_theme_mode() -> str:
    """Return the persisted theme mode (\"dark\" or \"light\")."""
    v = str(_settings().value("theme_mode", DEFAULT_MODE)).lower()
    return v if v in ("dark", "light") else DEFAULT_MODE


def save_settings(font_pt: int, theme_mode: str) -> None:
    s = _settings()
    s.setValue("font_point_size", int(font_pt))
    s.setValue("theme_mode", theme_mode if theme_mode in ("dark", "light") else DEFAULT_MODE)
    s.sync()


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Lightweight settings dialog: font size + theme mode.

    Changes are persisted via QSettings on Apply / OK.  Both options take
    effect after the GUI restarts (Qt's live-restyle on QSS + QPalette is
    flaky for nested widgets — restart gives a clean result).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DIC-HREBSD — Settings")
        self.resize(420, 220)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # ── Permanent "under development" banner ─────────────────────────────
        # Construction emoji + short notice, sits at the top of the dialog
        # regardless of which setting is currently being shown.
        wip_row = QHBoxLayout()
        wip_row.setSpacing(8)
        wip_icon = QLabel("🚧")
        wip_icon.setStyleSheet("font-size: 32px;")
        wip_icon.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        wip_text = QLabel(
            "<b>Settings menu is still under development</b><br>"
            "Layout and options are likely to change in future releases."
        )
        wip_text.setWordWrap(True)
        wip_row.addWidget(wip_icon)
        wip_row.addWidget(wip_text, stretch=1)
        outer.addLayout(wip_row)

        form = QFormLayout()
        form.setSpacing(8)

        self._font_spin = QSpinBox()
        self._font_spin.setRange(MIN_FONT_PT, MAX_FONT_PT)
        self._font_spin.setSuffix(" pt")
        self._font_spin.setValue(saved_font_pt())
        form.addRow("Application font size:", self._font_spin)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Dark (default)", userData="dark")
        self._theme_combo.addItem("Light",          userData="light")
        cur = saved_theme_mode()
        idx = 1 if cur == "light" else 0
        self._theme_combo.setCurrentIndex(idx)
        form.addRow("Colour theme:", self._theme_combo)

        outer.addLayout(form)

        note = QLabel(
            "<i>Changes take effect after the GUI is restarted.</i>"
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        outer.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        outer.addLayout(btn_row)

    def _on_save(self) -> None:
        font_pt = int(self._font_spin.value())
        mode    = self._theme_combo.currentData() or DEFAULT_MODE
        save_settings(font_pt, mode)
        QMessageBox.information(
            self, "Settings saved",
            "Settings have been saved.\n\nRestart the DIC-HREBSD GUI for the "
            "changes to take effect.",
        )
        self.accept()

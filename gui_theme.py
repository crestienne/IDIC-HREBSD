"""
gui_theme.py — colour theme, QPalette setup, and shared widget helpers.

Edit the THEME dict to restyle the whole GUI without touching any other file.
"""

import os

from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QWidget,
)
from PyQt6.QtGui import QColor, QPalette


# ─────────────────────────────────────────────────────────────────────────────
# Colour theme  ← edit this dict to restyle the whole GUI
# ─────────────────────────────────────────────────────────────────────────────

THEME = {
    # ── Backgrounds ──────────────────────────────────────────────────────────
    #colors generated via coolers website

    "window_bg":    "#102B3F",   # outer window
    "surface_bg":   "#163146",   # group-box interiors, text areas
    "input_bg":     "#829cbc",   # spin boxes, line edits, combo boxes

    # ── Text ──────────────────────────────────────────────────────────────────
    "text":         "#eef0f2",   # primary text
    "text_disabled":"#585b70",   # greyed-out labels / inactive widgets
    "text_hint":    "#6c7086",   # small helper labels (_note())

    # ── Accent / interactive ──────────────────────────────────────────────────
    "accent":       "#fdca40",   # button faces, selected items, focus ring
    "accent_hover": "#efc143",   # button hover
    "accent_text":  "#050505",   # text ON accent-coloured buttons

    # ── Borders ───────────────────────────────────────────────────────────────
    "border":       "#45475a",   # group-box frames, separators

    # ── Status colours ────────────────────────────────────────────────────────
    "success":      "#a6e3a1",   # "Done" messages
    "error":        "#f38ba8",   # error messages
    "warning":      "#fab387",   # warnings / in-progress

    # ── Wizard header bar (title + subtitle strip at the top) ────────────────
    "header_bg":    "#21295c",   # background of the step-title banner
    "header_text":  "#eef0f2",   # title and subtitle text colour

    # ── Special widgets ───────────────────────────────────────────────────────
    "run_btn_bg":   "#2e7d32",   # Run Pipeline button background
    "run_btn_text": "#ffffff",
}


def apply_theme(app) -> None:
    """
    Apply THEME to *app*.  Call once after QApplication is created.

    Two mechanisms are used:
      1. QPalette — controls the standard Qt colour roles (background,
         text, highlight, etc.) understood by all widgets.
      2. QSS stylesheet — fine-tunes specific widgets that QPalette
         can't reach (borders, radii, padding, …).

    To change the look, edit the THEME dict above — you don't need to
    touch this function.
    """
    t = THEME  # shorthand

    # ── 1. QPalette ───────────────────────────────────────────────────────────
    pal = QPalette()

    def c(key):
        return QColor(t[key])

    pal.setColor(QPalette.ColorRole.Window,          c("window_bg"))
    pal.setColor(QPalette.ColorRole.WindowText,      c("text"))
    pal.setColor(QPalette.ColorRole.Base,            c("input_bg"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   c("surface_bg"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     c("surface_bg"))
    pal.setColor(QPalette.ColorRole.ToolTipText,     c("text"))
    pal.setColor(QPalette.ColorRole.Text,            c("text"))
    pal.setColor(QPalette.ColorRole.Button,          c("accent"))
    pal.setColor(QPalette.ColorRole.ButtonText,      c("accent_text"))
    pal.setColor(QPalette.ColorRole.BrightText,      c("accent_hover"))
    pal.setColor(QPalette.ColorRole.Highlight,       c("accent"))
    pal.setColor(QPalette.ColorRole.HighlightedText, c("accent_text"))
    pal.setColor(QPalette.ColorRole.Link,            c("accent_hover"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, c("text_disabled"))

    # Header background — QWizard ModernStyle paints the header strip using a
    # gradient between Light (top) and Midlight (bottom).  Setting both to
    # header_bg gives a flat, solid-colour header.
    pal.setColor(QPalette.ColorRole.Light,    c("header_bg"))
    pal.setColor(QPalette.ColorRole.Midlight, c("header_bg"))

    # Disabled state
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, c("text_disabled"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       c("text_disabled"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c("text_disabled"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button,     c("surface_bg"))

    app.setPalette(pal)

    # ── 2. QSS stylesheet ─────────────────────────────────────────────────────
    app.setStyleSheet(f"""
        QWizard, QDialog {{
            background-color: {t["window_bg"]};
        }}
        QWizardPage {{
            background-color: {t["window_bg"]};
        }}
        QGroupBox {{
            background-color: {t["surface_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            margin-top: 10px;
            padding: 8px 6px 6px 6px;
            font-weight: bold;
            color: {t["text"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 4px;
            color: {t["text"]};
        }}
        QLabel {{
            color: {t["text"]};
            background: transparent;
        }}
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {t["input_bg"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 4px;
            padding: 3px 5px;
            selection-background-color: {t["accent"]};
            font-family: "Arial";
        }}
        QSpinBox, QDoubleSpinBox {{
            background-color: {t["input_bg"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 4px;
            padding: 2px 4px;
        }}
        QComboBox {{
            background-color: {t["input_bg"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 4px;
            padding: 2px 6px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {t["surface_bg"]};
            color: {t["text"]};
            selection-background-color: {t["accent"]};
        }}
        QPushButton {{
            background-color: {t["accent"]};
            color: {t["accent_text"]};
            border: none;
            border-radius: 4px;
            padding: 5px 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {t["accent_hover"]};
        }}
        QPushButton:disabled {{
            background-color: {t["surface_bg"]};
            color: {t["text_disabled"]};
        }}
        QCheckBox {{
            color: {t["text"]};
            spacing: 6px;
        }}
        QCheckBox::indicator {{
            width: 14px;
            height: 14px;
            border: 1px solid {t["border"]};
            border-radius: 3px;
            background-color: {t["input_bg"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {t["accent"]};
            border-color: {t["accent"]};
        }}
        QScrollArea, QScrollBar {{
            background-color: {t["window_bg"]};
        }}
        QScrollBar:vertical {{
            width: 10px;
            background: {t["surface_bg"]};
        }}
        QScrollBar::handle:vertical {{
            background: {t["border"]};
            border-radius: 5px;
            min-height: 20px;
        }}
        QProgressBar {{
            border: 1px solid {t["border"]};
            border-radius: 4px;
            background-color: {t["surface_bg"]};
            text-align: center;
            color: {t["text"]};
        }}
        QProgressBar::chunk {{
            background-color: {t["accent"]};
            border-radius: 3px;
        }}
        QSplitter::handle {{
            background-color: {t["border"]};
        }}
        QTabWidget::pane {{
            border: 1px solid {t["border"]};
            background-color: {t["window_bg"]};
        }}
        QTabBar::tab {{
            background-color: {t["surface_bg"]};
            color: {t["text_hint"]};
            padding: 8px 16px;
            min-width: 160px;
            border: 1px solid {t["border"]};
            border-right: none;
            border-radius: 4px 0 0 4px;
            text-align: left;
        }}
        QTabBar::tab:selected {{
            background-color: {t["window_bg"]};
            color: {t["accent"]};
            border-left: 3px solid {t["accent"]};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {t["border"]};
            color: {t["text"]};
        }}
        /* ── Wizard header strip ── */
        QWizard QLabel#qt_wizard_titleLabel {{
            color: {t["header_text"]};
            font-size: 14px;
            font-weight: bold;
        }}
        QWizard QLabel#qt_wizard_subTitleLabel {{
            color: {t["header_text"]};
            font-size: 11px;
        }}
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Shared widget helpers
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

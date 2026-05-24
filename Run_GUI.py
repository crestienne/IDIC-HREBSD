"""
gui_qt.py — entry point for the DIC-HREBSD PyQt6 wizard GUI.

Run:
    python Run_GUI.py

Module layout
─────────────
  gui_theme.py        THEME dict, apply_theme(), shared widget helpers
  gui_workers.py      QThread subclasses (pipeline, IPF, segmentation, vis)
  gui_visualization.py  _vis_plot_all(), VisualizationDialog
  gui_pages.py        All six QWizardPage subclasses
  Run_GUI.py           HREBSDWizard + __main__ entry point  ← you are here
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the matplotlib backend before any other matplotlib import.
import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import QApplication, QWizard
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QFont, QPainter, QColor

from gui_theme import apply_theme, set_theme_mode, THEME
from gui_pages import (
    LoadFilesPage,
    ScanGeometryPage,
    ROISelectionPage,
    ReferencePatternPage,
    PatternProcessingPage,
    OptimizationRunPage,
)
from gui_help import HelpDialog
from gui_settings import (
    SettingsDialog, saved_font_pt, saved_theme_mode,
    ORG_NAME, APP_NAME,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main wizard
# ─────────────────────────────────────────────────────────────────────────────

class HREBSDWizard(QWizard):

    def __init__(self):
        super().__init__()
        self.ang_data            = None   # cached result of utilities.read_ang
        self.ang_loaded_path     = ""     # ang path used for the current cache
        self.ang_loaded_patshape = None   # patshape used when loading the cache
        self.setWindowTitle("DIC-HREBSD Pipeline")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(760, 720)
        self.setOption(QWizard.WizardOption.HaveHelpButton, True)
        self.helpRequested.connect(self._show_help)

        # ── Always-visible custom buttons in the wizard's bottom row ──────
        # CustomButton1 → Open Results Viewer
        # CustomButton2 → Settings (lives right next to Help)
        self.setOption(QWizard.WizardOption.HaveCustomButton1, True)
        self.setButtonText(QWizard.WizardButton.CustomButton1, "Open Results Viewer")
        self.setOption(QWizard.WizardOption.HaveCustomButton2, True)
        self.setButtonText(QWizard.WizardButton.CustomButton2, "Settings…")
        self.customButtonClicked.connect(self._on_custom_button_clicked)
        # Bottom-row order: [Help] [Settings] [Open Results Viewer] [Stretch]
        # [Back] [Next] [Finish] [Cancel].
        self.setButtonLayout([
            QWizard.WizardButton.HelpButton,
            QWizard.WizardButton.CustomButton2,
            QWizard.WizardButton.CustomButton1,
            QWizard.WizardButton.Stretch,
            QWizard.WizardButton.BackButton,
            QWizard.WizardButton.NextButton,
            QWizard.WizardButton.FinishButton,
            QWizard.WizardButton.CancelButton,
        ])

        self.files_page      = LoadFilesPage()
        self.geometry_page   = ScanGeometryPage()
        self.roi_page        = ROISelectionPage()
        self.reference_page  = ReferencePatternPage()
        self.processing_page = PatternProcessingPage()
        self.run_page        = OptimizationRunPage()

        self.addPage(self.files_page)
        self.addPage(self.geometry_page)
        self.addPage(self.processing_page)
        self.addPage(self.roi_page)
        self.addPage(self.reference_page)
        self.addPage(self.run_page)

        # The yellow progress strip (paintEvent) depends on the current
        # page — force a repaint whenever the user navigates.
        self.currentIdChanged.connect(lambda _id: self.update())

    def paintEvent(self, event):
        """Paint a thin yellow progress strip across the bottom of the
        wizard, just above the button row.  The strip grows from 1/6 of
        the wizard width on Step 1 to the full width on Step 6."""
        super().paintEvent(event)
        btn = self.button(QWizard.WizardButton.HelpButton)
        if btn is None or not btn.isVisible():
            return
        ids = self.pageIds()
        try:
            step_index = ids.index(self.currentId())   # 0..5
        except ValueError:
            return
        n_steps = max(len(ids), 1)
        fraction = (step_index + 1) / n_steps
        top = btn.mapTo(self, QPoint(0, 0)).y()
        painter = QPainter(self)
        painter.fillRect(
            0, top - 18,
            int(self.width() * fraction), 5,
            QColor(THEME["accent"]),
        )

    def _show_help(self):
        dlg = HelpDialog(current_page_index=self.currentId(), parent=self)
        dlg.exec()

    def _on_custom_button_clicked(self, which):
        """Dispatch for the custom buttons in the wizard's bottom row.

        PyQt6 delivers ``which`` as a ``QWizard.WizardButton`` enum member
        (calling ``int()`` on it raises TypeError).  Normalise both sides
        through ``.value`` so the comparison works regardless of whether a
        future build switches to plain ints.
        """
        def _wb_int(x):
            return x.value if hasattr(x, "value") else int(x)
        if _wb_int(which) == _wb_int(QWizard.WizardButton.CustomButton2):
            self._launch_settings_dialog()
        else:   # CustomButton1 (or anything else registered later)
            self._launch_vis_dialog()

    def _launch_vis_dialog(self):
        """Open the results viewer with an empty run-params dict — the
        VisualizationDialog will prompt the user for a homographies .npy
        from a previous pipeline run."""
        # Lazy import so we don't pay the matplotlib-dialog cost just
        # because the wizard opened.
        from gui_visualization import VisualizationDialog
        dlg = VisualizationDialog({}, parent=self)
        dlg.show()

    def _launch_settings_dialog(self):
        """Open the persistent settings dialog (font size + dark/light theme).
        Changes are saved via QSettings; the user is told to restart for the
        new look to apply cleanly."""
        dlg = SettingsDialog(parent=self)
        dlg.exec()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # Fusion gives QPalette full control on all platforms

    # Tell Qt where QSettings should persist (needs to match the strings
    # used in gui_settings.py).  Setting these here ensures QSettings()
    # with no args also works for any downstream code.
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)

    # Apply persisted user preferences before any widgets are constructed.
    set_theme_mode(saved_theme_mode())
    app.setFont(QFont("Arial", saved_font_pt()))
    apply_theme(app)

    wiz = HREBSDWizard()
    wiz.show()
    sys.exit(app.exec())

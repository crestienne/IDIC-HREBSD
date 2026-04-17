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
from PyQt6.QtGui import QFont

from gui_theme import apply_theme
from gui_pages import (
    LoadFilesPage,
    ScanGeometryPage,
    ROISelectionPage,
    ReferencePatternPage,
    PatternProcessingPage,
    OptimizationRunPage,
)
from gui_help import HelpDialog


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

        self.files_page      = LoadFilesPage()
        self.geometry_page   = ScanGeometryPage()
        self.roi_page        = ROISelectionPage()
        self.reference_page  = ReferencePatternPage()
        self.processing_page = PatternProcessingPage()
        self.run_page        = OptimizationRunPage()

        self.addPage(self.files_page)
        self.addPage(self.geometry_page)
        self.addPage(self.roi_page)
        self.addPage(self.reference_page)
        self.addPage(self.processing_page)
        self.addPage(self.run_page)

    def _show_help(self):
        dlg = HelpDialog(current_page_index=self.currentId(), parent=self)
        dlg.exec()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # Fusion gives QPalette full control on all platforms
    app.setFont(QFont("Arial"))
    apply_theme(app)
    wiz = HREBSDWizard()
    wiz.show()
    sys.exit(app.exec())

"""
gui_help.py — Help dialog for the DIC-HREBSD wizard.

EDITING GUIDE
─────────────
All user-facing help text lives in HELP_CONTENT below — one entry per wizard
step.  Each entry is a dict with:

  "title"  : tab label shown in the dialog
  "body"   : HTML string rendered in the text area

You can use basic HTML tags: <b>, <i>, <h3>, <p>, <ul>/<li>, <br>, <hr>.
You do NOT need to touch the HelpDialog class itself to update content.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTabBar, QWidget, QTextBrowser, QPushButton,
    QStylePainter, QStyleOptionTab, QStyle,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPen

from gui_theme import THEME


# ─────────────────────────────────────────────────────────────────────────────
# Help content — edit this section freely
# ─────────────────────────────────────────────────────────────────────────────

HELP_CONTENT = [

    {"title": "Step 1 — Load Files",              "body": """
<h3>Input Files</h3>
<p>Two files are required before the pipeline can run:</p>
<ul>
  <li><b>UP2 pattern file (.up2)</b> — the raw EBSD pattern, currently only EDAX files are supported. 
     </li>
  <li><b>ANG scan file (.ang)</b> — the OIM / TSL scan file that stores the Euler angles, scan-step size, and grid layout for every scan point. Currently EDAX and EMsoft ang files are supported.</li>
</ul>

<hr>

<h3>Output</h3>
<ul>
  <li><b>Run name</b> — a short label for this run. A subfolder with this name
      will be created inside the parent directory you choose below.</li>
  <li><b>Save results in</b> — the parent directory where the run subfolder will
      be created. Use the Browse button to pick a folder.</li>
  <li><b>Will create</b> — a live preview of the full output path that will be
      created when the pipeline runs. No files are written until Step 6.</li>
</ul>

<hr>

<h3>Pattern Preview</h3>
<p>Once a valid UP2 file is loaded, the first pattern in the stack is displayed
here in greyscale. This is a quick sanity-check only — it confirms the file
can be read and shows the raw (unprocessed) pattern. Actual pattern
pre-processing is configured in Step 5.</p>

<hr>

<h3>Material Properties</h3>
<p>The elastic constants are used to convert the measured homographies into
deviatoric strain tensors via Hooke's law. Only cubic materials are currently supported.</p>

<p>Use the <b>Crystal preset</b> drop-down to load values for a known material
automatically. You can then edit any field to fine-tune the constants. The
<b>Structure</b> label (e.g. <code>cubic</code>) is set by the preset and
determines which symmetry relations are applied during the strain calculation.</p>
"""},



{"title": "Step 2 — Scan Geometry",            "body": """
<h3>This Section will be Completed</h3>
<p> Insert content here </p>
 
 """},

{"title": "Step 3 — Segmentation & ROI",   "body": """
 
 <p> This sections enables regions of the scan to be selected for analysis.</p>
<h3> Segmentation</h3>
<p> There are two presets available for segmentation:  </p>
 
<li><b>Misorientation</b> — the minium misorientation angle (in degrees) that define the grain boundaries </li>
<li><b>Minimum Grain Size </b> — the minimum number of pixels a grain must have to be considered </li>
 
 <h3> ROI Selection</h3>
 <p> Enables sections of the scan to be selected for analysis. Eventually there will be more options than the rectangular selection tool, but for now this is the only option. </p>
 
 """},




    {"title": "Step 4 — Reference Pattern",        "body": "<p>TO DO</p>"},
    {"title": "Step 5 — Pattern Processing",       "body": "<p>TO DO</p>"},
    {"title": "Step 6 — Run",                      "body": "<p>TO DO</p>"},
    {"title": "Reference Frames",                  "body": "<p>TO DO</p>"},
    {"title": "Acknowledgements",                  "body": "<p>TO DO</p>"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Dialog — no need to edit below this line
# ─────────────────────────────────────────────────────────────────────────────

class _HorizontalTabBar(QTabBar):
    """
    Keeps tab labels horizontal when the tab bar sits on the left (West).
    Qt rotates text 90° by default for West/East tab bars — this undoes that.
    """

    def tabSizeHint(self, index: int) -> QSize:
        s = super().tabSizeHint(index)
        s.transpose()                  # s.height() is now the text width
        return QSize(s.height(), 66)   # bar wide enough for text, 36 px per tab

    def paintEvent(self, _event):
        painter = QStylePainter(self)
        opt     = QStyleOptionTab()
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            # Draw the tab background shape
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabShape, opt)
            # Draw text directly — horizontal, no rotation
            selected = bool(opt.state & QStyle.StateFlag.State_Selected)
            color    = THEME["accent"] if selected else THEME["text_hint"]
            painter.setPen(QPen(QColor(color)))
            painter.drawText(
                opt.rect,
                Qt.AlignmentFlag.AlignCenter,
                opt.text,
            )

class HelpDialog(QDialog):
    """Tabbed help dialog — one tab per wizard step."""

    def __init__(self, current_page_index: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DIC-HREBSD Help")
        self.setMinimumSize(780, 560)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.setTabBar(_HorizontalTabBar())
        tabs.setTabPosition(QTabWidget.TabPosition.West)

        for entry in HELP_CONTENT:
            page        = QWidget()
            page_layout = QVBoxLayout(page)
            browser     = QTextBrowser()
            browser.setOpenExternalLinks(False)
            browser.setReadOnly(True)
            browser.setHtml(_wrap_html(entry["body"]))
            page_layout.addWidget(browser)
            tabs.addTab(page, entry["title"])

        # Open on the tab matching the current wizard page
        if 0 <= current_page_index < tabs.count():
            tabs.setCurrentIndex(current_page_index)

        layout.addWidget(tabs)

        btn_row   = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)


def _wrap_html(body: str) -> str:
    """Wrap body HTML in a styled page that matches the dark theme."""
    bg   = THEME["surface_bg"]
    fg   = THEME["text"]
    acc  = THEME["accent"]
    hint = THEME["text_hint"]
    return f"""
    <html><head><style>
      body      {{ background-color:{bg}; color:{fg};
                   font-family:Arial, Helvetica; font-size:13px;
                   margin:12px; line-height:1.5; }}
      h3        {{ color:{acc}; margin-top:14px; margin-bottom:4px; }}
      b         {{ color:{fg}; }}
      li        {{ margin-bottom:4px; }}
      code      {{ background:#2a3f52; padding:1px 4px; border-radius:3px;
                   font-family:Arial; font-size:12px; }}
      hr        {{ border:none; border-top:1px solid #45475a; margin:10px 0; }}
      p         {{ color:{hint}; }}
    </style></head><body>{body}</body></html>
    """
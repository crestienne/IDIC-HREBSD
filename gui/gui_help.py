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
from PyQt6.QtGui import QColor, QPen, QFont

from gui_theme import THEME


# ─────────────────────────────────────────────────────────────────────────────
# Help content — edit this section freely
# ─────────────────────────────────────────────────────────────────────────────

HELP_CONTENT = [

    {"title": "Introduction", "body": """
<h3>IDIC-HREBSD Overview</h3>
<p>This work was supported by the ARO MURI Program
(ARO W911NF-25-2-0164).</p>

<p>HR-EBSD pipeline to determine absolute elastic strain via the use of
dynamically simulated EBSD patterns. It is implemented using an
<b>inverse-compositional Gauss-Newton</b> optimization routine to determine
the linear homography required to warp a target EBSP to match a simulated
reference EBSP in Python.</p>

<p>This code follows the HR-EBSD calculations outlined in the
<b>ATEX EBSD</b> software developed by Jean-Jacques Fundenberger and Benoit
Beausir. The linear homography approach was developed by Clement Ernould
during his PhD research, and more information can be found
<a href="#">here</a>.</p>

<p>This code builds upon the HR-EBSD implementation by <b>Dr. James Lamb</b>,
whose implementation, <i>pyHREBSD</i>, can be found
<a href="#">here</a>.</p>

<p>IDIC-HREBSD is designed specifically to support the use of dynamically
simulated reference patterns generated via <b>EMsoft</b>. This work also
utilizes a modified version of <i>EBSDtorch</i>, developed by
<b>Dr. Zachary Varley</b>, linked <a href="#">here</a>.</p>
"""},

    {"title": "Step 1 — Load Files", "body": """
<h3>Input Files</h3>
<p>The required input files for the GUI are currently a <b>.up2</b> file and a
<b>.ang</b> file.  Support for Oxford's <b>.ebsp</b> and <b>.ctf</b> file
formats is still in development.</p>
<p>The <b>.up2 file</b> includes all the raw patterns. As soon as a .up2 file is
added, the first pattern should appear in the pattern preview section. It is
highly recommended that the user double-check that this matches what they
expected to see.</p>
<p>The <b>.ang file</b> can either be a direct output from EDAX's OIM software
or an output from EMsoft's <code>EMDI.f90</code> program. It is critical that
the user has refined the orientations as much as physically possible.</p>

<hr>

<h3>Output Folder</h3>
<p>A location for the output folder must be specified. A folder will be created
in the specified location with the name specified in the <b>Run name</b>
(sub-folder). All output files will be saved in this folder.</p>

<hr>

<h3>Material Properties</h3>
<p>If you would like to apply the partial traction-free boundary condition, the
material stiffness tensor must be specified. Currently, only cubic crystals are
supported, but additional functionalities are being added.</p>

<hr>

<p>If the user has already run the IDIC-HREBSD algorithm, the results
visualization tab can be launched via the button labeled
<b>"Open Results Viewer"</b>.</p>
"""},

    {"title": "Step 2 — Scan Geometry", "body": """
<h3>Scan Geometry</h3>
<p>This section contains all the information regarding the scan geometry. Some
of the fields will be auto-populated from the uploaded .ang file, but some
must be entered manually. Critically, this includes the
<b>detector pixel size</b>.</p>
<p>The <b>scan strategy</b> must also be defined on this page. For almost all
applications, the <b>Standard</b> scan strategy is suggested. The
<b>"Apply pattern centre drift correction"</b> option should also be selected
for almost all applications, unless the overall scan size is so small that this
is negligible.</p>

<hr>

<h3>Parameters</h3>
<ul>
  <li><b>Sample Tilt:</b> the angle at which your sample is inclined, in
      degrees. Typically, the sample tilt for EBSD applications is
      ≈ 70°.</li>
  <li><b>Detector Tilt:</b> the angle measured from the horizontal at which
      your detector is inclined. Units are degrees. Positive is an angle
      above the horizontal.</li>
  <li><b>Use identity R:</b> checking this removes the transformation of the
      strain from the detector reference frame to the sample reference frame.
      All reported values will instead be in the detector reference frame.</li>
  <li><b>Xstar (x*):</b> the x coordinate of the pattern centre, defined via
      the EDAX pattern center convention. The pattern centre you utilize
      should align with the pattern you define as your reference pattern.</li>
  <li><b>Ystar (y*):</b> the y coordinate of the pattern centre, EDAX
      convention — the origin of the y pattern centre is located in the lower
      part of the screen. Again, this value should align with the pattern you
      define as your reference pattern.</li>
  <li><b>Zstar (z*):</b> the z coordinate of your pattern centre, EDAX
      convention.</li>
  <li><b>Detector Pixel Size:</b> the physical size of your pixels on the
      detector screen. For the best success using the IDIC-HREBSD algorithm,
      you should <i>not</i> bin your patterns.</li>
  <li><b>Pattern Height:</b> the height of your patterns in pixels.</li>
  <li><b>Pattern Width:</b> the width of your patterns in pixels.</li>
  <li><b>Rows:</b> the number of rows in your scan. Note that currently only
      square (rectangular) scans can be interpreted. An error will occur if you
      attempt to load a hexagonal scan.</li>
  <li><b>Columns:</b> the number of columns in your scan.</li>
  <li><b>Scan Strategy:</b> the scanning strategy utilized. The
      <b>Standard</b> scan strategy should be used most of the time. Incorrect
      selection of the scan strategy will affect your pattern centre
      correction.</li>
</ul>
"""},

    {"title": "Step 3 — Pattern Processing", "body": """
<h3>Pattern Processing</h3>
<p>It is advisable that all patterns be processed prior to the application of
the IDIC-HREBSD algorithm. The primary levers can be adjusted in the upper-left
section of the screen. The resultant pattern before and after pre-processing is
shown, with the pattern on top, labelled <b>raw pattern</b>, being an
unprocessed version of the pattern shown below it, labelled
<b>filtered pattern</b>.</p>

<hr>

<h3>Parameters</h3>
<ul>
  <li><b>High Pass Sigma:</b> the sigma value used to filter the patterns. A
      good starting value is around 10.</li>
  <li><b>Low Pass Sigma:</b> the sigma value for the low-pass filter; a good
      value is around 1.</li>
  <li><b>Gamma:</b> the gamma correction factor.</li>
  <li><b>Mask Type:</b> currently two kinds of masks are implemented — a
      <b>circular</b> mask and a <b>centre-cross</b> mask.</li>
  <li><b>Crop Fraction:</b> the size of the region utilised for IC-GN. The
      default value is 0.8. When the mask is circular, the crop region becomes
      a disc inscribed inside the pattern, centered on the geometric image
      centre; otherwise the crop region is a centred square.</li>
</ul>
"""},

    {"title": "Step 4 — Segmentation & ROI", "body": """
<h3>Grain Segmentation &amp; Region of Interest</h3>
<p>If the whole scan is too large, the option in this step is to select a
region of interest by manually setting the starting and stopping rows and
columns of the scan. The resultant region of interest will then be outlined
on the IPF map.</p>
<p>If grain segmentation is utilized, via the section labelled
<b>Grain Segmentation</b> in the upper right, the option also exists to
select a region of interest based on a grain. This is achieved by first setting
the <b>misorientation threshold</b> and a <b>minimum number of pixels</b>
allowable within a grain. Clicking a grain on the Grain ID Map dialog ticks its
checkbox in the legend, and the <b>Select Region of Interest</b> button
finalises the ROI — non-selected grains are whited out on both the grain map
and the IPF map.</p>

<hr>

<h3>Boundaries and KAM</h3>
<p>Grain boundaries are drawn in black on top of both the IPF map and the
grain ID map.  A KAM (kernel-average misorientation) map is also computed
during segmentation — open it via the <b>Show KAM Map…</b> button.</p>
"""},

    {"title": "Step 5 — Reference Pattern", "body": """
<h3>Reference Pattern Selection</h3>
<p>The reference pattern can be selected either manually via the
<b>Reference Position</b> spin boxes (row and column) or by clicking on the IPF
map shown on the left.</p>
<p>In addition to using a single real reference pattern, the option also exists
to utilize a <b>simulated reference pattern</b> generated via HREBSD.py /
SimPatGen, or to use one <b>reference pattern per grain</b> when a grain ROI
has been selected on Step 4.  The Reference Pattern Type (Real / Simulated)
and Reference Pattern Count (Single / Multiple) toggles control which mode is
active.</p>

<hr>

<h3>Tips</h3>
<ul>
  <li>The reference-pattern viewer on the lower-left shows the
      <b>Step-3-processed</b> pattern — the same image the IC-GN solver will
      see — for the currently-selected reference position.</li>
  <li>If you click a pixel outside the Step-4 grain ROI, the GUI will warn
      you before setting the reference there.</li>
  <li>The <b>Live Tuner</b> dialog lets you interactively adjust Euler angles
      and PC for a simulated reference and see the checkerboard / flicker view
      update in real time.</li>
</ul>
"""},

    {"title": "Step 6 — Run", "body": """
<h3>Optimization Parameters &amp; Run</h3>
<p>Step 6 sets all the optimization parameters:</p>
<ul>
  <li><b>Max iterations:</b> the total allowable iterations per pattern. The
      default of 150 iterations is often sufficient.</li>
  <li><b>CPU threads:</b> the IC-GN pipeline is parallelised on the CPU, so
      this controls the total number of CPU threads used. The spin box is
      capped at the number of logical CPUs reported by your machine.</li>
  <li><b>Init type:</b> the type of initial-guess procedure to use. The three
      options are <code>none</code>, <code>partial</code>, and <code>full</code>.
      It is advisable to use the <b>full</b> initial-guess procedure (the
      default).</li>
  <li><b>Optimizer:</b> two optimisers are available. The first is the
      traditional IC-GN; the second reverses the roles of reference and
      target patterns (experimental).</li>
</ul>
<p>Once all the parameters have been set, the optimization is run via the
green <b>▶ Run Pipeline</b> button. A progress bar tracks per-pattern
completion as joblib dispatches the work across the CPU threads.</p>
"""},

    {"title": "Reference Frames", "body": "<p>TO DO</p>"},
    {"title": "Acknowledgements", "body": "<p>TO DO</p>"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Dialog — no need to edit below this line
# ─────────────────────────────────────────────────────────────────────────────

class _HorizontalTabBar(QTabBar):
    """
    Keeps tab labels horizontal when the tab bar sits on the left (West).
    Qt rotates text 90° by default for West/East tab bars — this undoes that.
    """

    # Slightly larger font than the wizard default so the tab labels are
    # easy to read on the dark side bar.
    _TAB_FONT_POINT = 13

    def tabSizeHint(self, index: int) -> QSize:
        s = super().tabSizeHint(index)
        s.transpose()                  # s.height() is now the text width
        # Make the tab strip a bit wider + taller to accommodate the larger font.
        return QSize(s.height() + 24, 72)

    def paintEvent(self, _event):
        painter = QStylePainter(self)
        opt     = QStyleOptionTab()
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            # Draw the tab background shape
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabShape, opt)
            # All labels in white; selected tab gets bold weight as the only
            # visual cue for which tab is active.
            selected = bool(opt.state & QStyle.StateFlag.State_Selected)
            font = QFont(self.font())
            font.setPointSize(self._TAB_FONT_POINT)
            font.setBold(selected)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#ffffff")))
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
    fg   = "#ffffff"   # bumped to pure white for max readability
    acc  = THEME["accent"]
    return f"""
    <html><head><style>
      body      {{ background-color:{bg}; color:{fg};
                   font-family:Arial, Helvetica; font-size:16px;
                   margin:14px; line-height:1.55; }}
      h3        {{ color:{acc}; font-size:19px;
                   margin-top:16px; margin-bottom:6px; }}
      p, li, ul, ol, a {{ color:{fg}; font-size:16px; }}
      b         {{ color:{fg}; }}
      li        {{ margin-bottom:5px; }}
      code      {{ background:#2a3f52; color:{fg};
                   padding:1px 4px; border-radius:3px;
                   font-family:Arial; font-size:15px; }}
      hr        {{ border:none; border-top:1px solid #45475a; margin:12px 0; }}
    </style></head><body>{body}</body></html>
    """
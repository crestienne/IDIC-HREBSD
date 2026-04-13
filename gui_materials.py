"""
gui_materials.py — shared material preset loader.

Kept in its own module so that both gui_pages.py and gui_visualization.py
can import from it without creating a circular dependency.
"""

import os
import json

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

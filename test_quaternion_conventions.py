"""
test_quaternion_conventions.py — Compare HREBSD.py's Bunge→quaternion path
against the project's existing canonical EMsoft form in rotations.py.

Run from the project root:

    python test_quaternion_conventions.py

If the two implementations agree, every line will print "OK".
If they don't, you'll see which test case fails and how.

The two implementations under test:

    HREBSD.eu2qu(E, "ZXZ")          — chains om2qu(eu2om(E, "ZXZ"))
                                       (active-rotation matrix → quaternion)
    rotations.eu2qu(E)              — the EMsoft P=+1 canonical formula
                                       (passive-quaternion form, matches
                                        ebsdtorch's bu2qu)

These two CAN be either:
  (a) identical (same sign convention)
  (b) conjugates: q_a = (w, x, y, z) vs q_b = (w, -x, -y, -z)
      — meaning they encode inverse rotations
  (c) something else entirely (e.g. axis swap) — would be a real bug
"""

import numpy as np
import torch

# Import the two implementations we want to compare
from HREBSD import eu2qu as hrebsd_eu2qu
from HREBSD import bu2qu_emsoft
from rotations import eu2qu as rotations_eu2qu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hrebsd_quat(phi1_rad, Phi_rad, phi2_rad):
    """Call HREBSD.eu2qu and return a 4-element numpy array (w, x, y, z)."""
    E = torch.tensor([[phi1_rad, Phi_rad, phi2_rad]], dtype=torch.float64)
    q = hrebsd_eu2qu(E, "ZXZ").detach().cpu().numpy().ravel()
    return q


def hrebsd_bu2qu(phi1_rad, Phi_rad, phi2_rad):
    """Call HREBSD.bu2qu_emsoft (the EMsoft-canonical helper)."""
    E = torch.tensor([[phi1_rad, Phi_rad, phi2_rad]], dtype=torch.float64)
    q = bu2qu_emsoft(E).detach().cpu().numpy().ravel()
    return q


def rotations_quat(phi1_rad, Phi_rad, phi2_rad):
    """Call rotations.eu2qu (EMsoft canonical) and return (w, x, y, z)."""
    eu = np.array([phi1_rad, Phi_rad, phi2_rad])[None, :]
    q = rotations_eu2qu(eu).ravel()
    return q


def fmt(q):
    """Pretty-print a quaternion."""
    return "(" + ", ".join(f"{x:+.6f}" for x in q) + ")"


def relation(qa, qb, tol=1e-6):
    """
    Classify the relationship between two quaternions:
      'same'        → qa ≈ qb
      'opposite'    → qa ≈ -qb           (same rotation, sign-flipped)
      'conjugate'   → qa ≈ (w, -x,-y,-z) (INVERSE rotation — the bug we expect)
      'neg-conj'    → qa ≈ (-w, x, y, z) (conjugate, sign-flipped)
      'different'   → none of the above
    """
    if np.allclose(qa,  qb, atol=tol):       return "same"
    if np.allclose(qa, -qb, atol=tol):       return "opposite"
    qa_conj = qa * np.array([1.0, -1.0, -1.0, -1.0])
    if np.allclose(qa_conj,  qb, atol=tol):  return "conjugate"
    if np.allclose(qa_conj, -qb, atol=tol):  return "neg-conj"
    return "different"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

DEG = np.pi / 180.0

# A handful of representative Bunge triples (degrees → radians).
# The first one is the identity (no rotation).  The others are arbitrary
# but cover positive/negative angles and the ambiguous-axis case at Φ=0.
CASES_DEG = [
    ( 0.0,   0.0,  0.0),     # identity
    (45.0,  30.0, 60.0),     # generic
    (90.0,  45.0, 90.0),     # 90° rotations
    (10.0,   5.0, 15.0),     # small angles
    ( 0.0,  70.0,  0.0),     # sample-tilt-like
    (123.4, 45.6, 78.9),     # arbitrary
]


# ---------------------------------------------------------------------------
# Run the comparisons
# ---------------------------------------------------------------------------

print("Test 1: HREBSD.eu2qu  vs  rotations.eu2qu (sanity check — expected MISMATCH)")
print(f"{'φ1':>6} {'Φ':>6} {'φ2':>6}    "
      f"{'HREBSD.eu2qu':<38} {'rotations.eu2qu':<38} relation")
print("-" * 110)

verdict_counts = {}

for phi1_deg, Phi_deg, phi2_deg in CASES_DEG:
    phi1, Phi, phi2 = phi1_deg * DEG, Phi_deg * DEG, phi2_deg * DEG

    q_hrebsd    = hrebsd_quat(phi1, Phi, phi2)
    q_rotations = rotations_quat(phi1, Phi, phi2)
    rel         = relation(q_hrebsd, q_rotations)

    print(f"{phi1_deg:6.1f} {Phi_deg:6.1f} {phi2_deg:6.1f}    "
          f"{fmt(q_hrebsd):<38} {fmt(q_rotations):<38} {rel}")

    verdict_counts[rel] = verdict_counts.get(rel, 0) + 1

# Test 2: the fix — bu2qu_emsoft should match rotations.eu2qu exactly.
print()
print("Test 2: HREBSD.bu2qu_emsoft  vs  rotations.eu2qu (should all be 'same')")
print(f"{'φ1':>6} {'Φ':>6} {'φ2':>6}    "
      f"{'HREBSD.bu2qu_emsoft':<38} {'rotations.eu2qu':<38} relation")
print("-" * 110)

fix_counts = {}

for phi1_deg, Phi_deg, phi2_deg in CASES_DEG:
    phi1, Phi, phi2 = phi1_deg * DEG, Phi_deg * DEG, phi2_deg * DEG

    q_fixed     = hrebsd_bu2qu(phi1, Phi, phi2)
    q_rotations = rotations_quat(phi1, Phi, phi2)
    rel         = relation(q_fixed, q_rotations)

    print(f"{phi1_deg:6.1f} {Phi_deg:6.1f} {phi2_deg:6.1f}    "
          f"{fmt(q_fixed):<38} {fmt(q_rotations):<38} {rel}")

    fix_counts[rel] = fix_counts.get(rel, 0) + 1


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

print()
print("Test 1 summary:", verdict_counts)
print("Test 2 summary:", fix_counts)
print()

ok_test2 = list(fix_counts.keys()) == ["same"]
if ok_test2:
    print("OK — bu2qu_emsoft matches rotations.eu2qu for every case.")
    print("     SimPatGen now produces the same orientation convention as the")
    print("     rest of the project.")
else:
    print("FAIL — bu2qu_emsoft does NOT match rotations.eu2qu in some case.")
    print("       This means the formula in HREBSD.bu2qu_emsoft was copied")
    print("       incorrectly — re-check the +/- signs against rotations.py.")

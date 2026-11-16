"""Anchor-point coverage under the strict TAL rule, with signed depth (single source of truth for all analyses).

An anchor point c is a TAL candidate for box (x1, y1, x2, y2) only if it lies strictly inside
(vendored select_candidates_in_gts: min(cx-x1, cy-y1, x2-cx, y2-cy) > 1e-9). The signed depth of an anchor is
that minimum; the best depth m of a box is its maximum over all anchors of all levels.

  m > 0  inside : at least one strict candidate
  m = 0  edge   : best anchor lies exactly on the box boundary (no candidate, but LTRB-representable)
  m < 0  outside: nearest anchor outside the box (no candidate; IoU ceiling < 1 for non-negative LTRB)

AI-TOD boxes are ~100% integer and anchor centres are integers (P3: 4 mod 8, P4: 8 mod 16, P5: 16 mod 32), so m is
an integer; it is rounded to 1e-6 so float noise such as 355.99999999999994 cannot flip a label.
"""
import math
import re

INPUT = 800
BASE_STRIDES = (8, 16, 32)   # native SpikeYOLO P3-P5
P2_STRIDES = (4, 8, 16, 32)  # dense P2 head added to P3-P5
TOL = 1e-6


def axis_depth(lo, hi, s, n, offset=0.0):
    """max over anchors c=(k+0.5)s - offset, k in [0, n), of min(c - lo, hi - c)."""
    k = int(math.floor((lo + hi) / 2 / s + offset / s))  # anchor index nearest the centre
    best = -math.inf
    for kk in (k - 1, k, k + 1):
        kk = min(max(kk, 0), n - 1)
        c = (kk + 0.5) * s - offset
        best = max(best, min(c - lo, hi - c))
    return best


def best_depth(box, strides=BASE_STRIDES, offset=(0.0, 0.0), input_size=INPUT):
    x1, y1, x2, y2 = box
    m = -math.inf
    for s in strides:
        n = input_size // s
        m = max(m, min(axis_depth(x1, x2, s, n, offset[0]), axis_depth(y1, y2, s, n, offset[1])))
    return round(m, 6)


def stratum(m):
    return 'inside' if m > TOL else ('edge' if m >= -TOL else 'outside')


def level_depths(box, strides=BASE_STRIDES, offset=(0.0, 0.0), input_size=INPUT):
    return {s: best_depth(box, (s,), offset, input_size) for s in strides}


def iou_ceiling(box, strides=BASE_STRIDES, input_size=INPUT):
    """Best IoU any anchor can reach with non-negative LTRB (the box must contain its anchor point)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    best = 0.0
    for s in strides:
        n = input_size // s
        dx = max(0.0, -axis_depth(x1, x2, s, n))
        dy = max(0.0, -axis_depth(y1, y2, s, n))
        best = max(best, (w * h) / ((w + dx) * (h + dy)))
    return best


def source_and_parent(file_name):
    """AI-TOD-v2 mother dataset and parent scene from the file name (crops of one scene share a parent)."""
    fn = file_name
    if re.match(r'^P\d{4}__', fn):
        return 'DOTA', 'DOTA:' + fn.split('__')[0]
    if re.match(r'^\d{7}_\d{5}_d_\d{7}__', fn):
        return 'VisDrone', 'VisDrone:' + fn.split('__')[0]
    if re.match(r'^\d+__\d+_\d+\.', fn):
        return 'xView', 'xView:' + fn.split('__')[0]
    if re.match(r'^[0-9a-f]{9}\.', fn):
        return 'Airbus', 'Airbus:' + fn
    if re.match(r'^\d{5}\.', fn):
        return 'DIOR', 'DIOR:' + fn
    return 'other', 'other:' + fn

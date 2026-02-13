"""Bracketing for line search (mnbrak-style, Numerical Recipes)."""
from __future__ import annotations

GOLD = 1.618034
TINY = 1e-20


def mnbrak(f, ax: float, bx: float, cx: float = None):
    """
    Bracketing triplet (ax, bx, cx) with ax < bx < cx and f(bx) < f(ax), f(bx) < f(cx).
    Returns (ax, bx, cx, fa, fb, fc).
    """
    if cx is None:
        cx = bx + (bx - ax) * GOLD
    fa = f(ax)
    fb = f(bx)
    fc = f(cx)
    while fb > fa or fb > fc:
        if fb > fa:
            ax, bx = bx, bx + GOLD * (bx - ax)
            fa, fb = fb, f(bx)
        else:
            cx, bx = bx, bx - (1.0 / GOLD) * (cx - bx)
            fc, fb = fb, f(bx)
        if ax > cx:
            ax, cx = cx, ax
            fa, fc = fc, fa
    return ax, bx, cx, fa, fb, fc

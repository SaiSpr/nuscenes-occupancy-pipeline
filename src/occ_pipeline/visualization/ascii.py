from __future__ import annotations

import numpy as np


def ascii_topdown(occupancy_zyx: np.ndarray, max_width: int = 80) -> str:
    """Return a compact top-down map where # means at least one occupied z bin."""
    topdown = occupancy_zyx.max(axis=0)
    y_bins, x_bins = topdown.shape
    stride = max(1, int(np.ceil(x_bins / max_width)))
    sampled = topdown[::stride, ::stride]
    rows = ["".join("#" if value else "." for value in row) for row in sampled[::-1]]
    return "\n".join(rows)


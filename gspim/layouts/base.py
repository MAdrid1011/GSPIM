"""Shared validation helpers for representation adapters."""

def require_window(window_start: float, window_end: float) -> tuple[float, float]:
    """Validate the paper's explicit half-open temporal window [start, end)."""

    start, end = float(window_start), float(window_end)
    if end <= start:
        raise ValueError("temporal window end must exceed its start")
    return start, end

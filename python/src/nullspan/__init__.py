"""nullspan — Calibrated p_null abstention, grounded [start, end] span extraction, System One typed decisions, and distribution forecasting."""
from .engine import DEFAULT_HF_REPO, NullspanExtractor, decide, extract

__version__ = "1.2.1"
__all__ = ["extract", "decide", "NullspanExtractor", "DEFAULT_HF_REPO"]

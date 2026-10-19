"""nullspan — Calibrated p_null abstention, grounded [start, end] span extraction, and distribution forecasting."""
from .engine import DEFAULT_HF_REPO, NullspanExtractor, extract

__version__ = "1.0.0"
__all__ = ["extract", "NullspanExtractor", "DEFAULT_HF_REPO"]

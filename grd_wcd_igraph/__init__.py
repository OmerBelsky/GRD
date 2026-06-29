"""Standalone iGraph-based WCD implementation.

This package is intentionally additive and does not modify the legacy offline
WCD implementation.
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]

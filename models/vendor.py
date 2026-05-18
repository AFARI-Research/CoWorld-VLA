"""Helpers for vendored third-party model code."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_vggt_import_path() -> Path:
    """Make the vendored VGGT package importable as ``vggt``."""
    vendor_root = Path(__file__).resolve().parent / "vggt"
    package_root = vendor_root / "vggt"
    if not package_root.is_dir():
        raise FileNotFoundError(
            "Vendored VGGT package not found. Expected source at "
            f"{package_root}."
        )
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)
    return vendor_root

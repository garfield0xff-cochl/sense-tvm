# SPDX-License-Identifier: Apache-2.0
"""
Transform Utilities

Helper functions for Sense transformation passes.
"""

from typing import Tuple, Dict
import numpy as np


def get_buffer_size(shape: Tuple, dtype: str = "float32") -> int:
    """Calculate buffer size in bytes.

    Parameters
    ----------
    shape : Tuple
        Tensor shape.
    dtype : str
        Data type.

    Returns
    -------
    int
        Size in bytes.
    """
    dtype_sizes = {
        "float32": 4,
        "float16": 2,
        "int32": 4,
        "int8": 1,
        "uint8": 1,
    }
    element_size = dtype_sizes.get(dtype, 4)
    return int(np.prod(shape)) * element_size


def get_total_memory(buffers: Dict[str, Tuple]) -> int:
    """Calculate total memory for all buffers.

    Parameters
    ----------
    buffers : Dict[str, Tuple]
        Dictionary of buffer name to shape.

    Returns
    -------
    int
        Total size in bytes.
    """
    total = 0
    for name, shape in buffers.items():
        total += get_buffer_size(shape)
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form.

    Parameters
    ----------
    size_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Formatted string (e.g., "1.5 MB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.2f} MB"

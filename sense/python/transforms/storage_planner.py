# SPDX-License-Identifier: Apache-2.0
"""
Static Storage Planner

Analyzes buffer liveness and generates static storage layout.
Implements TVM MCU Strategy: Partial Graph AOT with static memory planning.

Reference: model_legacy/docs2/TVM_MCU_GAP_ANALYSIS.md Section 2
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class BufferInfo:
    """Information about a buffer/tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    first_use: int  # First operation that uses this buffer
    last_use: int   # Last operation that uses this buffer
    offset: int = 0  # Offset in unified buffer (set by planner)


@dataclass
class StoragePlan:
    """Static storage allocation plan."""
    buffers: List[BufferInfo]
    total_size: int
    peak_usage: int
    reuse_count: int  # Number of buffer reuses


class StaticStoragePlanner:
    """Plan static storage layout with buffer reuse.

    Based on liveness analysis to minimize memory footprint.
    Implements TVM MCU Strategy for static memory management.
    """

    def __init__(self):
        self.buffers = []
        self.operations = []

    def add_buffer(self, name: str, shape: Tuple, dtype: str = "float32"):
        """Register a buffer for planning."""
        dtype_size = {"float32": 4, "int32": 4, "int8": 1, "uint8": 1}
        size_bytes = int(np.prod(shape)) * dtype_size.get(dtype, 4)

        buffer = BufferInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            first_use=-1,
            last_use=-1
        )
        self.buffers.append(buffer)
        return buffer

    def set_buffer_lifetime(self, name: str, first_use: int, last_use: int):
        """Set when a buffer is first and last used."""
        for buf in self.buffers:
            if buf.name == name:
                buf.first_use = first_use
                buf.last_use = last_use
                break

    def plan_storage(self) -> StoragePlan:
        """Plan storage layout using greedy bin packing with liveness.

        Returns
        -------
        StoragePlan
            Static storage plan with buffer offsets.
        """
        # Sort buffers by first use
        sorted_buffers = sorted(self.buffers, key=lambda b: b.first_use)

        # Track allocated regions: (start_offset, end_offset, last_use_time)
        allocated_regions = []
        current_offset = 0
        peak_offset = 0
        reuse_count = 0

        for buf in sorted_buffers:
            # Find a free region that's no longer in use
            placed = False
            for i, (start, end, last_use) in enumerate(allocated_regions):
                # Can reuse this region if its last use < current buffer's first use
                if last_use < buf.first_use:
                    region_size = end - start
                    if region_size >= buf.size_bytes:
                        # Reuse this region
                        buf.offset = start
                        allocated_regions[i] = (start, start + buf.size_bytes, buf.last_use)
                        placed = True
                        reuse_count += 1
                        break

            if not placed:
                # Allocate new region
                buf.offset = current_offset
                allocated_regions.append((current_offset, current_offset + buf.size_bytes, buf.last_use))
                current_offset += buf.size_bytes
                peak_offset = max(peak_offset, current_offset)

        return StoragePlan(
            buffers=self.buffers,
            total_size=peak_offset,
            peak_usage=peak_offset,
            reuse_count=reuse_count
        )

    def generate_c_declarations(self, plan: StoragePlan) -> str:
        """Generate C code for static storage declarations."""
        code = f'''/*========================================
 * Static Storage Layout (TVM MCU Strategy)
 * Total: {plan.total_size / 1024 / 1024:.2f} MB
 * Buffers: {len(plan.buffers)}
 * Reused: {plan.reuse_count}
 *========================================*/
#define UNIFIED_BUFFER_SIZE {plan.total_size}
static float __attribute__((aligned(64))) g_unified_buffer[UNIFIED_BUFFER_SIZE / sizeof(float)];

/* Buffer offsets (compile-time constants) */
'''
        for buf in plan.buffers:
            offset_floats = buf.offset // 4
            code += f'#define {buf.name.upper()}_OFFSET {offset_floats}  // {buf.size_bytes / 1024:.1f} KB\n'

        code += '\n/* Buffer getters (inlined to pointers) */\n'
        for buf in plan.buffers:
            code += f'#define GET_{buf.name.upper()}() (&g_unified_buffer[{buf.name.upper()}_OFFSET])\n'

        return code


def analyze_liveness(operations: List[Dict], tensor_shapes: Dict) -> Dict[str, Tuple[int, int]]:
    """Analyze buffer liveness from operations.

    Parameters
    ----------
    operations : List[Dict]
        List of operation dictionaries.
    tensor_shapes : Dict
        Tensor name to shape mapping.

    Returns
    -------
    liveness : Dict[str, Tuple[int, int]]
        Buffer name to (first_use, last_use) mapping.
    """
    liveness = {}

    for op_idx, op in enumerate(operations):
        args = op.get('args', [])

        # Track all buffer references
        for arg_type, arg_val in args:
            if arg_type == 'alloc':
                buffer_name = arg_val

                if buffer_name not in liveness:
                    liveness[buffer_name] = (op_idx, op_idx)
                else:
                    first, _ = liveness[buffer_name]
                    liveness[buffer_name] = (first, op_idx)

    return liveness

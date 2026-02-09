# SPDX-License-Identifier: Apache-2.0
"""
IR Analyzer using PyExprVisitor

Extracts buffer lifetimes, operations, and constant indices from Relax IR
using TVM's PyExprVisitor for accurate and automated analysis.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor, visitor


@dataclass
class BufferLifetime:
    """Buffer lifetime information."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    first_use: int
    last_use: int
    size_bytes: int


@dataclass
class OperationInfo:
    """Operation information extracted from IR."""
    index: int
    op_name: str
    var_name: str
    input_buffers: List[str]
    output_buffer: Optional[str]
    weight_indices: List[int]


@visitor
class IRAnalyzer(PyExprVisitor):
    """Complete IR analyzer tracking buffers, operations, and constants.

    Extracts:
    - Buffer lifetimes (first/last use)
    - Operation sequence with inputs/outputs
    - Constant indices and data
    - Kill operations
    """

    def __init__(self):
        super().__init__()
        self.operations = []
        self.current_op_idx = 0
        self.buffer_first_def = {}  # var_name -> op_idx
        self.buffer_last_use = {}   # var_name -> op_idx
        self.var_shapes = {}        # var_name -> shape
        self.constants = []         # Constant data
        self.const_index_map = {}   # data_id -> index
        self.var_to_const = {}      # var_name -> const_index (for reshaped constants)

    def visit_var_binding_(self, binding):
        """Visit variable binding to track everything."""
        var_name = binding.var.name_hint

        # Record definition
        if var_name not in self.buffer_first_def:
            self.buffer_first_def[var_name] = self.current_op_idx
        self.buffer_last_use[var_name] = self.current_op_idx

        # Get shape
        if hasattr(binding.var, 'struct_info'):
            struct_info = binding.var.struct_info
            if hasattr(struct_info, 'shape'):
                try:
                    shape = tuple(int(d) for d in struct_info.shape)
                    self.var_shapes[var_name] = shape
                except:
                    pass

        # Visit the value
        if isinstance(binding.value, relax.Call):
            call = binding.value

            # Get operation name
            op_name = "unknown"
            if hasattr(call.op, 'name_hint'):
                op_name = call.op.name_hint
            elif hasattr(call.op, 'global_symbol'):
                op_name = call.op.global_symbol
                if '.' in op_name:
                    op_name = op_name.split('.')[-1]

            # Track inputs and constants
            input_buffers = []
            weight_indices = []

            for arg in call.args:
                if isinstance(arg, relax.Var):
                    input_var = arg.name_hint
                    input_buffers.append(input_var)
                    self.buffer_last_use[input_var] = self.current_op_idx
                elif isinstance(arg, relax.Constant):
                    # Found a constant argument - record it
                    try:
                        const_data = arg.data.numpy()
                        const_id = id(const_data)

                        if const_id not in self.const_index_map:
                            const_idx = len(self.constants)
                            self.constants.append({
                                'index': const_idx,
                                'data': const_data,
                                'shape': const_data.shape,
                                'size': const_data.size
                            })
                            self.const_index_map[const_id] = const_idx

                        weight_indices.append(self.const_index_map[const_id])
                    except:
                        weight_indices.append(-1)

            self.operations.append(OperationInfo(
                index=self.current_op_idx,
                op_name=op_name,
                var_name=var_name,
                input_buffers=input_buffers,
                output_buffer=var_name,
                weight_indices=weight_indices
            ))

        self.current_op_idx += 1
        self.visit_expr(binding.value)

    def visit_binding_block_(self, block):
        """Visit all bindings in order."""
        for binding in block.bindings:
            self.visit_binding(binding)

    def get_buffer_lifetimes(self) -> List[BufferLifetime]:
        """Get buffer lifetime information with shapes."""
        lifetimes = []

        for var_name in self.buffer_first_def.keys():
            if var_name in self.var_shapes:
                shape = self.var_shapes[var_name]
                size_bytes = int(np.prod(shape)) * 4

                lifetimes.append(BufferLifetime(
                    name=var_name,
                    shape=shape,
                    dtype="float32",
                    first_use=self.buffer_first_def.get(var_name, 0),
                    last_use=self.buffer_last_use.get(var_name, 0),
                    size_bytes=size_bytes
                ))

        # Sort by first use
        lifetimes.sort(key=lambda x: x.first_use)
        return lifetimes


def analyze_ir_with_visitor(ir_mod) -> Tuple[List[BufferLifetime], List[OperationInfo], List[Dict]]:
    """Analyze IR using PyExprVisitor to extract buffer lifetimes and operations.

    Parameters
    ----------
    ir_mod : tvm.IRModule
        Relax IR module.

    Returns
    -------
    buffer_lifetimes : List[BufferLifetime]
        Buffer lifetime information.
    operations : List[OperationInfo]
        Operation information.
    constants : List[Dict]
        Constant information.
    """
    main_func = ir_mod["main"]

    # Use IRAnalyzer to get everything
    analyzer = IRAnalyzer()
    analyzer.visit_expr(main_func)

    buffer_lifetimes = analyzer.get_buffer_lifetimes()
    operations = analyzer.operations
    constants = analyzer.constants

    return buffer_lifetimes, operations, constants

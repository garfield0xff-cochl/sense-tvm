# SPDX-License-Identifier: Apache-2.0
"""
Unified IR Extractor

Single PyExprVisitor that extracts everything:
- IR operation sequence (388 compute ops)
- Constants with auto-matching (352 constants)
- Buffer lifetimes (494 buffers)
- Tensor shapes

One visit, complete extraction.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

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
class IROperation:
    """Single compute operation."""
    index: int
    op_name: str
    output_buffer: str
    input_buffers: List[str]
    weight_indices: List[int]
    output_shape: Tuple


@dataclass
class ConstantInfo:
    """Constant information."""
    index: int
    data: np.ndarray
    shape: Tuple
    size: int


@visitor
class UnifiedExtractor(PyExprVisitor):
    """Extract everything in one pass.

    Extracts:
    - Compute operations (cls.xxx)
    - Constants (weights)
    - Buffer lifetimes
    - Shapes
    """

    def __init__(self):
        super().__init__()
        # Operations
        self.operations = []
        self.op_idx = 0

        # Constants
        self.constants = []

        # Buffers
        self.buffer_first_def = {}
        self.buffer_last_use = {}
        self.var_shapes = {}

    def visit_var_binding_(self, binding):
        """Visit each binding."""
        var_name = binding.var.name_hint
        value = binding.value

        # Track buffer definition
        if var_name not in self.buffer_first_def:
            self.buffer_first_def[var_name] = self.op_idx
        self.buffer_last_use[var_name] = self.op_idx

        # Get shape
        if hasattr(binding.var, 'struct_info') and hasattr(binding.var.struct_info, 'shape'):
            try:
                shape = tuple(int(d) for d in binding.var.struct_info.shape)
                self.var_shapes[var_name] = shape
            except:
                pass

        if not isinstance(value, relax.Call):
            self.op_idx += 1
            self.visit_expr(value)
            return

        # Get op name
        op_name = self._get_op_name(value)

        # Filter: Only compute operations
        is_compute = (
            hasattr(value.op, 'name_hint') and
            not op_name.startswith('alloc') and
            not op_name.startswith('kill') and
            not op_name.startswith('check') and
            not op_name.startswith('match') and
            op_name not in ['null_value', 'unknown', 'reshape']
        )

        if is_compute:
            # Get output buffer (last argument)
            output_buffer = None
            if len(value.args) > 0:
                last_arg = value.args[-1]
                if isinstance(last_arg, relax.Var):
                    output_buffer = last_arg.name_hint

            # Get inputs and constants
            input_buffers = []
            weight_indices = []

            for arg in value.args[:-1]:  # Exclude last (output)
                if isinstance(arg, relax.Var):
                    input_buffers.append(arg.name_hint)
                    self.buffer_last_use[arg.name_hint] = self.op_idx
                elif isinstance(arg, relax.Constant):
                    # Extract constant
                    try:
                        const_data = arg.data.numpy()
                        const_idx = len(self.constants)
                        self.constants.append(ConstantInfo(
                            index=const_idx,
                            data=const_data,
                            shape=const_data.shape,
                            size=const_data.size
                        ))
                        weight_indices.append(const_idx)
                    except:
                        pass

            # Get output shape
            output_shape = None
            if output_buffer and output_buffer in self.var_shapes:
                output_shape = self.var_shapes[output_buffer]
            elif hasattr(binding.var, 'struct_info'):
                try:
                    output_shape = tuple(int(d) for d in binding.var.struct_info.shape)
                except:
                    pass

            self.operations.append(IROperation(
                index=len(self.operations),
                op_name=op_name,
                output_buffer=output_buffer if output_buffer else var_name,
                input_buffers=input_buffers,
                weight_indices=weight_indices,
                output_shape=output_shape
            ))

        self.op_idx += 1
        self.visit_expr(value)

    def visit_binding_block_(self, block):
        """Visit all bindings."""
        for binding in block.bindings:
            self.visit_binding(binding)

    def _get_op_name(self, call):
        """Get operation name."""
        if hasattr(call.op, 'name_hint'):
            return call.op.name_hint
        elif hasattr(call.op, 'global_symbol'):
            name = call.op.global_symbol
            return name.split('.')[-1] if '.' in name else name
        return "unknown"

    def get_buffer_lifetimes(self) -> List[BufferLifetime]:
        """Get buffer lifetime information."""
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

        lifetimes.sort(key=lambda x: x.first_use)
        return lifetimes


def extract_tir_functions(ir_mod) -> Dict[str, any]:
    """Extract TIR function bodies for inlining.

    Returns
    -------
    tir_functions : Dict[str, PrimFunc]
        TIR function name to PrimFunc mapping.
    """
    from tvm.tir import PrimFunc

    tir_funcs = {}
    for gvar, func in ir_mod.functions.items():
        if isinstance(func, PrimFunc):
            func_name = gvar.name_hint
            tir_funcs[func_name] = func

    return tir_funcs


def extract_all(ir_mod, extract_tir: bool = False) -> Tuple[List[IROperation], List[ConstantInfo], List[BufferLifetime], Dict, Optional[Dict]]:
    """Extract everything from IR in one pass.

    Parameters
    ----------
    ir_mod : tvm.IRModule
        Relax IR module.
    extract_tir : bool
        If True, also extract TIR function bodies for inlining.

    Returns
    -------
    operations : List[IROperation]
        388 compute operations.
    constants : List[ConstantInfo]
        352 constants (in order).
    buffer_lifetimes : List[BufferLifetime]
        494 buffers with lifetimes.
    var_shapes : Dict
        Variable name to shape mapping.
    tir_functions : Optional[Dict]
        TIR function bodies (if extract_tir=True).
    """
    extractor = UnifiedExtractor()
    extractor.visit_expr(ir_mod["main"])

    tir_funcs = None
    if extract_tir:
        tir_funcs = extract_tir_functions(ir_mod)

    return (
        extractor.operations,
        extractor.constants,
        extractor.get_buffer_lifetimes(),
        extractor.var_shapes,
        tir_funcs
    )

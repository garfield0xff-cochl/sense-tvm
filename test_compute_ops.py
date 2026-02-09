#!/usr/bin/env python3
"""Test compute operations count"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sense.python.parser import parse_model
from sense.python.ir_sequence_extractor import extract_ir_sequence
from tvm import relax

# Parse and optimize
model_path = "sense_onnx/model_main_17.onnx"
print(f"Parsing {model_path}...")
ir_mod, _, _, _ = parse_model(model_path)

print(f"Optimizing...")
ir_mod = relax.get_pipeline("default_build")(ir_mod)

# Extract sequence
print(f"\nExtracting IR sequence...")
ir_sequence, var_shapes = extract_ir_sequence(ir_mod)

print(f"\nResults:")
print(f"  Total IR bindings: {len(ir_sequence)}")

# Filter compute ops
compute_ops = [op for op in ir_sequence if not op.is_kill and
               op.op_name not in ['unknown', 'alloc_storage', 'alloc_tensor',
                                   'check_tensor_info', 'match_shape', 'null_value']]

print(f"  Compute operations: {len(compute_ops)}")

# Show first 10
print(f"\nFirst 10 compute operations:")
for i, op in enumerate(compute_ops[:10]):
    print(f"  {i}: {op.output_var} = {op.op_name}(...)")

# Show last 5
print(f"\nLast 5 compute operations:")
for i, op in enumerate(compute_ops[-5:], start=len(compute_ops)-5):
    print(f"  {i}: {op.output_var} = {op.op_name}(...)")

# Count alloc buffers
alloc_vars = [name for name in var_shapes.keys() if name.startswith('alloc')]
print(f"\nAllocated buffers (allocXXX): {len(alloc_vars)}")
print(f"  Last alloc: {max(alloc_vars, key=lambda x: int(x[5:]) if x[5:].isdigit() else 0)}")

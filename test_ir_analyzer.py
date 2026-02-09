#!/usr/bin/env python3
"""Test IR Analyzer with PyExprVisitor"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sense.python.parser import parse_model
from sense.python.ir_analyzer import analyze_ir_with_visitor
from tvm import relax

# Parse model
model_path = "sense_onnx/model_main_17.onnx"
print(f"Parsing {model_path}...")
ir_mod, input_info, output_info, weights = parse_model(model_path)

# Apply optimization
print(f"\nApplying optimization...")
ir_mod = relax.get_pipeline("default_build")(ir_mod)

# Analyze IR with PyExprVisitor
print(f"\nAnalyzing IR with PyExprVisitor...")
buffer_lifetimes, operations, constants = analyze_ir_with_visitor(ir_mod)

print(f"\nAnalysis Results:")
print(f"  Buffers tracked: {len(buffer_lifetimes)}")
print(f"  Operations: {len(operations)}")
print(f"  Constants: {len(constants)}")

# Show buffer lifetime samples
print(f"\nBuffer Lifetimes (first 10):")
for i, buf in enumerate(buffer_lifetimes[:10]):
    lifetime_span = buf.last_use - buf.first_use + 1
    print(f"  {i}: {buf.name}")
    print(f"      Shape: {buf.shape}, Size: {buf.size_bytes / 1024:.1f} KB")
    print(f"      Lifetime: ops {buf.first_use}-{buf.last_use} (span={lifetime_span})")

# Calculate potential reuse
print(f"\nBuffer Reuse Opportunities:")
overlaps = 0
for i, buf1 in enumerate(buffer_lifetimes):
    for buf2 in buffer_lifetimes[i+1:]:
        # Can reuse if lifetimes don't overlap
        if buf1.last_use < buf2.first_use:
            if buf1.size_bytes >= buf2.size_bytes * 0.9:  # Similar size
                overlaps += 1
                if overlaps <= 5:  # Show first 5
                    print(f"  {buf1.name} (ops {buf1.first_use}-{buf1.last_use}) → ")
                    print(f"    can be reused by {buf2.name} (ops {buf2.first_use}-{buf2.last_use})")

print(f"\nTotal reuse opportunities: {overlaps}")

# Calculate memory with/without reuse
total_without_reuse = sum(buf.size_bytes for buf in buffer_lifetimes)
print(f"\nMemory Usage:")
print(f"  Without reuse: {total_without_reuse / 1024 / 1024:.2f} MB")
print(f"  Potential savings: {overlaps} buffers can be reused")

# Show operations
print(f"\nOperations (first 10):")
for i, op in enumerate(operations[:10]):
    print(f"  {i}: {op.var_name} = {op.op_name}(...)")
    print(f"      Inputs: {op.input_buffers}")
    print(f"      Weights: {op.weight_indices}")

# Show constants
print(f"\nConstants (first 10):")
for i, const in enumerate(constants[:10]):
    print(f"  Constant[{i}]: shape={const['shape']}, size={const['size']}")

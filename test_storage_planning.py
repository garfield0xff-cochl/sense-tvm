#!/usr/bin/env python3
"""Test Static Storage Planning with Liveness Analysis"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sense.python.parser import parse_model
from sense.python.ir_analyzer import analyze_ir_with_visitor
from sense.python.transforms import StaticStoragePlanner
from tvm import relax

# Parse model
model_path = "sense_onnx/model_main_17.onnx"
print(f"Parsing {model_path}...")
ir_mod, input_info, output_info, weights = parse_model(model_path)

# Apply optimization
print(f"Applying optimization...")
ir_mod = relax.get_pipeline("default_build")(ir_mod)

# Analyze IR
print(f"\nAnalyzing IR with PyExprVisitor...")
buffer_lifetimes, operations, constants = analyze_ir_with_visitor(ir_mod)

print(f"\nExtracted:")
print(f"  Buffers: {len(buffer_lifetimes)}")
print(f"  Operations: {len(operations)}")
print(f"  Constants: {len(constants)}")

# Create storage planner
print(f"\nPlanning static storage...")
planner = StaticStoragePlanner()

# Add all buffers with lifetimes
for buf in buffer_lifetimes:
    planner.add_buffer(buf.name, buf.shape, buf.dtype)
    planner.set_buffer_lifetime(buf.name, buf.first_use, buf.last_use)

# Plan storage
plan = planner.plan_storage()

print(f"\nStorage Plan:")
print(f"  Total size: {plan.total_size / 1024 / 1024:.2f} MB")
print(f"  Peak usage: {plan.peak_usage / 1024 / 1024:.2f} MB")
print(f"  Buffers: {len(plan.buffers)}")
print(f"  Reused: {plan.reuse_count}")

# Calculate memory reduction
total_without_reuse = sum(buf.size_bytes for buf in buffer_lifetimes)
reduction = (1 - plan.total_size / total_without_reuse) * 100

print(f"\nMemory Reduction:")
print(f"  Without reuse: {total_without_reuse / 1024 / 1024:.2f} MB")
print(f"  With reuse: {plan.total_size / 1024 / 1024:.2f} MB")
print(f"  Reduction: {reduction:.1f}%")

# Show reuse examples
print(f"\nBuffer Reuse Examples (first 10):")
count = 0
for i, buf1 in enumerate(plan.buffers):
    for buf2 in plan.buffers[i+1:]:
        if buf1.offset == buf2.offset and buf1.name != buf2.name:
            print(f"  Offset {buf1.offset / 1024:.1f} KB:")
            print(f"    {buf1.name} (ops {buf1.first_use}-{buf1.last_use}) {buf1.size_bytes / 1024:.1f} KB")
            print(f"    {buf2.name} (ops {buf2.first_use}-{buf2.last_use}) {buf2.size_bytes / 1024:.1f} KB")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break

# Generate C declarations
print(f"\nGenerating C code...")
c_code = planner.generate_c_declarations(plan)
print(f"Generated {len(c_code)} bytes of C declarations")
print(f"\nSample C code (first 30 lines):")
for i, line in enumerate(c_code.split('\n')[:30]):
    print(f"  {line}")

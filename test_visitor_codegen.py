#!/usr/bin/env python3
"""Test visitor-based codegen to find the issue"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sense.python.parser import parse_model
from sense.python.ir_analyzer import analyze_ir_with_visitor
from sense.python.transforms import StaticStoragePlanner
from sense.python.codegen_visitor import generate_with_visitor
from tvm import relax

# Parse and optimize
model_path = "sense_onnx/model_main_17.onnx"
print(f"Parsing {model_path}...")
ir_mod, _, _, weights = parse_model(model_path)

print(f"Optimizing...")
ir_mod = relax.get_pipeline("default_build")(ir_mod)

# Analyze IR
print(f"\nAnalyzing IR...")
buffer_lifetimes, operations, constants = analyze_ir_with_visitor(ir_mod)
print(f"  Buffers: {len(buffer_lifetimes)}")
print(f"  Operations: {len(operations)}")
print(f"  Constants: {len(constants)}")

# Plan storage
print(f"\nPlanning storage...")
planner = StaticStoragePlanner()
for buf in buffer_lifetimes:
    planner.add_buffer(buf.name, buf.shape, buf.dtype)
    planner.set_buffer_lifetime(buf.name, buf.first_use, buf.last_use)

plan = planner.plan_storage()
buffer_offsets = {buf.name: buf.offset // 4 for buf in plan.buffers}
print(f"  Buffer offsets: {len(buffer_offsets)}")

# Weight offsets (dummy for now)
weight_offsets = {i: i * 1000 for i in range(220)}

# Test visitor codegen
print(f"\nTesting visitor codegen...")
try:
    c_code = generate_with_visitor(ir_mod, buffer_offsets, weight_offsets)

    print(f"Generated C code:")
    print(f"  Total lines: {len(c_code.split(chr(10)))}")
    print(f"  Total chars: {len(c_code)}")

    # Show first 50 lines
    print(f"\nFirst 50 lines of generated code:")
    for i, line in enumerate(c_code.split('\n')[:50]):
        print(f"{i+1:3}: {line}")

    # Check for operations
    op_count = c_code.count('__tvm_ffi_')
    print(f"\nOperation calls found: {op_count}")

    # Check for specific patterns
    print(f"\nPattern analysis:")
    print(f"  'args[' count: {c_code.count('args[')}")
    print(f"  'make_arg' count: {c_code.count('make_arg')}")
    print(f"  'init_tensor' count: {c_code.count('init_tensor')}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

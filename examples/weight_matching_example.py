#!/usr/bin/env python3
"""
Weight Matching with PyExprVisitor

Demonstrates how to extract Constants from Relax IR and match them with weights.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tvm.relax.expr_functor import PyExprVisitor, visitor
from tvm import relax

from sense.python.parser import parse_model


@visitor
class ConstantExtractor(PyExprVisitor):
    """Extract all Constants with their indices."""

    def __init__(self):
        super().__init__()
        self.constants = []  # List of (index, numpy_data)
        self.constant_index = 0

    def visit_constant_(self, const):
        """Extract constant data."""
        try:
            data = const.data.numpy()
            self.constants.append({
                'index': self.constant_index,
                'shape': data.shape,
                'data': data,
                'size': data.size
            })
            self.constant_index += 1
        except Exception as e:
            # Some constants might not be convertible
            self.constant_index += 1

    def visit_call_(self, call):
        """Visit call arguments."""
        for arg in call.args:
            self.visit_expr(arg)

    def visit_tuple_(self, tup):
        """Visit tuple fields."""
        for field in tup.fields:
            self.visit_expr(field)

    def visit_binding_block_(self, block):
        """Visit bindings in order."""
        for binding in block.bindings:
            self.visit_binding(binding)


def match_constant_to_weight(const_data, weights):
    """Match a constant to a weight by shape and values."""
    const_shape = const_data.shape
    const_size = const_data.size

    # Find weights with matching shape
    candidates = []
    for name, weight in weights.items():
        if weight.shape == const_shape:
            candidates.append(name)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Compare values for exact match
    const_flat = const_data.flatten()[:10]
    for name in candidates:
        weight_flat = weights[name].flatten()[:10]
        if len(weight_flat) == len(const_flat):
            if np.allclose(const_flat, weight_flat, atol=1e-6):
                return name

    return candidates[0]  # Return first candidate as fallback


def main():
    print("=" * 70)
    print("Weight Matching with PyExprVisitor")
    print("=" * 70)

    # Parse model
    model_path = "../sense_onnx/model_main_17.onnx"
    print(f"\nParsing {model_path}...")
    ir_mod, input_info, output_info, weights = parse_model(model_path)

    print(f"  Weights from ONNX: {len(weights)}")

    # Apply optimization
    print(f"\nApplying optimization...")
    ir_mod = relax.get_pipeline("default_build")(ir_mod)

    # Extract constants using PyExprVisitor
    print(f"\nExtracting constants with PyExprVisitor...")
    main_func = ir_mod["main"]
    extractor = ConstantExtractor()
    extractor.visit_expr(main_func)

    print(f"  Found {len(extractor.constants)} constants")

    # Match constants to weights
    print(f"\nMatching constants to weights...")
    matched = 0
    matched_mapping = {}

    for const_info in extractor.constants:
        idx = const_info['index']
        data = const_info['data']
        shape = const_info['shape']

        weight_name = match_constant_to_weight(data, weights)
        if weight_name:
            matched_mapping[idx] = weight_name
            matched += 1

    print(f"  Matched: {matched}/{len(extractor.constants)}")

    # Show sample mappings
    print(f"\nSample mappings (first 10):")
    for idx in sorted(matched_mapping.keys())[:10]:
        name = matched_mapping[idx]
        const_shape = next(c['shape'] for c in extractor.constants if c['index'] == idx)
        print(f"  Constant[{idx}] -> {name} (shape={const_shape})")

    # Find MatMul weights
    print(f"\nMatMul weight mappings:")
    for idx, name in matched_mapping.items():
        if 'MatMul' in name:
            const_shape = next(c['shape'] for c in extractor.constants if c['index'] == idx)
            print(f"  Constant[{idx}] -> {name} (shape={const_shape})")

    # Find bias weights
    print(f"\nBias weight mappings (first 5):")
    bias_count = 0
    for idx, name in sorted(matched_mapping.items()):
        if 'bias' in name.lower():
            const_shape = next(c['shape'] for c in extractor.constants if c['index'] == idx)
            print(f"  Constant[{idx}] -> {name} (shape={const_shape})")
            bias_count += 1
            if bias_count >= 5:
                break

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  - Total constants in IR: {len(extractor.constants)}")
    print(f"  - Matched to weights: {matched}")
    print(f"  - This mapping can be used in codegen to automatically")
    print(f"    map Constant[idx] to g_w[idx]")


if __name__ == "__main__":
    main()

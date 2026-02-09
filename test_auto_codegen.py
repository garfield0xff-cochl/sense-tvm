#!/usr/bin/env python3
"""Test automated codegen with PyExprVisitor weight mapping"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sense.python.sense import Sense

# Configure with auto codegen enabled
config = {
    "common": {
        "output_dir": "./bin",
        "target": "c",
        "opt_level": 3,
    },
    "standalone": {
        "use_auto_codegen": True,  # ⭐ Enable automated weight mapping
    },
    "export": {
        "save_ir": True,
        "save_metadata": True,
        "generate_test_harness": True,
    }
}

print("=" * 70)
print("Testing AUTOMATED Code Generation (PyExprVisitor)")
print("=" * 70)

sense = Sense(config)
sense.compile(
    model_path="sense_onnx/model_main_17.onnx",
    target="c",
    name="sense_model"
)

print("\n" + "=" * 70)
print("Now build and test:")
print("  cd bin")
print("  make clean && make")
print("  ./sense_model_standalone --runs 10")
print("=" * 70)

#!/usr/bin/env python3
"""
Test Sense with TVM MCU Strategy enabled

Demonstrates the new architecture with MCU optimizations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sense.python.sense import Sense

# Configure with MCU strategy enabled
config = {
    "common": {
        "output_dir": "./bin",
        "target": "c",
        "opt_level": 3,
    },
    "mcu_strategy": {
        "enable": True,              # ⭐ Enable TVM MCU optimizations
        "static_storage": True,      # Static buffer planning
        "aggressive_inline": False,  # (Future) Function inlining
        "minimal_ffi": False,        # (Future) Reduce FFI calls
        "target_memory_mb": 50,      # Target 50MB instead of 400MB
    },
    "export": {
        "save_ir": True,
        "save_metadata": True,
        "generate_test_harness": True,
    }
}

print("=" * 80)
print("Testing Sense with TVM MCU Strategy")
print("=" * 80)
print("\nMCU Strategy Settings:")
print(f"  Static Storage: {config['mcu_strategy']['static_storage']}")
print(f"  Target Memory: {config['mcu_strategy']['target_memory_mb']} MB")
print(f"  Aggressive Inline: {config['mcu_strategy']['aggressive_inline']}")
print("=" * 80)

sense = Sense(config)
sense.compile(
    model_path="sense_onnx/model_main_17.onnx",
    target="c",
    name="sense_model"
)

print("\n" + "=" * 80)
print("Architecture Status:")
print("=" * 80)
print("✅ Completed:")
print("  - ONNX parsing module (parser.py)")
print("  - Transform framework (transforms/)")
print("  - MCU strategy pipeline integration")
print()
print("📝 In Progress:")
print("  - Static storage planner (placeholder)")
print("  - Liveness analysis (placeholder)")
print()
print("🔧 Future:")
print("  - Partial Graph AOT implementation")
print("  - Aggressive function inlining")
print("  - Minimal FFI optimization")
print()
print("See SENSE_ARCHITECTURE.md for roadmap")
print("=" * 80)

#!/usr/bin/env python3
"""
Sense Model Compilation Main Script

This script compiles ONNX models to standalone C code using the Sense compiler.
"""

import sys
import argparse
from pathlib import Path

# Add sense to path
sys.path.insert(0, str(Path(__file__).parent))

from sense.python.sense import Sense


def main():
    parser = argparse.ArgumentParser(
        description="Sense Standalone C Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile model_main_17.onnx to C
  python main.py --model sense_onnx/model_main_17.onnx

  # Compile with custom output directory
  python main.py --model sense_onnx/model_main_17.onnx --output ./custom_output

  # Compile with specific target
  python main.py --model sense_onnx/model_main_17.onnx --target "c -keys=cpu -march=armv8-a"
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="sense_onnx/model_main_17.onnx",
        help="Path to ONNX model file (default: sense_onnx/model_main_17.onnx)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./bin",
        help="Output directory for generated C code (default: ./bin)"
    )

    parser.add_argument(
        "--target", "-t",
        type=str,
        default="c",
        help="Target device specification (default: c)"
    )

    parser.add_argument(
        "--name", "-n",
        type=str,
        default="sense_model",
        help="Model name for generated files (default: sense_model)"
    )

    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Optimization level (default: 3)"
    )

    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir

    print("=" * 80)
    print("Sense Standalone C Compiler")
    print("=" * 80)
    print(f"Model:  {model_path}")
    print(f"Output: {output_dir}")
    print(f"Target: {args.target}")
    print(f"Name:   {args.name}")
    print("=" * 80)
    print()

    # Configure Sense compiler
    config = {
        "common": {
            "output_dir": str(output_dir),
            "target": args.target,
            "opt_level": args.opt_level,
        },
        "parser": {
            "input_name": None,
            "input_shape": None,
            "input_dtype": "float32",
        },
        "optimizer": {
            "apply_default_pipeline": True,
            "fuse_ops": True,
            "fold_constant": True,
        },
        "standalone": {
            "generate_entry": True,
            "embed_weights": True,
            "pool_size_mb": 300,
        },
        "export": {
            "save_ir": True,
            "save_metadata": True,
            "generate_test_harness": True,
            "extract_c_to_bin": True,  # New option to extract C to bin
        }
    }

    try:
        # Initialize Sense compiler
        sense = Sense(config)

        # Compile the model
        sense.compile(
            model_path=str(model_path),
            target=args.target,
            name=args.name
        )

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Generated files are in: {output_dir}")
        print()
        print("Generated structure:")
        print(f"  {output_dir}/")
        print(f"    ├── lib0.c              # TVM generated C code")
        print(f"    ├── devc.c              # Device code (if VM used)")
        print(f"    ├── {args.name}.tar     # Export library archive")
        print(f"    ├── generated/")
        print(f"    │   └── {args.name}_ir.txt")
        print(f"    ├── weights/")
        print(f"    │   ├── *.bin")
        print(f"    │   └── weights_manifest.json")
        print(f"    ├── src/")
        print(f"    │   └── test_{args.name}.c")
        print(f"    └── {args.name}_metadata.json")
        print("=" * 80)

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR!")
        print("=" * 80)
        print(f"Compilation failed: {e}")

        import traceback
        print("\nTraceback:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())

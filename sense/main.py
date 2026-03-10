import argparse
import json
from pathlib import Path

from config import SenseConfig
from sense import Sense
from validate import run_validation


def parse_args():
    """Parse command line arguments for Sense compiler"""
    parser = argparse.ArgumentParser(
        description="Sense Compiler - Multi-backend AI model compilation tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file (includes model_path)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate outputs and performance against ONNX runtime"
    )

    return parser.parse_args()


def load_config(json_path: str) -> SenseConfig:
    """Load configuration from JSON file

    Parameters:
        json_path: Path to JSON configuration file

    Returns:
        SenseConfig object

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is invalid
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")

    with open(json_file, 'r') as f:
        config_dict = json.load(f)

    sense_config = SenseConfig.from_dict(config_dict)

    return sense_config


def main():
    """Main entry point for Sense compiler"""
    args = parse_args()

    # Load configuration from JSON file only
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        return 1

    # Validate model path
    if not config.model_path:
        print("Error: Model path not specified. Specify 'model_path' in config file.")
        return 1

    model_path = Path(config.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Validate configuration
    if not config.validate():
        print("Error: Invalid configuration")
        return 1

    # Print configuration summary
    print("=" * 60)
    print("Sense Compiler Configuration")
    print("=" * 60)
    print(config)
    print("=" * 60)

    # Create Sense compiler instance with SenseConfig object
    sense = Sense(config=config)

    # Execute compilation pipeline
    sense.execute()

    output_root = Path(config.export.output_dir) / config.hardware.device
    if output_root.exists():
        print("=" * 60)
        print("Generated Artifacts")
        print("=" * 60)
        size_mb = lambda p: p.stat().st_size / (1024.0 * 1024.0)
        size_kb = lambda p: p.stat().st_size / 1024.0

        weights_path = output_root / "lib" / "weights.bin"
        manifest_path = output_root / "metadata" / "weights.json"
        model_c_path = output_root / "lib" / f"{config.export.model_name}.c"
        lib0_path = output_root / "lib" / "lib0.c"
        tir_path = output_root / "metadata" / "tir.txt"
        makefile_path = output_root / "Makefile"

        if lib0_path.exists():
            print(f"  TVM: {lib0_path} ({size_kb(lib0_path):.2f} KB)")

        if model_c_path.exists():
            print(f"  SENSE_CODEGEN: {model_c_path} ({size_kb(model_c_path):.2f} KB)")
        if weights_path.exists():
            print(f"  SENSE_CODEGEN: {weights_path} ({size_mb(weights_path):.2f} MB)")
        if makefile_path.exists():
            print(f"  SENSE_CODEGEN: {makefile_path} ({size_kb(makefile_path):.2f} KB)")

        if manifest_path.exists():
            print(f"  SENSE_METADATA: {manifest_path} ({size_kb(manifest_path):.2f} KB)")
        if tir_path.exists():
            print(f"  SENSE_METADATA: {tir_path} ({size_kb(tir_path):.2f} KB)")

        for path in sorted(output_root.rglob("*")):
            if path.is_dir():
                continue
            if path in {
                weights_path,
                manifest_path,
                tir_path,
                model_c_path,
                lib0_path,
                makefile_path,
            }:
                continue
            rel_path = path.relative_to(output_root)
            print(f"  OTHER: {rel_path} ({size_kb(path):.2f} KB)")
        print("=" * 60)

    print("\n[INFO] Compilation completed successfully!")

    if args.validate:
        if not run_validation(sense, model_path):
            return 1

    return 0


if __name__ == "__main__":
    exit(main())

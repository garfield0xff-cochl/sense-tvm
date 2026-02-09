# SPDX-License-Identifier: Apache-2.0
"""Sense: TVM Standalone C Codegen for Embedded Targets"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import tvm
from tvm import relax, tir

from .parser import parse_model, parse_onnx, parse_weights, get_model_info


class SenseConfig:
    """Configuration manager for Sense compilation flow."""

    def __init__(self, config_dict: Dict = None):
        """Initialize configuration from dictionary or defaults."""
        self.config = config_dict or {}

        # Default configuration
        self.defaults = {
            "common": {
                "output_dir": "./build/sense_output",
                "log_level": "INFO",
                "target": "c",
                "opt_level": 3,
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
                "use_auto_codegen": False,  # Use automated weight mapping
            },
            "export": {
                "save_ir": True,
                "save_metadata": True,
                "generate_test_harness": True,
            }
        }

        # Merge with defaults
        for key, value in self.defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                self.config[key] = {**value, **self.config.get(key, {})}

    def get(self, section: str) -> Dict:
        """Get configuration section."""
        return self.config.get(section, {})


class Sense:
    """The class which organizes all standalone compilation API functions.

    Sense provides a compilation flow for generating standalone C code from
    ONNX models, targeting embedded/MCU deployments without VM dependencies.

    Attributes
    ----------
    ir_mod : tvm.relax.IRModule
        The Relax IR between each compilation phase.
    config : SenseConfig
        Configuration manager for the compilation flow.
    input_info : dict
        Information about model inputs (shape, dtype).
    output_info : dict
        Information about model outputs (shape, dtype).
    """

    def __init__(self, config: Dict = None):
        """Initialize Sense compiler.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If None, uses defaults.
        """
        self.ir_mod = None
        self.config = SenseConfig(config)
        self.input_info = {}
        self.output_info = {}
        self.weights = {}
        self.compiled_module = None

        # Create output directory
        output_dir = self.config.get("common")["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Sense Compiler initialized")
        print(f"  TVM Version: {tvm.__version__}")
        print(f"  Output Directory: {output_dir}")

    def _timer(self, func):
        """Decorator for timing compilation phases."""
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t_start
            print(f"  Elapsed time: {elapsed:.2f}s")
            return result
        return wrapper

    def parse(self, model_path: str):
        """Parse ONNX model to Relax IR.

        Parameters
        ----------
        model_path : str
            Path to ONNX model file.
        """
        t_start = time.perf_counter()
        print(f"\n[1/6] Parsing ONNX model...")

        parser_cfg = self.config.get("parser")

        # Use parser module for ONNX parsing
        self.ir_mod, self.input_info, self.output_info = parse_onnx(
            model_path,
            input_name=parser_cfg["input_name"],
            input_shape=parser_cfg["input_shape"],
            input_dtype=parser_cfg["input_dtype"]
        )

        # Extract weights using parser module
        self.weights = parse_weights(model_path)

        print(f"  Model: {model_path}")
        print(f"  Inputs: {self.input_info}")
        print(f"  Outputs: {self.output_info}")
        print(f"  Weights: {len(self.weights)} tensors")
        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")

        return self

    def optimize(self):
        """Apply Relax-level optimizations.

        This phase applies the default Relax optimization pipeline including:
        - FuseOps: Operator fusion
        - FoldConstant: Constant folding
        - DecomposeOps: Operator decomposition
        """
        t_start = time.perf_counter()
        print(f"\n[2/6] Optimizing...")

        opt_cfg = self.config.get("optimizer")

        if opt_cfg["apply_default_pipeline"]:
            # Apply default Relax build pipeline
            self.ir_mod = relax.get_pipeline("default_build")(self.ir_mod)
            print(f"  Applied default_build pipeline")
        else:
            # Apply custom passes
            passes = []
            if opt_cfg["fold_constant"]:
                passes.append(relax.transform.FoldConstant())
            if opt_cfg["fuse_ops"]:
                passes.append(relax.transform.FuseOps())

            with tvm.transform.PassContext(opt_level=self.config.get("common")["opt_level"]):
                self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)
            print(f"  Applied {len(passes)} optimization passes")

        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")
        return self

    def transform_to_standalone(self):
        """Transform Relax TIR to standalone-compatible format.

        This phase transforms private TIR functions from T.Buffer parameters
        to T.handle parameters with match_buffer in the body, required for
        standalone compilation without VM dependencies.

        Transformation:
        - T.Buffer params → T.handle params
        - buffer_map cleared (size = 0)
        - T.match_buffer added to function body
        """
        t_start = time.perf_counter()
        print(f"\n[3/6] Transforming to standalone TIR...")

        # TODO: Implement T.Buffer -> T.handle transformation
        # This is the key transformation for standalone builds
        # See model/docs/SOLUTION_SIMPLE_TIR_ENTRY_PATTERN.md

        print(f"  Note: Standalone TIR transformation not yet implemented")
        print(f"  Current IR uses T.Buffer parameters (Relax default)")
        print(f"  Standalone requires T.handle parameters for private functions")

        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")
        return self

    def build_standalone(self, target: str = "c"):
        """Build standalone C module without VM dependencies.

        Parameters
        ----------
        target : str
            Target device specification (default: "c").
        """
        t_start = time.perf_counter()
        print(f"\n[4/6] Building standalone module...")

        common_cfg = self.config.get("common")
        target_str = target or common_cfg["target"]

        # Create TVM target
        if target_str == "c":
            tvm_target = tvm.target.Target("c -keys=cpu")
        else:
            tvm_target = tvm.target.Target(target_str)

        print(f"  Target: {tvm_target}")

        # Build with aggressive optimizations for embedded targets
        with tvm.transform.PassContext(
            opt_level=common_cfg["opt_level"],
            config={
                "tir.disable_vectorize": True,  # Disable for embedded
            }
        ):
            self.compiled_module = relax.build(self.ir_mod, target=tvm_target)

        print(f"  Compilation successful")
        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")
        return self

    def generate_standalone_c(self, name: str = "sense_model", use_auto: bool = None):
        """Generate standalone C code using code generator.

        This method generates a complete standalone C implementation by parsing
        the Relax IR and generating model_forward() with all operations.

        Parameters
        ----------
        name : str
            Model name for generated C file.
        use_auto : bool, optional
            Use automated weight mapping with PyExprVisitor.
            If None, uses config setting.
        """
        t_start = time.perf_counter()
        print(f"\n[5.5/6] Generating standalone C code...")

        output_dir = Path(self.config.get("common")["output_dir"])
        ir_path = output_dir / "generated" / f"{name}_ir.txt"
        weights_dir = output_dir / "weights"
        output_path = output_dir / f"{name}_standalone.c"

        # Get input/output shapes
        input_name = list(self.input_info.keys())[0]
        input_shape = self.input_info[input_name]["shape"]
        output_name = list(self.output_info.keys())[0]
        output_shape = self.output_info[output_name]["shape"]

        # Check if auto codegen should be used
        standalone_cfg = self.config.get("standalone")
        if use_auto is None:
            use_auto = standalone_cfg.get("use_auto_codegen", False)

        if use_auto:
            # Use fully automated weight mapping
            from .codegen_auto import generate_standalone_c_auto
            success = generate_standalone_c_auto(
                ir_mod=self.ir_mod,
                weights=self.weights,
                weights_dir=weights_dir,
                output_path=output_path,
                model_name=name,
                input_shape=input_shape,
                output_shape=output_shape
            )
        else:
            # Use manual weight mapping
            from .codegen import generate_standalone_c
            success = generate_standalone_c(
                ir_path=ir_path,
                weights_dir=weights_dir,
                output_path=output_path,
                model_name=name,
                input_shape=input_shape,
                output_shape=output_shape
            )

        if success:
            print(f"  Standalone C: {output_path}")
        else:
            print(f"  Warning: Standalone C generation failed")

        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")
        return self

    def export(self, name: str = "sense_model"):
        """Export compiled model and metadata.

        Parameters
        ----------
        name : str
            Base name for generated files.
        """
        import tarfile
        import tempfile
        import shutil

        t_start = time.perf_counter()
        print(f"\n[5/6] Exporting...")

        export_cfg = self.config.get("export")
        output_dir = Path(self.config.get("common")["output_dir"])

        # Create subdirectories
        generated_dir = output_dir / "generated"
        weights_dir = output_dir / "weights"
        generated_dir.mkdir(exist_ok=True)
        weights_dir.mkdir(exist_ok=True)

        # Save IR if requested
        if export_cfg["save_ir"]:
            ir_path = generated_dir / f"{name}_ir.txt"
            with open(ir_path, 'w') as f:
                f.write(self.ir_mod.script(show_meta=True))
            print(f"  IR saved: {ir_path}")

        # Save weights
        weights_manifest = []
        total_weight_size = 0
        for weight_name, weight_data in self.weights.items():
            weight_file = weights_dir / f"{weight_name}.bin"
            weight_data.tofile(weight_file)

            weights_manifest.append({
                "name": weight_name,
                "shape": list(weight_data.shape),
                "dtype": str(weight_data.dtype),
                "size_bytes": weight_data.nbytes,
                "file": weight_file.name
            })
            total_weight_size += weight_data.nbytes

        manifest_path = weights_dir / "weights_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                "model": name,
                "total_weights": len(weights_manifest),
                "total_size_bytes": total_weight_size,
                "weights": weights_manifest
            }, f, indent=2)

        print(f"  Weights: {len(weights_manifest)} tensors ({total_weight_size / 1024 / 1024:.2f} MB)")
        print(f"  Weights manifest: {manifest_path}")

        # Export library to .tar and extract C sources
        c_extracted = False
        if self.compiled_module and hasattr(self.compiled_module, 'mod'):
            lib = self.compiled_module.mod

            # Try export_library to generate .tar
            tar_path = output_dir / f"{name}.tar"
            try:
                print(f"  Exporting library to: {tar_path}")
                lib.export_library(str(tar_path))
                print(f"  Export successful: {tar_path}")

                # Extract .tar and find C source files
                with tempfile.TemporaryDirectory() as tmpdir:
                    print(f"  Extracting .tar contents...")
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(tmpdir)

                    # Find and copy C source files to output_dir
                    tmpdir_path = Path(tmpdir)
                    c_files = list(tmpdir_path.rglob("*.c"))

                    if c_files:
                        print(f"  Found {len(c_files)} C source file(s):")
                        for c_file in c_files:
                            dest = output_dir / c_file.name
                            shutil.copy(c_file, dest)
                            file_size = dest.stat().st_size
                            print(f"    - {c_file.name} ({file_size / 1024:.2f} KB) -> {dest}")
                        c_extracted = True
                    else:
                        print(f"  Warning: No C source files found in .tar")

            except Exception as e:
                print(f"  Note: export_library failed ({e})")

            # Try direct get_source as fallback
            if not c_extracted:
                try:
                    if hasattr(lib, 'get_source'):
                        source = lib.get_source()
                        if source and len(source) > 100:
                            c_path = output_dir / "lib0.c"
                            with open(c_path, 'w') as f:
                                f.write(source)
                            print(f"  C source (direct): {c_path} ({len(source)} bytes)")
                            c_extracted = True
                except Exception as e:
                    print(f"  Note: Direct C source extraction failed ({e})")

        if not c_extracted:
            print(f"  Warning: Could not extract C source code")
            print(f"          This may be expected for Relax with C target")
            print(f"          The .tar file contains the compiled module")

        # Save metadata
        if export_cfg["save_metadata"]:
            metadata = {
                "model_name": name,
                "target": self.config.get("common")["target"],
                "inputs": self.input_info,
                "outputs": self.output_info,
                "weights": {
                    "count": len(self.weights),
                    "total_size_bytes": total_weight_size
                },
                "tvm_version": tvm.__version__,
                "config": self.config.config,
                "c_sources_extracted": c_extracted
            }

            metadata_path = output_dir / f"{name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"  Metadata: {metadata_path}")

        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")
        return self

    def generate_test_harness(self, name: str = "sense_model"):
        """Generate C test harness for standalone testing.

        Parameters
        ----------
        name : str
            Model name for generated test file.
        """
        t_start = time.perf_counter()
        print(f"\n[6/6] Generating test harness...")

        output_dir = Path(self.config.get("common")["output_dir"])
        src_dir = output_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Get input/output info
        input_name = list(self.input_info.keys())[0]
        input_shape = self.input_info[input_name]["shape"]
        input_dtype = self.input_info[input_name]["dtype"]
        input_size = int(np.prod(input_shape))

        output_name = list(self.output_info.keys())[0]
        output_shape = self.output_info[output_name]["shape"]
        output_dtype = self.output_info[output_name]["dtype"]
        output_size = int(np.prod(output_shape))

        # Generate test harness C code
        test_harness = f"""/**
 * Sense Model Test Harness
 * Model: {name}
 * Generated by Sense Compiler
 *
 * Input: {input_name} {input_shape} ({input_dtype})
 * Output: {output_name} {output_shape} ({output_dtype})
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Model dimensions */
#define INPUT_SIZE {input_size}
#define OUTPUT_SIZE {output_size}

/* Buffers */
static float input_data[INPUT_SIZE];
static float output_data[OUTPUT_SIZE];

/* Generate random input */
void generate_random_input(void) {{
    srand((unsigned int)time(NULL));
    for (int i = 0; i < INPUT_SIZE; i++) {{
        input_data[i] = (float)(rand() / (double)RAND_MAX);
    }}
}}

/* Load input from binary file */
int load_input_from_file(const char* filename) {{
    FILE* f = fopen(filename, "rb");
    if (!f) {{
        fprintf(stderr, "Error: Cannot open input file: %s\\n", filename);
        return -1;
    }}

    size_t read = fread(input_data, sizeof(float), INPUT_SIZE, f);
    fclose(f);

    if (read != INPUT_SIZE) {{
        fprintf(stderr, "Error: Expected %d elements, read %zu\\n", INPUT_SIZE, read);
        return -1;
    }}

    return 0;
}}

/* Save output to binary file */
int save_output_to_file(const char* filename) {{
    FILE* f = fopen(filename, "wb");
    if (!f) {{
        fprintf(stderr, "Error: Cannot create output file: %s\\n", filename);
        return -1;
    }}

    fwrite(output_data, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);

    return 0;
}}

int main(int argc, char* argv[]) {{
    printf("=== Sense Model Test Harness ===\\n");
    printf("Model: {name}\\n");
    printf("Input: {input_name} {input_shape}\\n");
    printf("Output: {output_name} {output_shape}\\n");
    printf("================================\\n\\n");

    const char* input_file = NULL;
    const char* output_file = NULL;
    int use_random = 1;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {{
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {{
            input_file = argv[++i];
            use_random = 0;
        }} else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {{
            output_file = argv[++i];
        }} else if (strcmp(argv[i], "-h") == 0) {{
            printf("Usage: %s [-i input.bin] [-o output.bin]\\n", argv[0]);
            printf("  -i  Input binary file (raw float data)\\n");
            printf("  -o  Output binary file (save results)\\n");
            return 0;
        }}
    }}

    /* Load or generate input */
    if (use_random) {{
        printf("[Input] Generating random input data...\\n");
        generate_random_input();
    }} else {{
        printf("[Input] Loading from file: %s\\n", input_file);
        if (load_input_from_file(input_file) != 0) {{
            return 1;
        }}
    }}

    printf("[Note] This is a test harness template.\\n");
    printf("       Actual inference requires TVM runtime integration.\\n");

    /* TODO: Add model inference call here */
    /* model_forward(input_data, output_data); */

    /* Save output if requested */
    if (output_file) {{
        printf("[Output] Saving to file: %s\\n", output_file);
        if (save_output_to_file(output_file) != 0) {{
            return 1;
        }}
    }}

    printf("\\n=== Test Complete ===\\n");
    return 0;
}}
"""

        test_path = src_dir / f"test_{name}.c"
        with open(test_path, 'w') as f:
            f.write(test_harness)

        print(f"  Test harness: {test_path}")
        print(f"  Elapsed time: {time.perf_counter() - t_start:.2f}s")
        return self

    def compile(self, model_path: str, target: str = "c", name: str = "sense_model"):
        """All-in-one compilation API.

        Parameters
        ----------
        model_path : str
            Path to ONNX model file.
        target : str
            Target device specification.
        name : str
            Model name for generated files.

        Returns
        -------
        self : Sense
            The Sense instance for chaining.
        """
        print("=" * 70)
        print("Sense Standalone C Compiler")
        print("=" * 70)

        (self
            .parse(model_path)
            .optimize()
            .transform_to_standalone()
            .build_standalone(target=target)
            .export(name=name)
            .generate_standalone_c(name=name)
            .generate_test_harness(name=name))

        print("\n" + "=" * 70)
        print("Compilation Complete!")
        print(f"Output directory: {self.config.get('common')['output_dir']}")
        print("=" * 70)

        return self

    def save(self, path: str, show_meta: bool = True):
        """Save current IR module to file.

        Parameters
        ----------
        path : str
            Output file path.
        show_meta : bool
            Include metadata in output.
        """
        with open(path, 'w') as f:
            f.write(self.ir_mod.script(show_meta=show_meta))

    def load(self, path: str):
        """Load IR module from file.

        Parameters
        ----------
        path : str
            Input file path.
        """
        from tvm import script
        with open(path) as f:
            self.ir_mod = script.from_source(f.read())


def main():
    """Example usage of Sense compiler."""
    import argparse

    parser = argparse.ArgumentParser(description="Sense Standalone C Compiler")
    parser.add_argument("--model", "-m", required=True, help="Path to ONNX model")
    parser.add_argument("--output", "-o", default="./build/sense_output", help="Output directory")
    parser.add_argument("--target", "-t", default="c", help="Target device")
    parser.add_argument("--name", "-n", default="sense_model", help="Model name")

    args = parser.parse_args()

    config = {
        "common": {
            "output_dir": args.output,
            "target": args.target,
        }
    }

    sense = Sense(config)
    sense.compile(args.model, target=args.target, name=args.name)


if __name__ == "__main__":
    main()

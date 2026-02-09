# SPDX-License-Identifier: Apache-2.0
"""
Sense Profiling and Comparison Tool

Compares inference performance and correctness between:
1. Sense Standalone C (compiled with gcc)
2. ONNX Runtime (Python)

Based on tvm_native_c_codegen/scripts/compare_runtimes.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class BenchmarkResult:
    """Benchmark result data class"""
    runtime_name: str
    runs: int
    avg_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput: float  # inferences/sec
    output_sample: List[float]
    implementation_scope: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison between two runtimes"""
    sense_result: Optional[BenchmarkResult]
    onnx_result: Optional[BenchmarkResult]
    speedup: Optional[float]  # ONNX time / Sense time
    output_similarity: Optional[float]  # Cosine similarity
    max_abs_diff: Optional[float]
    mean_abs_diff: Optional[float]
    comparison_valid: bool
    notes: List[str]


@dataclass
class CorrectnessResult:
    """Correctness validation result"""
    passed: bool
    tolerance: float
    max_abs_diff: float
    mean_abs_diff: float
    relative_error: float
    num_mismatches: int
    total_elements: int
    mismatch_percentage: float
    notes: List[str]


class SenseNativeBenchmark:
    """Run Sense Native C benchmark with JSON output"""

    def __init__(self, build_dir: Path, num_runs: int = 100):
        self.build_dir = build_dir
        self.num_runs = num_runs
        self.executable = build_dir / "sense_model_standalone"

    def is_available(self) -> bool:
        """Check if the benchmark executable exists"""
        return self.executable.exists()

    def build(self) -> bool:
        """Build the benchmark if needed"""
        if self.executable.exists():
            return True

        try:
            result = subprocess.run(
                ["make", "all"],
                cwd=self.build_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Build failed: {e}")
            return False

    def run(self, seed: int = 42, input_file: Optional[Path] = None,
            output_file: Optional[Path] = None) -> BenchmarkResult:
        """Run the Sense Native C benchmark with JSON output"""
        if not self.is_available():
            if not self.build():
                return BenchmarkResult(
                    runtime_name="Sense Native C",
                    runs=0,
                    avg_ms=0,
                    std_ms=0,
                    min_ms=0,
                    max_ms=0,
                    throughput=0,
                    output_sample=[],
                    implementation_scope="N/A",
                    success=False,
                    error_message="Failed to build benchmark executable"
                )

        try:
            # Build command
            cmd = [str(self.executable), "--runs", str(self.num_runs), "--json"]
            if input_file:
                cmd.extend(["--input", str(input_file)])
            if output_file:
                cmd.extend(["--output", str(output_file)])

            # Run benchmark and capture output
            result = subprocess.run(
                cmd,
                cwd=self.build_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return BenchmarkResult(
                    runtime_name="Sense Native C",
                    runs=0,
                    avg_ms=0,
                    std_ms=0,
                    min_ms=0,
                    max_ms=0,
                    throughput=0,
                    output_sample=[],
                    implementation_scope="N/A",
                    success=False,
                    error_message=f"Benchmark failed: {result.stderr}"
                )

            # Parse JSON output
            return self._parse_json_output(result.stdout)

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                runtime_name="Sense Native C",
                runs=0,
                avg_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                throughput=0,
                output_sample=[],
                implementation_scope="N/A",
                success=False,
                error_message="Benchmark timed out"
            )
        except Exception as e:
            return BenchmarkResult(
                runtime_name="Sense Native C",
                runs=0,
                avg_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                throughput=0,
                output_sample=[],
                implementation_scope="N/A",
                success=False,
                error_message=str(e)
            )

    def _parse_json_output(self, output: str) -> BenchmarkResult:
        """Parse JSON benchmark output"""
        try:
            # Extract JSON from mixed output
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                json_start = output.find('{')
                json_end = output.rfind('}')
                if json_start == -1 or json_end == -1:
                    raise json.JSONDecodeError("No JSON object found", output, 0)
                json_str = output[json_start:json_end + 1]
                data = json.loads(json_str)

            return BenchmarkResult(
                runtime_name=data.get("runtime", "Sense Native C"),
                runs=data.get("runs", 0),
                avg_ms=data.get("avg_ms", 0),
                std_ms=data.get("std_ms", 0),
                min_ms=data.get("min_ms", 0),
                max_ms=data.get("max_ms", 0),
                throughput=data.get("throughput", 0),
                output_sample=data.get("output_sample", [])[:10],
                implementation_scope=data.get("implementation_scope", "Full model (388 ops)"),
                success=data.get("success", False),
                error_message=data.get("error_message")
            )
        except json.JSONDecodeError as e:
            return BenchmarkResult(
                runtime_name="Sense Native C",
                runs=0,
                avg_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                throughput=0,
                output_sample=[],
                implementation_scope="N/A",
                success=False,
                error_message=f"Failed to parse JSON: {e}"
            )


class ONNXRuntimeBenchmark:
    """Run ONNX Runtime benchmark"""

    def __init__(self, model_path: Path, num_runs: int = 100):
        self.model_path = model_path
        self.num_runs = num_runs
        self._session = None
        self._input_name = None
        self._output_name = None
        self.last_input = None
        self.last_output = None

    def is_available(self) -> bool:
        """Check if ONNX Runtime is available"""
        try:
            import onnxruntime
            return self.model_path.exists()
        except ImportError:
            return False

    def _init_session(self):
        """Initialize ONNX Runtime session"""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX Runtime: {e}")

    def run(self, seed: int = 42, input_file: Optional[Path] = None,
            output_file: Optional[Path] = None) -> BenchmarkResult:
        """Run ONNX Runtime benchmark"""
        if not self.is_available():
            return BenchmarkResult(
                runtime_name="ONNX Runtime",
                runs=0,
                avg_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                throughput=0,
                output_sample=[],
                implementation_scope="N/A",
                success=False,
                error_message="ONNX Runtime not available or model not found"
            )

        try:
            self._init_session()

            # Generate or load input
            if input_file and input_file.exists():
                input_data = np.fromfile(str(input_file), dtype=np.float32)
                input_data = input_data.reshape(1, 128, 192, 1)
            else:
                np.random.seed(seed)
                input_data = np.random.rand(1, 128, 192, 1).astype(np.float32)
                if input_file:
                    input_data.tofile(str(input_file))

            self.last_input = input_data

            # Warmup
            for _ in range(5):
                _ = self._session.run([self._output_name], {self._input_name: input_data})

            # Benchmark
            times = []
            output = None
            for _ in range(self.num_runs):
                start = time.perf_counter()
                output = self._session.run([self._output_name], {self._input_name: input_data})
                times.append((time.perf_counter() - start) * 1000)

            times = np.array(times)
            avg_ms = float(np.mean(times))
            std_ms = float(np.std(times))
            min_ms = float(np.min(times))
            max_ms = float(np.max(times))

            output_array = output[0][0] if output else np.array([])
            output_sample = output_array[:10].tolist() if len(output_array) > 0 else []
            self.last_output = output_array

            # Save output for comparison
            if output_file and output:
                output_array.astype(np.float32).tofile(str(output_file))

            return BenchmarkResult(
                runtime_name="ONNX Runtime (CPU)",
                runs=self.num_runs,
                avg_ms=avg_ms,
                std_ms=std_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                throughput=1000.0 / avg_ms if avg_ms > 0 else 0,
                output_sample=output_sample,
                implementation_scope="Full model (all operations)",
                success=True
            )

        except ImportError:
            return BenchmarkResult(
                runtime_name="ONNX Runtime",
                runs=0,
                avg_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                throughput=0,
                output_sample=[],
                implementation_scope="N/A",
                success=False,
                error_message="onnxruntime package not installed"
            )
        except Exception as e:
            return BenchmarkResult(
                runtime_name="ONNX Runtime",
                runs=0,
                avg_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                throughput=0,
                output_sample=[],
                implementation_scope="N/A",
                success=False,
                error_message=str(e)
            )


def compute_similarity(a: List[float], b: List[float]) -> Dict[str, float]:
    """Compute similarity metrics between two output vectors"""
    if not a or not b:
        return {"cosine": 0.0, "max_abs_diff": float('inf'), "mean_abs_diff": float('inf')}

    a = np.array(a)
    b = np.array(b)

    # Truncate to same length
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    # Cosine similarity
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
    else:
        cosine = 0.0

    # Absolute differences
    abs_diff = np.abs(a - b)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    return {
        "cosine": cosine,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff
    }


def validate_correctness(sense_output: np.ndarray, onnx_output: np.ndarray,
                         atol: float = 1e-5, rtol: float = 1e-4) -> CorrectnessResult:
    """Validate correctness between Sense and ONNX Runtime outputs."""
    notes = []

    # Handle shape differences
    if sense_output.shape != onnx_output.shape:
        min_size = min(sense_output.size, onnx_output.size)
        sense_flat = sense_output.flatten()[:min_size]
        onnx_flat = onnx_output.flatten()[:min_size]
        notes.append(f"Shape mismatch: Sense {sense_output.shape} vs ONNX {onnx_output.shape}")
        notes.append(f"Comparing first {min_size} elements")
    else:
        sense_flat = sense_output.flatten()
        onnx_flat = onnx_output.flatten()

    # Compute differences
    abs_diff = np.abs(sense_flat - onnx_flat)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Relative error
    onnx_abs = np.abs(onnx_flat)
    valid_mask = onnx_abs > 1e-10
    if np.any(valid_mask):
        rel_error = np.mean(abs_diff[valid_mask] / onnx_abs[valid_mask])
    else:
        rel_error = 0.0

    # Count mismatches
    mismatches = np.sum(~np.isclose(sense_flat, onnx_flat, atol=atol, rtol=rtol))
    total_elements = len(sense_flat)
    mismatch_pct = 100.0 * mismatches / total_elements if total_elements > 0 else 0.0

    # Determine pass/fail
    passed = np.allclose(sense_flat, onnx_flat, atol=atol, rtol=rtol)

    if passed:
        notes.append("✅ Correctness PASSED: outputs match within tolerance")
    else:
        notes.append(f"❌ Correctness FAILED: {mismatches}/{total_elements} elements differ")
        if max_abs_diff > 1.0:
            notes.append(f"  Large deviation detected: max diff = {max_abs_diff:.6f}")

    return CorrectnessResult(
        passed=passed,
        tolerance=atol,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        relative_error=float(rel_error),
        num_mismatches=int(mismatches),
        total_elements=total_elements,
        mismatch_percentage=mismatch_pct,
        notes=notes
    )


def load_binary_output(file_path: Path, expected_size: int = 863) -> Optional[np.ndarray]:
    """Load binary output file"""
    if not file_path.exists():
        return None
    try:
        data = np.fromfile(str(file_path), dtype=np.float32)
        return data
    except Exception:
        return None


def compare_runtimes(sense_result: BenchmarkResult, onnx_result: BenchmarkResult) -> ComparisonResult:
    """Compare two benchmark results"""
    notes = []

    # Check if both succeeded
    if not sense_result.success:
        notes.append(f"Sense benchmark failed: {sense_result.error_message}")
    if not onnx_result.success:
        notes.append(f"ONNX benchmark failed: {onnx_result.error_message}")

    comparison_valid = sense_result.success and onnx_result.success

    # Calculate speedup
    speedup = None
    if comparison_valid and sense_result.avg_ms > 0:
        speedup = onnx_result.avg_ms / sense_result.avg_ms
        if speedup > 1:
            notes.append(f"Sense Native C is {speedup:.2f}x faster than ONNX Runtime")
        else:
            notes.append(f"ONNX Runtime is {1/speedup:.2f}x faster than Sense Native C")

    # Compare outputs
    similarity = None
    max_abs_diff = None
    mean_abs_diff = None

    if sense_result.output_sample and onnx_result.output_sample:
        sim = compute_similarity(sense_result.output_sample, onnx_result.output_sample)
        similarity = sim["cosine"]
        max_abs_diff = sim["max_abs_diff"]
        mean_abs_diff = sim["mean_abs_diff"]

        if similarity > 0.99:
            notes.append("Outputs are highly similar (cosine > 0.99)")
        elif similarity > 0.9:
            notes.append("Outputs are moderately similar (cosine > 0.9)")
        else:
            notes.append("Outputs differ significantly")

    return ComparisonResult(
        sense_result=sense_result,
        onnx_result=onnx_result,
        speedup=speedup,
        output_similarity=similarity,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        comparison_valid=comparison_valid,
        notes=notes
    )


def print_result(result: BenchmarkResult, verbose: bool = False):
    """Print benchmark result"""
    print(f"\n{'=' * 60}")
    print(f"  {result.runtime_name}")
    print(f"{'=' * 60}")

    if not result.success:
        print(f"  Status: FAILED")
        print(f"  Error: {result.error_message}")
        return

    print(f"  Status: SUCCESS")
    print(f"  Implementation: {result.implementation_scope}")
    print(f"  Runs: {result.runs}")
    print(f"  Average: {result.avg_ms:.4f} ms")
    print(f"  Std Dev: {result.std_ms:.4f} ms")
    print(f"  Min/Max: {result.min_ms:.4f} / {result.max_ms:.4f} ms")
    print(f"  Throughput: {result.throughput:.2f} inferences/sec")

    if verbose and result.output_sample:
        print(f"  Output sample: {result.output_sample[:5]}...")


def print_comparison(comparison: ComparisonResult):
    """Print comparison summary"""
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    if comparison.speedup is not None:
        print(f"  Speedup (ONNX/Sense): {comparison.speedup:.2f}x")

    if comparison.output_similarity is not None:
        print(f"  Output Cosine Similarity: {comparison.output_similarity:.4f}")
        print(f"  Max Absolute Difference: {comparison.max_abs_diff:.6f}")
        print(f"  Mean Absolute Difference: {comparison.mean_abs_diff:.6f}")

    print(f"\n  Notes:")
    for note in comparison.notes:
        print(f"    - {note}")


def print_correctness(result: CorrectnessResult):
    """Print correctness validation results"""
    print(f"\n{'=' * 60}")
    print(f"  CORRECTNESS VALIDATION")
    print(f"{'=' * 60}")
    print(f"  Status: {'PASSED ✅' if result.passed else 'FAILED ❌'}")
    print(f"  Tolerance (atol): {result.tolerance}")
    print(f"  Max Absolute Diff: {result.max_abs_diff:.6e}")
    print(f"  Mean Absolute Diff: {result.mean_abs_diff:.6e}")
    print(f"  Relative Error: {result.relative_error:.4%}")
    print(f"  Mismatches: {result.num_mismatches}/{result.total_elements} ({result.mismatch_percentage:.2f}%)")
    print(f"\n  Notes:")
    for note in result.notes:
        print(f"    {note}")


def save_results(comparison: ComparisonResult, output_path: Path,
                 correctness: Optional[CorrectnessResult] = None):
    """Save results to JSON file"""
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sense_result": to_dict(comparison.sense_result) if comparison.sense_result else None,
        "onnx_result": to_dict(comparison.onnx_result) if comparison.onnx_result else None,
        "speedup": comparison.speedup,
        "output_similarity": comparison.output_similarity,
        "max_abs_diff": comparison.max_abs_diff,
        "mean_abs_diff": comparison.mean_abs_diff,
        "comparison_valid": comparison.comparison_valid,
        "notes": comparison.notes,
        "correctness": to_dict(correctness) if correctness else None
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sense Native C and ONNX Runtime performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python -m sense.python.profile --runs 100

  # With correctness validation
  python -m sense.python.profile --runs 100 --validate

  # Save results to file
  python -m sense.python.profile --runs 100 --output results.json

  # Specify custom paths
  python -m sense.python.profile --model sense_onnx/model_main_17.onnx --bin-dir bin/
        """
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=100,
        help="Number of benchmark runs (default: 100)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path (default: bin/comparison_results.json)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Perform correctness validation with shared input"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5,
        help="Absolute tolerance for correctness check (default: 1e-5)"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-4,
        help="Relative tolerance for correctness check (default: 1e-4)"
    )
    parser.add_argument(
        "--model", type=str, default="sense_onnx/model_main_17.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--bin-dir", type=str, default="bin",
        help="Path to bin directory with standalone executable"
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = script_dir / model_path

    bin_dir = Path(args.bin_dir)
    if not bin_dir.is_absolute():
        bin_dir = script_dir / bin_dir

    print("=" * 60)
    print("  Sense Native C vs ONNX Runtime Comparison")
    print("=" * 60)
    print(f"  Model: {model_path.name}")
    print(f"  Input: (1, 128, 192, 1) float32")
    print(f"  Output: (1, 863) float32")
    print(f"  Runs: {args.runs}")
    print(f"  Seed: {args.seed}")
    if args.validate:
        print(f"  Correctness validation: ENABLED")
        print(f"  Tolerance: atol={args.atol}, rtol={args.rtol}")

    # Setup shared input/output files for validation
    shared_input = bin_dir / "shared_input.bin" if args.validate else None
    sense_output_file = bin_dir / "sense_output.bin" if args.validate else None
    onnx_output_file = bin_dir / "onnx_output.bin" if args.validate else None

    # Run ONNX Runtime benchmark first (to generate shared input)
    print("\n[1/2] Running ONNX Runtime benchmark...")
    onnx_bench = ONNXRuntimeBenchmark(model_path, args.runs)
    onnx_result = onnx_bench.run(args.seed, input_file=shared_input, output_file=onnx_output_file)
    print_result(onnx_result, args.verbose)

    # Run Sense Native C benchmark
    print("\n[2/2] Running Sense Native C benchmark...")
    sense_bench = SenseNativeBenchmark(bin_dir, args.runs)
    sense_result = sense_bench.run(args.seed, input_file=shared_input, output_file=sense_output_file)
    print_result(sense_result, args.verbose)

    # Compare results
    comparison = compare_runtimes(sense_result, onnx_result)
    print_comparison(comparison)

    # Correctness validation
    correctness_result = None
    if args.validate and sense_result.success and onnx_result.success:
        print("\n[3/3] Validating correctness...")

        # Load output files
        sense_output = load_binary_output(sense_output_file) if sense_output_file else None
        onnx_output = onnx_bench.last_output

        if sense_output is not None and onnx_output is not None:
            correctness_result = validate_correctness(
                sense_output, onnx_output,
                atol=args.atol, rtol=args.rtol
            )
            print_correctness(correctness_result)
        else:
            print("\n  ⚠️ Could not perform correctness validation:")
            if sense_output is None:
                print("    - Sense output not available")
            if onnx_output is None:
                print("    - ONNX output not available")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = bin_dir / "comparison_results.json"

    save_results(comparison, output_path, correctness_result)

    print("\n" + "=" * 60)
    print("  Comparison completed!")
    print("=" * 60)

    # Return code
    if correctness_result:
        return 0 if (comparison.comparison_valid and correctness_result.passed) else 1
    return 0 if comparison.comparison_valid else 1


if __name__ == "__main__":
    sys.exit(main())

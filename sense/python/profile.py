import numpy as np
import onnxruntime as ort
import subprocess
import json
import time
from pathlib import Path
from typing import Tuple, Dict
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


class ModelProfiler:
    def __init__(self, onnx_path: str, c_executable: str):
        self.onnx_path = Path(onnx_path)
        self.c_executable = Path(c_executable)

        # Load ONNX model
        self.ort_session = ort.InferenceSession(str(self.onnx_path))

        # Get input/output info
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.output_name = self.ort_session.get_outputs()[0].name
        self.output_shape = self.ort_session.get_outputs()[0].shape

        print(f"ONNX Model Info:")
        print(f"  Input: {self.input_name}, shape: {self.input_shape}")
        print(f"  Output: {self.output_name}, shape: {self.output_shape}")

        # Load TVM model
        print(f"\nLoading TVM model...")
        onnx_model = onnx.load(str(self.onnx_path))
        self.tvm_mod = from_onnx(onnx_model)
        print(f"  TVM model loaded")

        # Apply same optimization passes as main_sense2.py
        print(f"Applying optimization passes...")
        self.tvm_mod = relax.get_pipeline("default_build")(self.tvm_mod)
        print(f"  Applied default_build pipeline")

        # Save IR for comparison
        self.tvm_ir = self.tvm_mod.script(show_meta=True)

        # Build TVM VM executable
        print(f"Building TVM VM executable...")
        target = tvm.target.Target("llvm")
        with tvm.transform.PassContext(opt_level=3):
            self.tvm_exec = relax.build(self.tvm_mod, target)
        self.tvm_vm = relax.VirtualMachine(self.tvm_exec, tvm.cpu())
        print(f"  TVM VM ready")

    def generate_test_input(self, seed: int = 42) -> np.ndarray:
        """Generate random test input"""
        np.random.seed(seed)
        return np.random.randn(*self.input_shape).astype(np.float32)

    def run_onnx(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run ONNX model inference"""
        start = time.time()
        output = self.ort_session.run([self.output_name], {self.input_name: input_data})[0]
        elapsed = time.time() - start
        return output, elapsed

    def run_tvm_vm(self, input_data: np.ndarray, debug_intermediate: bool = False) -> Tuple[np.ndarray, float]:
        """Run TVM VM inference"""
        start = time.time()
        # TVM VM can accept numpy arrays directly
        tvm_output = self.tvm_vm["main"](input_data)
        elapsed = time.time() - start

        # Convert TVM tensor to numpy
        if hasattr(tvm_output, 'numpy'):
            output = tvm_output.numpy()
        elif hasattr(tvm_output, 'asnumpy'):
            output = tvm_output.asnumpy()
        else:
            output = np.array(tvm_output)

        if debug_intermediate:
            print("\n  TVM VM intermediate values (first 10 ops):")
            # We can't easily get intermediate values from VM, so we'll rerun with instrumentation
            # For now, just show final output stats
            print(f"    Final output: shape={output.shape}, min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}")

        return output, elapsed

    def run_c_code(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run C code inference"""
        if not self.c_executable.exists():
            raise FileNotFoundError(f"C executable not found: {self.c_executable}")

        # Save input to binary file
        input_file = self.c_executable.parent / "test_input.bin"
        output_file = self.c_executable.parent / "test_output.bin"

        input_data.astype(np.float32).tofile(input_file)

        # Run C executable from its directory (so it can find lib/weights.bin)
        start = time.time()
        result = subprocess.run(
            [str(self.c_executable.absolute()), str(input_file.absolute()), str(output_file.absolute())],
            cwd=str(self.c_executable.parent),
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"C code execution failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError("C code execution failed")

        # Read output
        if not output_file.exists():
            raise FileNotFoundError(f"Output file not found: {output_file}")

        output = np.fromfile(output_file, dtype=np.float32).reshape(self.output_shape)

        return output, elapsed

    def compare_outputs(self, onnx_output: np.ndarray, tvm_output: np.ndarray, c_output: np.ndarray) -> Dict:
        """Compare ONNX, TVM VM, and C outputs"""
        def calc_diff(ref, test):
            abs_diff = np.abs(ref - test)
            rel_diff = abs_diff / (np.abs(ref) + 1e-8)
            return {
                'max_abs_diff': float(np.max(abs_diff)),
                'mean_abs_diff': float(np.mean(abs_diff)),
                'max_rel_diff': float(np.max(rel_diff)),
                'mean_rel_diff': float(np.mean(rel_diff)),
                'mse': float(np.mean((ref - test) ** 2)),
                'cosine_similarity': float(
                    np.dot(ref.flatten(), test.flatten()) /
                    (np.linalg.norm(ref) * np.linalg.norm(test) + 1e-8)
                )
            }

        stats = {
            'tvm_vs_onnx': calc_diff(onnx_output, tvm_output),
            'c_vs_onnx': calc_diff(onnx_output, c_output),
            'c_vs_tvm': calc_diff(tvm_output, c_output)
        }

        return stats

    def profile(self, num_tests: int = 5, verbose: bool = True) -> Dict:
        """Run profiling with multiple tests"""
        results = {
            'num_tests': num_tests,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'tests': []
        }

        print(f"\n{'='*60}")
        print(f"Running {num_tests} profiling tests...")
        print(f"{'='*60}\n")

        for i in range(num_tests):
            print(f"Test {i+1}/{num_tests}:")

            # Generate test input
            input_data = self.generate_test_input(seed=42 + i)

            # Run ONNX
            onnx_output, onnx_time = self.run_onnx(input_data)
            print(f"  ONNX: {onnx_time*1000:.2f} ms")

            # Run TVM VM
            tvm_output, tvm_time = self.run_tvm_vm(input_data)
            print(f"  TVM VM: {tvm_time*1000:.2f} ms")

            # Run C code
            try:
                c_output, c_time = self.run_c_code(input_data)
                print(f"  C code: {c_time*1000:.2f} ms")

                # Compare
                stats = self.compare_outputs(onnx_output, tvm_output, c_output)

                if verbose:
                    print(f"  Comparison (TVM vs ONNX):")
                    print(f"    Max abs diff: {stats['tvm_vs_onnx']['max_abs_diff']:.6e}")
                    print(f"    Cosine similarity: {stats['tvm_vs_onnx']['cosine_similarity']:.8f}")
                    print(f"  Comparison (C vs ONNX):")
                    print(f"    Max abs diff: {stats['c_vs_onnx']['max_abs_diff']:.6e}")
                    print(f"    Cosine similarity: {stats['c_vs_onnx']['cosine_similarity']:.8f}")
                    print(f"  Comparison (C vs TVM):")
                    print(f"    Max abs diff: {stats['c_vs_tvm']['max_abs_diff']:.6e}")
                    print(f"    Cosine similarity: {stats['c_vs_tvm']['cosine_similarity']:.8f}")

                results['tests'].append({
                    'test_id': i,
                    'onnx_time_ms': onnx_time * 1000,
                    'tvm_time_ms': tvm_time * 1000,
                    'c_time_ms': c_time * 1000,
                    'comparison': stats
                })

            except Exception as e:
                print(f"  C code failed: {e}")
                results['tests'].append({
                    'test_id': i,
                    'onnx_time_ms': onnx_time * 1000,
                    'tvm_time_ms': tvm_time * 1000,
                    'error': str(e)
                })

            print()

        # Summary
        print(f"{'='*60}")
        print("Summary:")
        print(f"{'='*60}")

        successful_tests = [t for t in results['tests'] if 'error' not in t]

        if successful_tests:
            avg_onnx = np.mean([t['onnx_time_ms'] for t in successful_tests])
            avg_tvm = np.mean([t['tvm_time_ms'] for t in successful_tests])
            avg_c = np.mean([t['c_time_ms'] for t in successful_tests])

            avg_tvm_onnx_diff = np.mean([t['comparison']['tvm_vs_onnx']['max_abs_diff'] for t in successful_tests])
            avg_tvm_onnx_cos = np.mean([t['comparison']['tvm_vs_onnx']['cosine_similarity'] for t in successful_tests])
            avg_c_onnx_diff = np.mean([t['comparison']['c_vs_onnx']['max_abs_diff'] for t in successful_tests])
            avg_c_onnx_cos = np.mean([t['comparison']['c_vs_onnx']['cosine_similarity'] for t in successful_tests])
            avg_c_tvm_diff = np.mean([t['comparison']['c_vs_tvm']['max_abs_diff'] for t in successful_tests])
            avg_c_tvm_cos = np.mean([t['comparison']['c_vs_tvm']['cosine_similarity'] for t in successful_tests])

            print(f"Success rate: {len(successful_tests)}/{num_tests}")
            print(f"\nTiming:")
            print(f"  Avg ONNX time: {avg_onnx:.2f} ms")
            print(f"  Avg TVM VM time: {avg_tvm:.2f} ms")
            print(f"  Avg C code time: {avg_c:.2f} ms")
            print(f"\nAccuracy (TVM vs ONNX):")
            print(f"  Max abs diff: {avg_tvm_onnx_diff:.6e}")
            print(f"  Cosine similarity: {avg_tvm_onnx_cos:.8f}")
            print(f"\nAccuracy (C vs ONNX):")
            print(f"  Max abs diff: {avg_c_onnx_diff:.6e}")
            print(f"  Cosine similarity: {avg_c_onnx_cos:.8f}")
            print(f"\nAccuracy (C vs TVM):")
            print(f"  Max abs diff: {avg_c_tvm_diff:.6e}")
            print(f"  Cosine similarity: {avg_c_tvm_cos:.8f}")

            results['summary'] = {
                'success_rate': len(successful_tests) / num_tests,
                'avg_onnx_time_ms': avg_onnx,
                'avg_tvm_time_ms': avg_tvm,
                'avg_c_time_ms': avg_c,
                'tvm_vs_onnx': {
                    'max_abs_diff': avg_tvm_onnx_diff,
                    'cosine_similarity': avg_tvm_onnx_cos
                },
                'c_vs_onnx': {
                    'max_abs_diff': avg_c_onnx_diff,
                    'cosine_similarity': avg_c_onnx_cos
                },
                'c_vs_tvm': {
                    'max_abs_diff': avg_c_tvm_diff,
                    'cosine_similarity': avg_c_tvm_cos
                }
            }
        else:
            print("All tests failed!")
            results['summary'] = {'success_rate': 0}

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save profiling results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Profile and compare ONNX vs C code')
    parser.add_argument('--onnx', type=str, default='sense_onnx/model_main_17.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--exec', type=str, default='bin_sense2/sense_model',
                        help='Path to C executable')
    parser.add_argument('--num-tests', type=int, default=5,
                        help='Number of test runs')
    parser.add_argument('--output', type=str, default='profile_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable verbose output')

    args = parser.parse_args()

    profiler = ModelProfiler(args.onnx, args.exec)
    results = profiler.profile(num_tests=args.num_tests, verbose=not args.no_verbose)
    profiler.save_results(results, args.output)


if __name__ == '__main__':
    main()

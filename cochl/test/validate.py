#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np


def _bench(fn, runs: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    end = time.perf_counter()
    return (end - start) / runs


def _build_executable(lib_dir: Path) -> bool:
    exe = lib_dir / "main_entry"
    lib0 = lib_dir / "lib0.c"
    main_c = lib_dir / "main_entry.c"
    if not lib0.exists() or not main_c.exists():
        print("Error: lib0.c or main_entry.c not found.")
        return False

    repo_root = Path(__file__).resolve().parents[2]
    include_dir = repo_root
    cmd = [
        "c++",
        "-O3",
        "-DNDEBUG",
        "-ffast-math",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-fno-strict-aliasing",
        "-std=c++17",
        "-I",
        str(include_dir),
        str(main_c),
        str(lib0),
        "-o",
        str(exe),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        return False
    return True


def _build_sense_debug_executable() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    lib_dir = repo_root / "sense" / "bin" / "rpi2" / "lib"
    main_c = lib_dir / "sense_model_main.c"
    lib0 = lib_dir / "lib0.c"
    if not main_c.exists() or not lib0.exists():
        return None
    exe = Path("/tmp/sense_model_main_debug")
    cmd = [
        "c++",
        "-O3",
        "-DDEBUG_INTERMEDIATE",
        "-std=c++17",
        "-I",
        str(repo_root),
        str(main_c),
        str(lib0),
        "-o",
        str(exe),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Sense debug build failed:")
        print(result.stderr)
        return None
    return exe


def _parse_sense_debug(path: Path) -> Dict[int, List[float]]:
    data: Dict[int, List[float]] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if not line.startswith("[Op"):
            continue
        m = re.match(r"\[Op(\d+): .*?\] values=\[(.*)\]$", line)
        if not m:
            continue
        op_idx = int(m.group(1)) - 1  # sense debug is 1-based
        values_str = m.group(2)
        vals: List[float] = []
        for token in values_str.split(","):
            t = token.strip()
            if not t:
                continue
            if t.endswith("..."):
                t = t[:-3].strip()
                if t:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        pass
                break
            try:
                vals.append(float(t))
            except ValueError:
                break
        if vals:
            data[op_idx] = vals
    return data


def _parse_op_tensors(path: Path) -> Dict[int, Tuple[List[float], str]]:
    data: Dict[int, Tuple[List[float], str]] = {}
    if not path.exists():
        return data
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("op ") and "output" in line:
            parts = line.split()
            try:
                op_idx = int(parts[1])
            except (IndexError, ValueError):
                i += 1
                continue
            label = " ".join(parts[2:-1]) if len(parts) > 3 else ""
            # find data line
            vals = None
            for j in range(i + 1, min(i + 8, len(lines))):
                if "data[0:" in lines[j]:
                    vals = [float(x) for x in lines[j].split("=", 1)[1].split()]
                    break
            if vals:
                data[op_idx] = (vals, label)
        i += 1
    return data


def _parse_op_tensors_io(path: Path) -> Dict[int, Dict[str, Tuple[List[float], str]]]:
    data: Dict[int, Dict[str, Tuple[List[float], str]]] = {}
    if not path.exists():
        return data
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("op ") and (" input" in line or " output" in line):
            parts = line.split()
            try:
                op_idx = int(parts[1])
            except (IndexError, ValueError):
                i += 1
                continue
            kind = "input" if " input" in line else "output"
            label = " ".join(parts[2:-1]) if len(parts) > 3 else ""
            vals = None
            for j in range(i + 1, min(i + 8, len(lines))):
                if "data[0:" in lines[j]:
                    vals = [float(x) for x in lines[j].split("=", 1)[1].split()]
                    break
            if vals:
                data.setdefault(op_idx, {})[kind] = (vals, label)
        i += 1
    return data


def _write_op_match(
    ncnn_vals: Dict[int, Tuple[List[float], str]],
    sense_vals: Dict[int, List[float]],
    out_path: Path,
) -> Tuple[int | None, List[float] | None, List[float] | None, str | None, float | None, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_ops = sorted(set(ncnn_vals.keys()) | set(sense_vals.keys()))
    lines: List[str] = []
    first_fail: Tuple[int, List[float], List[float], str, float, int] | None = None
    for op in all_ops:
        a = ncnn_vals.get(op)
        b = sense_vals.get(op)
        if a is None:
            lines.append(f"op {op} MISSING_NCNN")
            continue
        if b is None:
            lines.append(f"op {op} MISSING_SENSE")
            continue
        vals, label = a
        mode = "fallback" if "fallback" in label else "ncnn"
        n = min(len(vals), len(b))
        if n == 0:
            lines.append(f"op {op} EMPTY {mode}")
            continue
        max_abs = max(abs(vals[i] - b[i]) for i in range(n))
        status = "PASS" if max_abs <= 1e-4 else "FAIL"
        lines.append(f"op {op} {status} max_abs={max_abs:.6g} count={n} {mode}")
        if status == "FAIL" and first_fail is None:
            first_fail = (op, vals[:n], b[:n], label, max_abs, n)
    out_path.write_text("\n".join(lines) + "\n")
    if first_fail is None:
        return (None, None, None, None, None, 0)
    return first_fail


def _write_op_match_aligned(
    ncnn_vals: Dict[int, Tuple[List[float], str]],
    sense_vals: Dict[int, List[float]],
    out_path: Path,
    window: int = 5,
    thresh: float = 1e-4,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncnn_ops = sorted(ncnn_vals.keys())
    sense_ops = sorted(sense_vals.keys())
    sense_pos = 0
    lines: List[str] = []
    for op in ncnn_ops:
        vals, label = ncnn_vals[op]
        mode = "fallback" if "fallback" in label else "ncnn"
        if sense_pos >= len(sense_ops):
            lines.append(f"op {op} -> sense NONE MISSING_SENSE {mode}")
            continue
        # find best match within window
        best = None
        best_idx = None
        for j in range(sense_pos, min(sense_pos + window, len(sense_ops))):
            sidx = sense_ops[j]
            b = sense_vals.get(sidx)
            if not b:
                continue
            n = min(len(vals), len(b))
            if n == 0:
                continue
            max_abs = max(abs(vals[i] - b[i]) for i in range(n))
            if best is None or max_abs < best:
                best = max_abs
                best_idx = sidx
            if max_abs <= thresh:
                best = max_abs
                best_idx = sidx
                break
        if best_idx is None:
            lines.append(f"op {op} -> sense NONE FAIL {mode}")
            continue
        n = min(len(vals), len(sense_vals[best_idx]))
        status = "PASS" if best is not None and best <= thresh else "FAIL"
        lines.append(f"op {op} -> sense {best_idx} {status} max_abs={best:.6g} count={n} {mode}")
        # advance sense cursor to keep order
        try:
            sense_pos = sense_ops.index(best_idx) + 1
        except ValueError:
            sense_pos += 1
    out_path.write_text("\n".join(lines) + "\n")


def _parse_op_map(path: Path) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        # format: "<idx>: <tvm_op> -> <ncnn_name> params=..."
        if ":" not in line or "->" not in line:
            continue
        try:
            idx_str, rest = line.split(":", 1)
            idx = int(idx_str.strip())
        except Exception:
            continue
        parts = rest.split("->", 1)
        if len(parts) != 2:
            continue
        tvm_op = parts[0].strip()
        ncnn_part = parts[1].strip()
        ncnn_name = ncnn_part.split()[0].strip()
        label = f"{tvm_op}:{ncnn_name}"
        items.append((idx, label))
    return items


def _write_op_match_from_map(
    ncnn_vals: Dict[int, Tuple[List[float], str]],
    sense_vals: Dict[int, List[float]],
    op_map: List[Tuple[int, str]],
    out_path: Path,
) -> None:
    # Align ncnn ops to op_map labels in order, then compare with sense by op_map index.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncnn_ops = sorted(ncnn_vals.keys())
    cursor = 0
    lines: List[str] = []
    for sense_idx, label in op_map:
        # find next ncnn op whose label matches
        matched = None
        while cursor < len(ncnn_ops):
            op_idx = ncnn_ops[cursor]
            cursor += 1
            vals, ncnn_label = ncnn_vals[op_idx]
            if ncnn_label == label:
                matched = (op_idx, vals, ncnn_label)
                break
        if matched is None:
            lines.append(f"op {sense_idx} MISSING_NCNN label={label}")
            continue
        op_idx, vals, ncnn_label = matched
        b = sense_vals.get(sense_idx)
        if b is None:
            lines.append(f"op {sense_idx} MISSING_SENSE label={label}")
            continue
        n = min(len(vals), len(b))
        if n == 0:
            lines.append(f"op {sense_idx} EMPTY label={label}")
            continue
        max_abs = max(abs(vals[i] - b[i]) for i in range(n))
        status = "PASS" if max_abs <= 1e-4 else "FAIL"
        lines.append(f"op {sense_idx} {status} max_abs={max_abs:.6g} count={n} {('fallback' if 'fallback' in ncnn_label else 'ncnn')}")
    out_path.write_text("\n".join(lines) + "\n")


def _write_op_diff(op_idx: int, ncnn_vals: List[float], sense_vals: List[float], label: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(ncnn_vals), len(sense_vals))
    lines = [
        f"op {op_idx} label={label}",
        f"count={n}",
        "idx ncnn sense abs_diff",
    ]
    for i in range(n):
        a = ncnn_vals[i]
        b = sense_vals[i]
        lines.append(f"{i} {a:.9g} {b:.9g} {abs(a - b):.9g}")
    out_path.write_text("\n".join(lines) + "\n")


def _cleanup_metadata(meta_dir: Path) -> None:
    keep = {
        "op_match.txt",
        "op_map.txt",
        "op_match_aligned.txt",
        "op_tensor.txt",
        "op_delay.txt",
        "op_unmatched.txt",
        "weights.json",
    }
    if not meta_dir.exists():
        return
    for path in meta_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink(missing_ok=True)


def _parse_op_tensor_labels(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        if not line.startswith("op "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            op_idx = int(parts[1])
        except ValueError:
            continue
        # line format: "op <idx> <label...> input|output"
        if parts[-1] in {"input", "output"}:
            label = " ".join(parts[2:-1])
        else:
            label = " ".join(parts[2:])
        if label:
            labels[op_idx] = label
    return labels


def _write_op_mode_only(ncnn_vals: Dict[int, Tuple[List[float], str]], out_path: Path, labels: Dict[int, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for op in sorted(ncnn_vals.keys()):
        _, label = ncnn_vals[op]
        mode = "fallback" if "fallback" in label else "ncnn"
        op_label = labels.get(op, label)
        lines.append(f"op {op} {mode} {op_label}")
    out_path.write_text("\n".join(lines) + "\n")


def _compare_outputs(onnx_output: np.ndarray, c_output: np.ndarray) -> Dict[str, Any]:
    onnx_flat = onnx_output.flatten()
    c_flat = c_output.flatten()

    abs_diff = np.abs(onnx_flat - c_flat)
    rel_diff = np.abs((onnx_flat - c_flat) / (np.abs(onnx_flat) + 1e-10))
    cosine_sim = float(np.dot(onnx_flat, c_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(c_flat)))
    is_close = bool(np.allclose(onnx_flat, c_flat, rtol=1e-4, atol=1e-5))

    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "cosine_similarity": cosine_sim,
        "is_close": is_close,
        "shape_match": onnx_output.shape == c_output.shape,
    }


def run_validation(onnx_path: Path, lib_dir: Path) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime is required for validation.")
        return False

    lib_dir = lib_dir.resolve()
    if not _build_executable(lib_dir):
        return False

    exe = (lib_dir / "main_entry").resolve()
    if not exe.exists():
        print(f"Error: executable not found at {exe}")
        return False

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_shape = [1 if s is None else int(s) for s in input_shape]
    input_dtype = np.float32

    rng = np.random.default_rng(42)
    input_data = rng.standard_normal(input_shape).astype(input_dtype)

    ort_outputs = sess.run(None, {input_name: input_data})
    if not ort_outputs:
        print("Error: ONNX runtime produced no outputs.")
        return False
    ort_output = ort_outputs[0]

    c_latency_ms = None
    c_output = None

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_out:
            try:
                input_data.tofile(tmp_in.name)
                tmp_in.flush()
                out_elems = int(np.prod(ort_output.shape))
                t0 = time.perf_counter()
                result = subprocess.run(
                    [str(exe), tmp_in.name, tmp_out.name, str(out_elems), str(lib_dir / "weights.bin")],
                    cwd=str(lib_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                t1 = time.perf_counter()
                c_latency_ms = (t1 - t0) * 1000.0
                if result.returncode != 0:
                    print(f"Execution failed: {result.stderr}")
                    return False
                c_output = np.fromfile(tmp_out.name, dtype=np.float32).reshape(ort_output.shape)
            finally:
                Path(tmp_out.name).unlink(missing_ok=True)

        # write op_match.txt as mode-only (ncnn/fallback) without accuracy compare
        meta_dir = lib_dir.parent / "metadata"
        op_tensor_path = meta_dir / "op_tensor.txt"
        ncnn_vals = _parse_op_tensors(op_tensor_path)
        labels = _parse_op_tensor_labels(op_tensor_path)
        _write_op_mode_only(ncnn_vals, meta_dir / "op_match.txt", labels)

        try:
            Path(tmp_in.name).unlink(missing_ok=True)
        finally:
            pass

    metrics = _compare_outputs(ort_output, c_output)

    ort_time = _bench(lambda: sess.run(None, {input_name: input_data}))

    speedup = None
    if c_latency_ms and c_latency_ms > 0:
        speedup = ort_time * 1000.0 / c_latency_ms

    status = "PASS" if metrics["is_close"] else "FAIL"
    print(f"Final output match: {status} | max_abs={metrics['max_abs_diff']:.6g} | mean_abs={metrics['mean_abs_diff']:.6g} | cosine={metrics['cosine_similarity']:.6g}")

    return metrics["is_close"]


def main() -> int:
    onnx_path = Path("sense/onnx/model_main_17.onnx")
    lib_dir = Path("cochl/test/lib")
    ok = run_validation(onnx_path, lib_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

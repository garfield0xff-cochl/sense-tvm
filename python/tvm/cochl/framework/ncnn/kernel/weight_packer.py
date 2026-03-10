# SPDX-License-Identifier: Apache-2.0
"""Weight packing utilities for NCNN backends."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tvm.cochl.core.translate import translate_onnx


class NcnnWeightPacker:
    """Pack weights into a single binary with optional JSON metadata."""

    @staticmethod
    def pack(
        weights: Dict[str, np.ndarray],
        weight_order: List[str],
        output_dir: Path,
        save_metadata: bool = True,
        transforms: Dict[int, np.ndarray] | None = None,
    ) -> Tuple[Path, Dict]:
        lib_dir = output_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)

        bin_path = lib_dir / "weights.bin"

        current_offset = 0
        weight_map = {}

        for idx, name in enumerate(weight_order):
            if name not in weights:
                continue
            if transforms and idx in transforms:
                weight_data = transforms[idx]
            else:
                weight_data = weights[name]
            size_bytes = weight_data.nbytes
            aligned_size = ((size_bytes + 63) // 64) * 64

            weight_map[idx] = {
                "name": name,
                "offset": current_offset,
                "size_bytes": size_bytes,
                "aligned_size": aligned_size,
                "shape": list(weight_data.shape),
                "dtype": str(weight_data.dtype),
            }

            current_offset += aligned_size

        total_floats = current_offset // 4
        packed_data = np.zeros(total_floats, dtype=np.float32)

        for idx, name in enumerate(weight_order):
            if name not in weights:
                continue
            if transforms and idx in transforms:
                weight_data = transforms[idx].astype(np.float32).flatten()
            else:
                weight_data = weights[name].astype(np.float32).flatten()
            offset_floats = weight_map[idx]["offset"] // 4
            size_floats = len(weight_data)

            packed_data[offset_floats : offset_floats + size_floats] = weight_data

        packed_data.tofile(bin_path)

        if save_metadata:
            metadata_dir = output_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = metadata_dir / "weights.json"
            metadata = {
                "total_size_bytes": current_offset,
                "total_size_mb": current_offset / (1024 * 1024),
                "num_weights": len(weight_map),
                "weights": weight_map,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Weights: {bin_path} ({current_offset / 1024 / 1024:.2f} MB)")
            print(f"  Metadata: {metadata_path}")
        else:
            print(f"  Weights: {bin_path} ({current_offset / 1024 / 1024:.2f} MB)")

        return bin_path, weight_map

    @staticmethod
    def pack_from_onnx(
        model_path: str,
        output_dir: Path,
        save_metadata: bool = True,
        transforms: Dict[int, np.ndarray] | None = None,
    ) -> Tuple[Path, Dict]:
        _, _, _, weights, weight_order = translate_onnx(Path(model_path))
        return NcnnWeightPacker.pack(weights, weight_order, output_dir, save_metadata, transforms)

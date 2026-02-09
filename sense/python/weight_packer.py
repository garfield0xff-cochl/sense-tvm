# SPDX-License-Identifier: Apache-2.0
"""
Weight Packer

Packs all weights into a single unified binary file with index-based access.
Similar to TVM VM's devc.c approach but for weights only.

Reference: TVM_VM_WEIGHT_ACCESS.md
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json


class WeightPacker:
    """Pack all weights into a single binary file with offset map."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        """Initialize weight packer.

        Parameters
        ----------
        weights : Dict[str, np.ndarray]
            Dictionary of weight name to numpy array.
        """
        self.weights = weights
        self.weight_offsets = {}  # weight_name -> offset in bytes
        self.weight_indices = {}  # weight_name -> index
        self.index_to_name = {}   # index -> weight_name
        self.total_size = 0

    def pack_weights(self, ordered_weight_names: List[str]) -> Tuple[np.ndarray, Dict]:
        """Pack weights into single array following given order.

        Parameters
        ----------
        ordered_weight_names : List[str]
            Ordered list of weight names (matches metadata indices).

        Returns
        -------
        packed_data : np.ndarray
            Single numpy array containing all weights.
        weight_map : Dict
            Metadata about weight packing: {index: {name, offset, size, shape}}.
        """
        # Calculate total size and offsets
        current_offset = 0
        weight_map = {}

        for idx, name in enumerate(ordered_weight_names):
            if name not in self.weights:
                print(f"Warning: Weight '{name}' not found, skipping")
                continue

            weight_data = self.weights[name]
            size_bytes = weight_data.nbytes

            # Align to 64 bytes for performance
            aligned_size = ((size_bytes + 63) // 64) * 64

            self.weight_offsets[name] = current_offset
            self.weight_indices[name] = idx
            self.index_to_name[idx] = name

            weight_map[idx] = {
                'name': name,
                'offset': current_offset,
                'size_bytes': size_bytes,
                'aligned_size': aligned_size,
                'shape': list(weight_data.shape),
                'dtype': str(weight_data.dtype)
            }

            current_offset += aligned_size

        self.total_size = current_offset

        # Allocate packed array
        packed_data = np.zeros(current_offset // 4, dtype=np.float32)

        # Pack weights
        for idx, name in enumerate(ordered_weight_names):
            if name not in self.weights:
                continue

            weight_data = self.weights[name].astype(np.float32).flatten()
            offset_floats = self.weight_offsets[name] // 4
            size_floats = len(weight_data)

            packed_data[offset_floats:offset_floats + size_floats] = weight_data

        return packed_data, weight_map

    def save_packed_weights(self, output_path: Path, packed_data: np.ndarray,
                           weight_map: Dict):
        """Save packed weights to binary file and generate metadata.

        Parameters
        ----------
        output_path : Path
            Output path for packed binary file.
        packed_data : np.ndarray
            Packed weight data.
        weight_map : Dict
            Weight metadata.
        """
        # Save binary file
        packed_data.tofile(output_path)

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'total_size_bytes': self.total_size,
            'total_size_mb': self.total_size / (1024 * 1024),
            'num_weights': len(weight_map),
            'weights': weight_map
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Packed weights: {output_path}")
        print(f"    Total size: {self.total_size / 1024 / 1024:.2f} MB")
        print(f"    Num weights: {len(weight_map)}")
        print(f"    Metadata: {metadata_path}")

    def generate_c_declarations(self, weight_map: Dict, var_name: str = "g_weights") -> str:
        """Generate C code for weight declarations.

        Parameters
        ----------
        weight_map : Dict
            Weight metadata from pack_weights().
        var_name : str
            Variable name for weight array.

        Returns
        -------
        str
            C code declarations.
        """
        code = f'''/*========================================
 * Unified Weights (TVM VM Style)
 * Total: {self.total_size / 1024 / 1024:.2f} MB
 * Weights: {len(weight_map)}
 *========================================*/
#define UNIFIED_WEIGHTS_SIZE {self.total_size}
static float __attribute__((aligned(64))) {var_name}[UNIFIED_WEIGHTS_SIZE / sizeof(float)];

/* Weight offsets (compile-time constants) */
'''
        for idx in sorted(weight_map.keys()):
            info = weight_map[idx]
            offset_floats = info['offset'] // 4
            size_kb = info['size_bytes'] / 1024
            safe_name = info['name'].replace('.bin', '').replace('-', '_').replace('.', '_').upper()[:40]
            code += f'#define WEIGHT_{idx}_OFFSET {offset_floats}  // [{idx}] {size_kb:.1f} KB\n'

        code += '\n/* Weight accessors (inlined to pointers) */\n'
        for idx in sorted(weight_map.keys()):
            code += f'#define GET_WEIGHT_{idx}() (&{var_name}[WEIGHT_{idx}_OFFSET])\n'

        code += f'\n/* Total weights size: {self.total_size / 1024 / 1024:.2f} MB */\n'

        return code


def pack_and_save_weights(weights: Dict[str, np.ndarray],
                          ordered_names: List[str],
                          output_dir: Path,
                          filename: str = "unified_weights.bin") -> Tuple[Path, Dict]:
    """Pack all weights into single file and save.

    Parameters
    ----------
    weights : Dict[str, np.ndarray]
        Dictionary of weights.
    ordered_names : List[str]
        Ordered list of weight names (matches metadata indices).
    output_dir : Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    output_path : Path
        Path to saved binary file.
    weight_map : Dict
        Weight metadata.
    """
    packer = WeightPacker(weights)
    packed_data, weight_map = packer.pack_weights(ordered_names)

    output_path = output_dir / filename
    packer.save_packed_weights(output_path, packed_data, weight_map)

    return output_path, weight_map

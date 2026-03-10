# SPDX-License-Identifier: Apache-2.0
"""MCU-optimized codegen for tvm_c backend."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import sources
from ...relax.standalone_packer import StandalonePacker, pack_tvm_dll, match_relax_const_idx
from ..kernel.weight_packer import WeightPacker

def _codegen_impl(
    ir_mod,
    input_name: str,
    input_shape: Tuple,
    output_shape: Tuple,
    weights: Dict[str, np.ndarray],
    weight_order: List[str],
    output_dir: Path,
    model_name: str = "sense_model",
    save_metadata: bool = True,
    **kwargs,
) -> bool:
    """
    MCU-optimized codegen: Direct function calls, no DLTensor overhead

    Parameters:
        ir_mod: TVM IR module
        input_name: Input tensor name
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        weights: Model weights
        weight_order: Weight names in order
        output_dir: Output directory
        model_name: Model name
        save_metadata: Whether to save metadata
        **kwargs:
            dump_ir_tensor_data (bool): Enable DEBUG_INTERMEDIATE code
    """
    # Extract device-specific parameters from kwargs
    dump_ir_tensor_data = kwargs.get("dump_ir_tensor_data", False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PACKING")
    print("=" * 60)

    ir_text = ir_mod.script(show_meta=True)

    parser = StandalonePacker(ir_text)
    parser.pack()
    if not parser.storages or not parser.operations:
        print("    WARNING: No VM storage/ops found in IR; skipping codegen.")
        return False

    input_size = int(np.prod(input_shape))
    output_size = int(np.prod(output_shape))

    # Convert input name: "input_1:0" -> "input_1_0", "audio_in:0" -> "audio_in_0"
    input_var_name = input_name.replace(':', '_')
    print(f"    Input variable name: {input_var_name}")

    # Extract actual constant indices used in IR.
    ir_constant_indices = set()
    for op in parser.operations:
        for arg in op.input_vars:
            if 'metadata["relax.expr.Constant"]' in arg:
                const_idx = match_relax_const_idx(arg)
                if const_idx is not None:
                    ir_constant_indices.add(const_idx)

    # Build actual weight list used in IR (filter out unused weights from weight_order)
    actual_weights_used = []
    for name in weight_order:
        if name in weights:
            w = weights[name]
            # Only include weights that will be used in operations
            # Skip very small tensors that are likely reshape constants
            if w.size >= 2:
                actual_weights_used.append(name)

    # Verify we have the right number of weights
    if len(actual_weights_used) != len(ir_constant_indices):
        print(
            f"    WARNING: IR uses {len(ir_constant_indices)} constants but found "
            f"{len(actual_weights_used)} weights"
        )
        print(f"    IR constant indices: {sorted(ir_constant_indices)}")
        print(f"    Weight names: {actual_weights_used}")

    print(f"    Actual weights used in IR: {len(actual_weights_used)}")

    # Build constant index -> packed weight slot map.
    # WeightPacker.pack() stores weights with compact 0..N-1 slots, while Relax IR may
    # reference sparse/shifted Constant indices.
    sorted_const_indices = sorted(ir_constant_indices)
    const_idx_to_weight_slot = {
        const_idx: slot for slot, const_idx in enumerate(sorted_const_indices)
    }

    # Parse lib0.c to get function signatures and workspace requirements FIRST
    func_arg_counts, func_declarations = pack_tvm_dll(output_dir)

    # Find max workspace size from lib0.c
    max_workspace_size = 0
    lib0_path = output_dir / "lib" / "lib0.c"
    if lib0_path.exists():
        with open(lib0_path, "r") as f:
            for line in f:
                if "TVMBackendAllocWorkspace" in line:
                    match = re.search(
                        r"TVMBackendAllocWorkspace\([^,]+,[^,]+,\s*\(uint64_t\)(\d+)",
                        line,
                    )
                    if match:
                        size = int(match.group(1))
                        max_workspace_size = max(max_workspace_size, size)

    # Add safety margin (20%) for alignment
    workspace_size = int(max_workspace_size * 1.2) if max_workspace_size > 0 else (1024 * 1024)

    print(f"    Parsed {len(func_arg_counts)} function signatures from lib0.c")
    print(
        f"    Workspace required: {workspace_size / 1024 / 1024:.2f} MB "
        f"(max alloc: {max_workspace_size / 1024 / 1024:.2f} MB)"
    )

    # Save weights using filtered list
    bin_path, weight_map = WeightPacker.pack(weights, actual_weights_used, output_dir, save_metadata)

    # Generate header using sources module (no workspace - it's in lib0.c)
    code = sources.generate_mcu_header()

    # Generate function declarations
    op_names = sorted(set(op.func_name for op in parser.operations))
    code += sources.generate_function_declarations(func_declarations, op_names)

    # Generate weights section using sources module
    total_size_bytes = sum(weight_map[i]['aligned_size'] for i in range(len(weight_map)))
    total_floats = total_size_bytes // 4
    code += sources.generate_weights_section(total_size_bytes, total_floats)

    # Generate storage buffers
    max_storage_id = max(s.storage_id for s in parser.storages.values())
    storage_sizes = [0] * (max_storage_id + 1)

    for storage in parser.storages.values():
        storage_sizes[storage.storage_id] = max(
            storage_sizes[storage.storage_id],
            storage.size_bytes
        )

    total_storage = sum(storage_sizes)
    code += sources.generate_storage_buffers_section(storage_sizes)

    # Generate debug helper if enabled
    if dump_ir_tensor_data:
        code += sources.generate_debug_helper()

    # Inference function header
    num_buffers = len([s for s in storage_sizes if s > 0])
    code += f"""
/*========================================
 * Inference Function - MCU Optimized
 *========================================*/
int {model_name}_inference(float* input, float* output) {{
    /* Input: {input_shape}, Output: {output_shape} */
    /* Storages: {num_buffers}, Total: {total_storage / (1024 * 1024):.2f} MB */
    /* Operations: {len(parser.operations)} */

"""

    # Create storage pointer mapping (no intermediate variables)
    tensor_names = set(parser.tensors.keys())

    # Build mapping: tensor_name -> storage address string
    tensor_to_storage = {}
    for name, tensor in parser.tensors.items():
        if tensor.storage_id >= 0:
            storage_id = tensor.storage_id
            offset_floats = tensor.offset_bytes // 4
            tensor_to_storage[name] = f'&g_storage_{storage_id}[{offset_floats}]'
        else:
            # Reshape tensor - will be assigned during operations
            tensor_to_storage[name] = None

    # Build weight mapping: packed slot -> address string
    weight_to_storage = {}
    for i, info in weight_map.items():
        offset_floats = info['offset'] // 4
        weight_to_storage[i] = f'(float*)&g_weights[{offset_floats}]'

    code += "    /* Direct storage access - no intermediate pointer variables */\n"
    code += "    /* Tensors map directly to g_storage_X arrays */\n"
    code += "    /* Weights map directly to g_weights array */\n"
    code += "\n    /* Scalar constants */\n"
    code += "    float scalar_zero = 0.0f;\n"
    code += "    float scalar_six = 6.0f;\n"

    # Track scalar constants that need to be declared
    scalar_constants = {}  # value_str -> variable_name

    # FIRST PASS: Collect all scalar constants used in operations
    for op in parser.operations:
        if op.func_name != 'reshape':  # reshape doesn't use scalars in operation calls
            for arg in op.input_vars:
                if 'R.const(' in arg:
                    const_match = re.search(r'R\.const\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', arg)
                    if const_match:
                        const_value_str = const_match.group(1)
                        const_value = float(const_value_str)

                        # Skip common values (already declared)
                        if abs(const_value) < 1e-10 or abs(const_value - 6.0) < 1e-6:
                            continue

                        # Add to dictionary if not already there
                        if const_value_str not in scalar_constants:
                            scalar_var = f'scalar_const_{len(scalar_constants)}'
                            scalar_constants[const_value_str] = scalar_var

    # Declare all collected scalar constants
    for const_value_str, scalar_var in scalar_constants.items():
        code += f"    float {scalar_var} = {const_value_str}f;\n"

    # Track reshape assignments (tensor_name -> source_name)
    reshape_assignments = {}

    code += "\n    /* Operations - direct function calls with inline storage addresses */\n"
    # Use operations in parsed order (already in IR order from parse_relax_main_entry)
    for i, op in enumerate(parser.operations):
        code += f"\n    /* Op {i+1}/{len(parser.operations)}: {op.func_name} */\n"

        # Handle reshape - track pointer aliasing
        if op.func_name == 'reshape':
            if op.output_var in parser.tensors:
                source_var = op.input_vars[0] if op.input_vars else None
                if source_var:
                    # Extract actual source storage address
                    if source_var == input_var_name:
                        # Input tensor
                        source_addr = "input"
                    elif 'metadata["relax.expr.Constant"]' in source_var:
                        const_idx = match_relax_const_idx(source_var)
                        if const_idx is not None:
                            packed_slot = const_idx_to_weight_slot.get(const_idx)
                            if packed_slot is not None and packed_slot in weight_to_storage:
                                source_addr = weight_to_storage[packed_slot]
                            else:
                                source_addr = source_var
                        else:
                            source_addr = source_var
                    elif source_var in reshape_assignments:
                        # This is a reshaped tensor, use its source
                        source_addr = reshape_assignments[source_var]
                    elif source_var in tensor_to_storage:
                        source_addr = tensor_to_storage[source_var]
                    else:
                        source_addr = source_var

                    # Reshape: just remember the aliasing for later use
                    reshape_assignments[op.output_var] = source_addr
                    tensor_to_storage[op.output_var] = source_addr

                    code += f"    /* Reshape: {op.output_var} aliases {source_addr} */\n"

                    # Debug dump (only if enabled)
                    if dump_ir_tensor_data:
                        if True:
                            tensor_size = int(np.prod(parser.tensors[op.output_var].shape))
                            code += "#ifdef DEBUG_INTERMEDIATE\n"
                            code += (
                                f'    dump_tensor_data("Op{i+1}: {op.output_var} (reshape)", '
                                f"{source_addr}, {tensor_size}, 10);\n"
                            )
                            code += "#endif\n"
            continue

        # Build argument list for direct function call - use storage addresses directly
        arg_ptrs = []
        for arg in op.input_vars:
            if arg == input_var_name:
                # Dynamic input name (e.g., input_1_0, audio_in_0)
                arg_ptrs.append("input")
            elif 'R.const(' in arg:
                # Parse R.const(value, dtype) - extract the value
                const_match = re.search(
                    r"R\.const\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                    arg,
                )
                if const_match:
                    const_value_str = const_match.group(1)
                    const_value = float(const_value_str)

                    # Use already declared scalar constants
                    if abs(const_value) < 1e-10:
                        arg_ptrs.append("&scalar_zero")
                    elif abs(const_value - 6.0) < 1e-6:
                        arg_ptrs.append("&scalar_six")
                    elif const_value_str in scalar_constants:
                        # Reference already declared constant
                        arg_ptrs.append(f"&{scalar_constants[const_value_str]}")
                    else:
                        # This shouldn't happen if first pass worked correctly
                        # But handle it gracefully
                        scalar_var = f'scalar_const_{len(scalar_constants)}'
                        scalar_constants[const_value_str] = scalar_var
                        code += "    /* WARNING: Missed constant in first pass */\n"
                        code += f"    float {scalar_var} = {const_value_str}f;\n"
                        arg_ptrs.append(f"&{scalar_var}")
                else:
                    # Couldn't parse, skip
                    pass
            elif 'metadata["relax.expr.Constant"]' in arg:
                const_idx = match_relax_const_idx(arg)
                if const_idx is not None:
                    packed_slot = const_idx_to_weight_slot.get(const_idx)
                    if packed_slot is not None and packed_slot in weight_to_storage:
                        arg_ptrs.append(weight_to_storage[packed_slot])
                    else:
                        arg_ptrs.append("NULL")
                else:
                    arg_ptrs.append("NULL")
            elif arg in tensor_to_storage and tensor_to_storage[arg]:
                # Direct storage address
                arg_ptrs.append(tensor_to_storage[arg])
            else:
                # Unknown argument - skip
                pass

        # Add output pointer - direct storage address
        if op.output_var in tensor_to_storage and tensor_to_storage[op.output_var]:
            arg_ptrs.append(tensor_to_storage[op.output_var])

        # Get expected argument count from lib0.c signature
        expected_arg_count = func_arg_counts.get(op.func_name, len(arg_ptrs))

        # Adjust arguments to match expected count
        # If we have more arguments than expected, truncate
        # If we have fewer, the function signature might be wrong
        if len(arg_ptrs) > expected_arg_count:
            # Take the last N arguments (output is always last)
            arg_ptrs = arg_ptrs[-expected_arg_count:]
        elif len(arg_ptrs) < expected_arg_count:
            # This shouldn't happen, but log a warning
            code += (
                f"    /* WARNING: Expected {expected_arg_count} args, have {len(arg_ptrs)} */\n"
            )

        # Direct function call - no FFI overhead
        args_str = ", ".join(arg_ptrs)
        code += f"    {op.func_name}({args_str});\n"

        # Debug dump (only if enabled)
        if dump_ir_tensor_data:
            if op.output_var in tensor_to_storage:
                if op.output_var in parser.tensors and tensor_to_storage[op.output_var]:
                    tensor_size = int(np.prod(parser.tensors[op.output_var].shape))
                    output_addr = tensor_to_storage[op.output_var]
                    code += "#ifdef DEBUG_INTERMEDIATE\n"
                    code += (
                        f'    dump_tensor_data("Op{i+1}: {op.output_var} ({op.func_name})", '
                        f"{output_addr}, {tensor_size}, 10);\n"
                    )
                    code += "#endif\n"

    # Copy final output
    final_output = parser.operations[-1].output_var if parser.operations else None
    if final_output and final_output in tensor_to_storage:
        final_output_addr = tensor_to_storage[final_output]
        code += "\n    /* Copy output - direct storage access */\n"
        code += f"    memcpy(output, {final_output_addr}, {output_size} * sizeof(float));\n"

    code += "\n    return 0;\n"
    code += "}\n"

    # Generate main function using sources module
    code += sources.generate_main_function(model_name, input_size, output_size)

    lib_dir = output_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    output_path = lib_dir / f"{model_name}.c"
    with open(output_path, "w") as f:
        f.write(code)

    print(f"  Generated: {output_path} ({len(code)} bytes)")
    print(f"  Storages: {len(storage_sizes)}, Total: {total_storage / 1024 / 1024:.2f} MB")
    print(f"  Tensors: {len(parser.tensors)}, Operations: {len(parser.operations)}")

    # Patch lib0.c to add workspace implementation (remove TVM dependencies)
    lib0_path = output_dir / "lib" / "lib0.c"
    if lib0_path.exists():
        with open(lib0_path, "r") as f:
            lib0_content = f.read()

        # Check if already patched
        if "Workspace management - static allocation" not in lib0_content:
            # Remove TVM header includes
            lib0_content = lib0_content.replace('#include "tvm/runtime/base.h"\n', "")
            lib0_content = lib0_content.replace('#include "tvm/runtime/c_backend_api.h"\n', "")
            lib0_content = lib0_content.replace('#include "tvm/ffi/c_api.h"\n', "")
            lib0_content = lib0_content.replace("#include <cstddef>\n", "")

            # Find insertion point (after #define TVM_EXPORTS or at beginning of includes)
            # Look for the pattern: #define TVM_EXPORTS
            insert_pos = lib0_content.find("#define TVM_EXPORTS")
            if insert_pos != -1:
                # Find next newline
                insert_pos = lib0_content.find("\n", insert_pos) + 1
            else:
                # Fallback: insert after comment line
                insert_pos = lib0_content.find("\n") + 1

            # Prepare header section
            header_section = """
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define TVM_DLL __attribute__((visibility("default")))
"""

            # Generate workspace patch
            workspace_patch = sources.generate_lib0_workspace_patch(workspace_size)

            # Build new content
            patched_content = (
                lib0_content[:insert_pos] +
                header_section +
                workspace_patch +
                lib0_content[insert_pos:]
            )

            # Remove old includes that might be duplicated
            lines = patched_content.split('\n')
            seen_includes = set()
            filtered_lines = []

            for line in lines:
                # Skip duplicate includes and TVM includes
                if line.startswith('#include'):
                    if 'tvm/' in line or line in seen_includes:
                        continue
                    seen_includes.add(line)
                filtered_lines.append(line)

            patched_content = '\n'.join(filtered_lines)

            # Write patched lib0.c
            with open(lib0_path, 'w') as f:
                f.write(patched_content)

            print(f"  Patched lib0.c: Added workspace implementation, removed TVM headers")

    return True


class MCUCodegen:
    """MCU codegen pipeline wrapper."""

    def __init__(self,
                 ir_mod,
                 input_name: str,
                 input_shape: Tuple,
                 output_shape: Tuple,
                 weights: Dict[str, np.ndarray],
                 weight_order: List[str],
                 output_dir: Path,
                 model_name: str = "sense_model",
                 save_metadata: bool = True,
                 **kwargs):
        self.ir_mod = ir_mod
        self.input_name = input_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = weights
        self.weight_order = weight_order
        self.output_dir = output_dir
        self.model_name = model_name
        self.save_metadata = save_metadata
        self.kwargs = kwargs

    def run(self) -> bool:
        return _codegen_impl(
            self.ir_mod,
            self.input_name,
            self.input_shape,
            self.output_shape,
            self.weights,
            self.weight_order,
            self.output_dir,
            self.model_name,
            self.save_metadata,
            **self.kwargs
        )


def codegen(ir_mod,
            input_name: str,
            input_shape: Tuple,
            output_shape: Tuple,
            weights: Dict[str, np.ndarray],
            weight_order: List[str],
            output_dir: Path,
            model_name: str = "sense_model",
            save_metadata: bool = True,
            **kwargs) -> bool:
    """MCU codegen entrypoint."""
    return MCUCodegen(
        ir_mod,
        input_name,
        input_shape,
        output_shape,
        weights,
        weight_order,
        output_dir,
        model_name,
        save_metadata,
        **kwargs
    ).run()

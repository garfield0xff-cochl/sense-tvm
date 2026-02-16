import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class StorageInfo:
    storage_id: int
    size_bytes: int
    dtype: str
    scope: str
    first_use_line: int


@dataclass
class TensorAlloc:
    var_name: str
    storage_id: int
    offset_bytes: int
    shape: Tuple[int, ...]
    dtype: str
    line: int


@dataclass
class Operation:
    output_var: str
    func_name: str
    input_vars: List[str]
    line: int

@dataclass
class KillInfo:
    var_name: str
    line: int

class MainEntryCodegen:
    def __init__(self, ir_text: str):
        self.ir_text = ir_text
        self.storages: Dict[str, StorageInfo] = {}
        self.tensors: Dict[str, TensorAlloc] = {}
        self.operations: List[Operation] = []
        self.kills: List[KillInfo] = []
        self.storage_counter = 0

    def parse_relax_main_entry(self):
        lines = self.ir_text.split('\n')

        # Find main function
        main_start = -1
        for i, line in enumerate(lines):
            if 'def main(' in line:
                main_start = i
                break

        if main_start == -1:
            raise ValueError("Could not find main function in IR")

        for i in range(main_start, len(lines)):
            line = lines[i].strip()

            # Parse R.vm.alloc_storage
            if 'R.vm.alloc_storage' in line:
                self._parse_alloc_storage(line, i)

            # Parse R.vm.alloc_tensor
            elif 'R.vm.alloc_tensor' in line:
                self._parse_alloc_tensor(line, i)

            # Parse R.call_packed (e.g., reshape)
            elif 'R.call_packed' in line:
                self._parse_call_packed(line, i)

            # Parse operation calls (cls.xxx or _: ... = cls.xxx)
            elif 'cls.' in line and '(' in line:
                self._parse_operation(line, i)

            # Parse R.vm.kill_object
            elif 'R.vm.kill_object' in line:
                self._parse_kill(line, i)

            # Check for end of main
            elif line.startswith('return '):
                break

        print(f"    Parsed IR: {len(self.storages)} storages, {len(self.tensors)} tensors, "
              f"{len(self.operations)} ops, {len(self.kills)} kills")

    def _parse_alloc_storage(self, line: str, line_num: int):
        match = re.match(r'(\w+):\s*R\.Object\s*=\s*R\.vm\.alloc_storage', line)
        if not match:
            return

        storage_name = match.group(1)

        size_match = re.search(r'R\.shape\(\[(\d+)\]\)', line)
        if not size_match:
            return

        size_bytes = int(size_match.group(1))

        dtype = "uint8"
        dtype_match = re.search(r'R\.dtype\("(\w+)"\)', line)
        if dtype_match:
            dtype = dtype_match.group(1)

        scope = "global"

        storage_id = self.storage_counter
        self.storage_counter += 1

        self.storages[storage_name] = StorageInfo(
            storage_id=storage_id,
            size_bytes=size_bytes,
            dtype=dtype,
            scope=scope,
            first_use_line=line_num
        )

    def _parse_alloc_tensor(self, line: str, line_num: int):
        match = re.match(r'(\w+):\s*R\.Tensor\(\(([^)]+)\),\s*dtype="(\w+)"\)\s*=\s*R\.vm\.alloc_tensor\((\w+)', line)
        if not match:
            return

        tensor_name = match.group(1)
        shape_str = match.group(2)
        dtype = match.group(3)
        storage_name = match.group(4)

        # Parse shape
        shape_parts = [s.strip() for s in shape_str.split(',')]
        shape = tuple(int(s) for s in shape_parts if s.isdigit())

        # Extract offset
        offset_match = re.search(r'R\.prim_value\((\d+)\)', line)
        offset_bytes = int(offset_match.group(1)) if offset_match else 0

        if storage_name not in self.storages:
            print(f"    Warning: storage '{storage_name}' not found for tensor '{tensor_name}'")
            return

        storage_id = self.storages[storage_name].storage_id

        self.tensors[tensor_name] = TensorAlloc(
            var_name=tensor_name,
            storage_id=storage_id,
            offset_bytes=offset_bytes,
            shape=shape,
            dtype=dtype,
            line=line_num
        )        

    def _parse_operation(self, line: str, line_num: int):
        func_match = re.search(r'cls\.(\w+)\(', line)
        if not func_match:
            return

        func_name = func_match.group(1)

        # Extract arguments - find the matching parentheses
        start_paren = line.find('(', line.find('cls.'))
        if start_paren == -1:
            return

        # Find the last closing parenthesis (to handle nested parens in R.const(...))
        end_paren = line.rfind(')')
        if end_paren == -1:
            return

        args_str = line[start_paren+1:end_paren]

        # Split by comma but handle nested structures (brackets and parentheses)
        input_vars = []
        current_arg = ""
        bracket_depth = 0
        paren_depth = 0

        for char in args_str:
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            elif char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and bracket_depth == 0 and paren_depth == 0:
                arg = current_arg.strip()
                if arg:
                    input_vars.append(arg)
                current_arg = ""
                continue
            current_arg += char

        # Add last argument
        if current_arg.strip():
            input_vars.append(current_arg.strip())

        # Output is usually the last argument
        output_var = input_vars[-1] if input_vars else ""

        # TVM operations have format: func(inputs..., output)
        # So we separate the last argument as output
        if len(input_vars) > 0:
            actual_output = input_vars[-1]
            actual_inputs = input_vars[:-1]  # All but last
        else:
            actual_output = ""
            actual_inputs = []

        self.operations.append(Operation(
            output_var=actual_output,
            func_name=func_name,
            input_vars=actual_inputs,
            line=line_num
        ))

    def _parse_call_packed(self, line: str, line_num: int):
        """Parse: lv3: R.Tensor((1, 24, 1, 1), dtype="float32") = R.call_packed("vm.builtin.reshape", ...)"""
        # Extract variable name and shape - handle double parentheses R.Tensor((shape))
        var_match = re.match(r'(\w+):\s*R\.Tensor\(\(([^)]+)\),\s*dtype="(\w+)"\)\s*=\s*R\.call_packed', line)
        if not var_match:
            return

        var_name = var_match.group(1)
        shape_str = var_match.group(2)
        dtype = var_match.group(3)

        # Parse shape
        shape_parts = [s.strip() for s in shape_str.split(',')]
        shape = tuple(int(s) for s in shape_parts if s.isdigit())

        # Check if it's a reshape
        if 'vm.builtin.reshape' in line:
            # Extract source: either metadata["relax.expr.Constant"][idx] or tensor_name
            source = None
            source_for_codegen = None  # For generate_reshape
            if 'metadata["relax.expr.Constant"]' in line:
                const_match = re.search(r'metadata\["relax\.expr\.Constant"\]\[(\d+)\]', line)
                if const_match:
                    const_idx = int(const_match.group(1))
                    source = f'metadata["relax.expr.Constant"][{const_idx}]'  # Keep original format for weight_idx increment
                    source_for_codegen = f'weight_{const_idx}'  # For generate_reshape
            else:
                source_match = re.search(r'R\.call_packed\("vm\.builtin\.reshape",\s*(\w+)', line)
                if source_match:
                    source = source_match.group(1)
                    source_for_codegen = source_match.group(1)

            if source:
                # Store tensor info
                self.tensors[var_name] = TensorAlloc(
                    var_name=var_name,
                    storage_id=-1,  # Special: indicates reshape (no storage allocation needed)
                    offset_bytes=0,
                    shape=shape,
                    dtype=dtype,
                    line=line_num
                )

                # Add as reshape operation
                self.operations.append(Operation(
                    output_var=var_name,
                    func_name='reshape',
                    input_vars=[source],
                    line=line_num
                ))

    def _parse_kill(self, line: str, line_num: int):
        """Parse: _ = R.vm.kill_object(alloc)"""
        match = re.search(r'R\.vm\.kill_object\((\w+)\)', line)
        if not match:
            return

        var_name = match.group(1)

        self.kills.append(KillInfo(
            var_name=var_name,
            line=line_num
        ))

def save_weights(weights: Dict[str, np.ndarray],
                 weight_order: List[str],
                 output_dir: Path,
                 save_metadata: bool = True) -> Tuple[Path, Dict]:

    lib_dir = output_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    bin_path = lib_dir / "weights.bin"

    current_offset = 0
    weight_map = {}

    for idx, name in enumerate(weight_order):
        if name not in weights:
            continue

        weight_data = weights[name]
        size_bytes = weight_data.nbytes
        aligned_size = ((size_bytes + 63) // 64) * 64

        weight_map[idx] = {
            'name': name,
            'offset': current_offset,
            'size_bytes': size_bytes,
            'aligned_size': aligned_size,
            'shape': list(weight_data.shape),
            'dtype': str(weight_data.dtype)
        }

        current_offset += aligned_size

    total_floats = current_offset // 4
    packed_data = np.zeros(total_floats, dtype=np.float32)

    for idx, name in enumerate(weight_order):
        if name not in weights:
            continue

        weight_data = weights[name].astype(np.float32).flatten()
        offset_floats = weight_map[idx]['offset'] // 4
        size_floats = len(weight_data)

        packed_data[offset_floats:offset_floats + size_floats] = weight_data

    packed_data.tofile(bin_path)

    if save_metadata:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = metadata_dir / "weights.json"
        metadata = {
            'total_size_bytes': current_offset,
            'total_size_mb': current_offset / (1024 * 1024),
            'num_weights': len(weight_map),
            'weights': weight_map
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Weights: {bin_path} ({current_offset / 1024 / 1024:.2f} MB)")
        print(f"  Metadata: {metadata_path}")
    else:
        print(f"  Weights: {bin_path} ({current_offset / 1024 / 1024:.2f} MB)")

    return bin_path, weight_map


def main_entry_codegen(ir_mod,
            input_shape: Tuple,
            output_shape: Tuple,
            weights: Dict[str, np.ndarray],
            weight_order: List[str],
            output_dir: Path,
            model_name: str = "sense_model",
            save_metadata: bool = True) -> bool:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ir_text = ir_mod.script(show_meta=True)

    bin_path, weight_map = save_weights(weights, weight_order, output_dir, save_metadata)

    parser = MainEntryCodegen(ir_text)
    parser.parse_relax_main_entry()

    input_size = int(np.prod(input_shape))
    output_size = int(np.prod(output_shape))

    op_names = sorted(set(op.func_name for op in parser.operations))

    from .builtin import BUILTIN_CODEGEN
    code = BUILTIN_CODEGEN['header']()

    for op in op_names:
        code += f'extern int32_t __tvm_ffi_{op}(void* self, void* args, int32_t num_args, void* result);\n'

    total_size_bytes = sum(weight_map[i]['aligned_size'] for i in range(len(weight_map)))
    total_floats = total_size_bytes // 4
    code += BUILTIN_CODEGEN['weights'](total_size_bytes, total_floats)

    max_storage_id = max(s.storage_id for s in parser.storages.values())
    storage_sizes = [0] * (max_storage_id + 1)

    for storage in parser.storages.values():
        storage_sizes[storage.storage_id] = max(
            storage_sizes[storage.storage_id],
            storage.size_bytes
        )

    total_storage = sum(storage_sizes)
    code += BUILTIN_CODEGEN['storage_buffers'](storage_sizes)

    code += BUILTIN_CODEGEN['helpers']()

    num_buffers = len([s for s in storage_sizes if s > 0])

    code += f'''
/*========================================
 * Inference Function
 *========================================*/
int {model_name}_inference(float* input, float* output) {{
    /* Input: {input_shape}, Output: {output_shape} */
    /* Storages: {num_buffers}, Total: {total_storage / (1024 * 1024):.2f} MB */
    /* Tensors: {len(parser.tensors)}, Operations: {len(parser.operations)} */

'''

    code += '    /* Tensor declarations */\n'
    for name, tensor in sorted(parser.tensors.items()):
        shape_str = ', '.join(map(str, tensor.shape))
        code += f'    int64_t {name}_shape[4] = {{{shape_str}}};\n'
        code += f'    DLTensor {name}_tensor;\n'

    tensor_names = set(parser.tensors.keys())

    code += '\n    /* Scalar constants */\n'
    code += '    float p_zero = 0.0f;\n'
    code += '    float p_six = 6.0f;\n'
    code += '    int64_t scalar_shape[1] = {1};\n'
    code += '    DLTensor Tzero, Tsix;\n'
    code += '    init_tensor(&Tzero, &p_zero, 1, scalar_shape);\n'
    code += '    init_tensor(&Tsix, &p_six, 1, scalar_shape);\n'

    code += '\n    /* Input tensor */\n'
    code += f'    int64_t input_shape[4] = {{{", ".join(map(str, input_shape))}}};\n'
    code += '    DLTensor input_tensor;\n'
    code += '    init_tensor(&input_tensor, input, 4, input_shape);\n'

    code += '\n    /* Weight tensors */\n'
    for i, info in weight_map.items():
        offset_floats = info['offset'] // 4
        shape = info['shape']
        while len(shape) < 4:
            shape.append(1)
        shape_str = ', '.join(str(s) for s in shape[:4])
        ndim = len(info['shape'])
        code += f'    int64_t weight_{i}_shape[4] = {{{shape_str}}};\n'
        code += f'    DLTensor weight_{i}_tensor;\n'
        code += f'    init_tensor(&weight_{i}_tensor, (float*)&g_weights[{offset_floats}], {ndim}, weight_{i}_shape);\n'

    code += '\n    /* Operations */\n'
    weight_idx = 0

    # Use operations in parsed order (already in IR order from parse_relax_main_entry)
    for i, op in enumerate(parser.operations):
        code += f'\n    /* Op {i+1}/{len(parser.operations)}: {op.func_name} (weight_idx={weight_idx}) */\n'

        # Handle reshape as special builtin operation
        if op.func_name == 'reshape':
            if op.output_var in parser.tensors:
                tensor = parser.tensors[op.output_var]
                source_var = op.input_vars[0] if op.input_vars else None
                if source_var:
                    # Extract actual weight name for codegen
                    if 'metadata["relax.expr.Constant"]' in source_var:
                        const_match = re.search(r'metadata\["relax\.expr\.Constant"\]\[(\d+)\]', source_var)
                        if const_match:
                            const_idx = int(const_match.group(1))
                            source_for_codegen = f'weight_{const_idx}'
                        else:
                            source_for_codegen = source_var
                    else:
                        source_for_codegen = source_var

                    from .builtin import generate_reshape
                    code += generate_reshape(op.output_var, source_for_codegen, tensor.shape)
                    # Debug dump - focus on Op242-262 range (every op)
                    is_checkpoint = i < 10 or i >= len(parser.operations) - 20 or (242 <= i <= 262)
                    if is_checkpoint:
                        code += f'#ifdef DEBUG_INTERMEDIATE\n'
                        code += f'    dump_tensor("Op{i+1}: {op.output_var} (after reshape)", &{op.output_var}_tensor, 10);\n'
                        code += f'#endif\n'
            # IMPORTANT: Increment weight_idx if source was a weight constant
            # Check if any input_vars contains metadata constant
            for arg in op.input_vars:
                if 'metadata["relax.expr.Constant"]' in arg:
                    weight_idx += 1
            continue

        if op.output_var in parser.tensors:
            tensor = parser.tensors[op.output_var]
            storage_id = tensor.storage_id
            offset_floats = tensor.offset_bytes // 4

            code += f'    init_tensor(&{op.output_var}_tensor, &g_storage_{storage_id}[{offset_floats}], {len(tensor.shape)}, {op.output_var}_shape);\n'

        args = []
        for arg in op.input_vars:
            if arg == 'input_1_0':
                args.append('make_arg(&input_tensor)')
            elif 'R.const(0.0' in arg or 'R.const(0,' in arg:
                args.append('make_arg(&Tzero)')
            elif 'R.const(6.0' in arg or 'R.const(6,' in arg:
                args.append('make_arg(&Tsix)')
            elif 'metadata["relax.expr.Constant"]' in arg:
                args.append(f'make_arg(&weight_{weight_idx}_tensor)')
                weight_idx += 1
            elif arg in tensor_names:
                args.append(f'make_arg(&{arg}_tensor)')
            else:
                args.append('make_arg(NULL)')

        if op.output_var in tensor_names:
            args.append(f'make_arg(&{op.output_var}_tensor)')

        code += '    {\n'
        code += f'        TVMFFIAny args[] = {{{", ".join(args)}}};\n'
        code += f'        __tvm_ffi_{op.func_name}(NULL, args, {len(args)}, NULL);\n'
        code += '    }\n'

        # Debug dump - focus on Op242-262 range (every op)
        is_checkpoint = i < 10 or i >= len(parser.operations) - 20 or (242 <= i <= 262)
        if is_checkpoint and op.output_var in tensor_names:
            code += f'#ifdef DEBUG_INTERMEDIATE\n'
            code += f'    dump_tensor("Op{i+1}: {op.output_var} (after {op.func_name})", &{op.output_var}_tensor, 10);\n'
            code += f'#endif\n'

    final_output = parser.operations[-1].output_var if parser.operations else None
    if final_output and final_output in tensor_names:
        code += f'\n    /* Copy output */\n'
        code += f'    memcpy(output, {final_output}_tensor.data, {output_size} * sizeof(float));\n'

    code += '\n    return 0;\n'
    code += '}\n'

    code += BUILTIN_CODEGEN['main'](model_name, input_size, output_size)

    lib_dir = output_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    output_path = lib_dir / f"{model_name}.c"
    with open(output_path, 'w') as f:
        f.write(code)

    print(f"  Generated: {output_path} ({len(code)} bytes)")
    print(f"  Storages: {len(storage_sizes)}, Total: {total_storage / 1024 / 1024:.2f} MB")
    print(f"  Tensors: {len(parser.tensors)}, Operations: {len(parser.operations)}")

    return True



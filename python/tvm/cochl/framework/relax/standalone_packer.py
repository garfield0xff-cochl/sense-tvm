# SPDX-License-Identifier: Apache-2.0
"""Shared Relax IR parsing and codegen utilities for C/NCNN backends."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


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


class StandalonePacker:
    """Pack Relax main entry IR text into storage/tensor/operation lists."""

    def __init__(self, ir_text: str):
        self.ir_text = ir_text
        self.storages: Dict[str, StorageInfo] = {}
        self.tensors: Dict[str, TensorAlloc] = {}
        self.operations: List[Operation] = []
        self.kills: List[KillInfo] = []
        self.storage_counter = 0

    def pack(self):
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
                self._pack_alloc_storage(line, i)

            # Parse R.vm.alloc_tensor
            elif 'R.vm.alloc_tensor' in line:
                self._pack_alloc_tensor(line, i)

            # Parse R.call_packed (e.g., reshape)
            elif 'R.call_packed' in line:
                self._pack_builtin_call(line, i)

            # Parse operation calls (cls.xxx or _: ... = cls.xxx)
            elif 'cls.' in line and '(' in line:
                self._pack_operation(line, i)

            # Parse R.vm.kill_object
            elif 'R.vm.kill_object' in line:
                self._pack_kill(line, i)

            # Check for end of main
            elif line.startswith('return '):
                break

        print(
            f"    Parsed IR: {len(self.storages)} storages, {len(self.tensors)} tensors, "
            f"{len(self.operations)} ops, {len(self.kills)} kills"
        )

    def _pack_alloc_storage(self, line: str, line_num: int):
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

    def _pack_alloc_tensor(self, line: str, line_num: int):
        match = re.match(
            r'(\w+):\s*R\.Tensor\(\(([^)]+)\),\s*dtype="(\w+)"\)\s*=\s*R\.vm\.alloc_tensor\((\w+)',
            line,
        )
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

    def _pack_operation(self, line: str, line_num: int):
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

        args_str = line[start_paren + 1:end_paren]

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

    def _pack_builtin_call(self, line: str, line_num: int):
        """Parse: lv3: R.Tensor((1, 24, 1, 1), dtype="float32") = R.call_packed(...)"""
        # Extract variable name and shape - handle double parentheses R.Tensor((shape))
        var_match = re.match(
            r'(\w+):\s*R\.Tensor\(\(([^)]+)\),\s*dtype="(\w+)"\)\s*=\s*R\.call_packed',
            line,
        )
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
            if 'metadata["relax.expr.Constant"]' in line:
                const_match = re.search(r'metadata\["relax\.expr\.Constant"\]\[(\d+)\]', line)
                if const_match:
                    const_idx = int(const_match.group(1))
                    source = f'metadata["relax.expr.Constant"][{const_idx}]'
            else:
                source_match = re.search(r'R\.call_packed\("vm\.builtin\.reshape",\s*(\w+)', line)
                if source_match:
                    source = source_match.group(1)

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

    def _pack_kill(self, line: str, line_num: int):
        """Parse: _ = R.vm.kill_object(alloc)"""
        match = re.search(r'R\.vm\.kill_object\((\w+)\)', line)
        if not match:
            return

        var_name = match.group(1)

        self.kills.append(KillInfo(
            var_name=var_name,
            line=line_num
        ))


def pack_tvm_dll(output_dir: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Extract TVM_DLL function signatures and argument counts from lib0.c."""
    lib0_path = output_dir / "lib" / "lib0.c"
    if not lib0_path.exists():
        return {}, {}

    func_arg_counts = {}
    func_declarations = {}

    with open(lib0_path, "r") as f:
        for line in f:
            if line.startswith("TVM_DLL int32_t"):
                # Parse: TVM_DLL int32_t func_name(float* arg1, float* arg2, float* arg3);
                match = re.match(r"TVM_DLL int32_t (\w+)\((.*?)\);", line)
                if match:
                    func_name = match.group(1)
                    args_str = match.group(2)
                    # Count float* arguments
                    arg_count = args_str.count("float*")
                    func_arg_counts[func_name] = arg_count
                    # Store the full declaration (without TVM_DLL)
                    func_declarations[func_name] = f"extern int32_t {func_name}({args_str});"

    return func_arg_counts, func_declarations


def match_relax_const_idx(arg: str) -> Optional[int]:
    """Extract relax constant index from metadata reference."""
    match = re.search(r'metadata\["relax\.expr\.Constant"\]\[(\d+)\]', arg)
    if not match:
        return None
    return int(match.group(1))

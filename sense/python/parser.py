from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


def parse(model_path: str) -> Tuple:
    model_path = Path(model_path)
    ext = model_path.suffix.lower()

    if ext == '.onnx':
        return parse_onnx(model_path)
    elif ext == '.tflite':
        return parse_tflite(model_path)
    elif ext in ['.pt', '.pth']:
        return parse_pytorch(model_path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def parse_onnx(model_path: Path) -> Tuple:
    import onnx
    import tvm
    from tvm.relax.frontend.onnx import from_onnx

    # Load and validate ONNX model
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)

    # Extract input/output information
    shape_dict = {}
    dtype_dict = {}
    input_info = {}
    output_info = {}

    dtype_map = {
        1: "float32",
        2: "uint8",
        3: "int8",
        6: "int32",
        7: "int64"
    }

    # Extract inputs (skip initializers/weights)
    initializer_names = {init.name for init in onnx_model.graph.initializer}

    for inp in onnx_model.graph.input:
        if inp.name in initializer_names:
            continue

        name = inp.name
        dims = [d.dim_value if d.dim_value else 1 for d in inp.type.tensor_type.shape.dim]
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, "float32")

        input_info[name] = {"shape": tuple(dims), "dtype": dtype}
        shape_dict[name] = tuple(dims)
        dtype_dict[name] = dtype

    # Extract outputs
    for out in onnx_model.graph.output:
        name = out.name
        dims = [d.dim_value if d.dim_value else 1 for d in out.type.tensor_type.shape.dim]
        dtype = dtype_map.get(out.type.tensor_type.elem_type, "float32")
        output_info[name] = {"shape": tuple(dims), "dtype": dtype}

    # Convert to Relax IR
    ir_mod = from_onnx(onnx_model, shape_dict, dtype_dict)

    # Extract weights in the order they are USED in graph.node (TVM Constant order)
    # TVM from_onnx processes initializers and stores them as Constant[i]
    # in the order they are first USED in operations, not initializer declaration order
    weight_order = []
    seen = set()

    for node in onnx_model.graph.node:
        for input_name_raw in node.input:
            if input_name_raw in initializer_names and input_name_raw not in seen:
                # Sanitize name for C compatibility
                sanitized = input_name_raw.replace(':', '_').replace('/', '_')
                weight_order.append(sanitized)
                seen.add(input_name_raw)

    # Extract all weight data
    weights = {}
    for initializer in onnx_model.graph.initializer:
        name = initializer.name.replace(':', '_').replace('/', '_')
        data = onnx.numpy_helper.to_array(initializer)
        weights[name] = data

    # Filter out scalars (TVM handles them as R.const(), not metadata Constant)
    # Remove weights with shape=() or total elements < 2
    filtered_weight_order = []
    removed_count = 0

    for name in weight_order:
        if name in weights:
            w = weights[name]
            total_size = int(np.prod(w.shape)) if w.ndim > 0 else 0

            # Skip scalars (TVM uses R.const())
            if total_size < 2:
                removed_count += 1
                continue

            # Skip special constants that TVM inlines (pad_const, const_axes)
            if 'pad_const' in name or 'const_axes' in name:
                removed_count += 1
                continue

            filtered_weight_order.append(name)

    print(f"  Weights: {len(filtered_weight_order)} in TVM Constant order ({removed_count} scalars removed)")

    return ir_mod, input_info, output_info, weights, filtered_weight_order


def parse_tflite(model_path: Path) -> Tuple:
    raise NotImplementedError("TFLite parsing not yet implemented")


def parse_pytorch(model_path: Path) -> Tuple:
    raise NotImplementedError("PyTorch parsing not yet implemented")

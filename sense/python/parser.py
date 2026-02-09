# SPDX-License-Identifier: Apache-2.0
"""
Sense Parser Module

ONNX model parsing and weight extraction utilities.
Modularizes the parsing logic from Sense class.
"""

from typing import Dict, Tuple
import numpy as np
import onnx

import tvm
from tvm.relax.frontend.onnx import from_onnx


def parse_onnx(model_path: str,
               input_name: str = None,
               input_shape: Tuple = None,
               input_dtype: str = "float32") -> Tuple[tvm.IRModule, Dict, Dict]:
    """Parse ONNX model to Relax IR and extract I/O information.

    Parameters
    ----------
    model_path : str
        Path to ONNX model file.
    input_name : str, optional
        Override input name (auto-detected if None).
    input_shape : Tuple, optional
        Override input shape (auto-detected if None).
    input_dtype : str
        Input data type (default: "float32").

    Returns
    -------
    ir_mod : tvm.IRModule
        Relax IR module.
    input_info : Dict
        Dictionary of input information {name: {"shape": tuple, "dtype": str}}.
    output_info : Dict
        Dictionary of output information {name: {"shape": tuple, "dtype": str}}.
    """
    # Load and validate ONNX model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # Extract input information
    shape_dict = {}
    dtype_dict = {}
    input_info = {}

    dtype_map = {
        1: "float32",
        2: "uint8",
        3: "int8",
        6: "int32",
        7: "int64"
    }

    for inp in onnx_model.graph.input:
        # Skip initializers (weights)
        initializer_names = {init.name for init in onnx_model.graph.initializer}
        if inp.name in initializer_names:
            continue

        name = inp.name
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else 1)

        dtype = dtype_map.get(inp.type.tensor_type.elem_type, "float32")

        input_info[name] = {"shape": tuple(dims), "dtype": dtype}
        shape_dict[name] = tuple(dims)
        dtype_dict[name] = dtype

    # Extract output information
    output_info = {}
    for out in onnx_model.graph.output:
        name = out.name
        dims = []
        for d in out.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else 1)

        dtype = dtype_map.get(out.type.tensor_type.elem_type, "float32")
        output_info[name] = {"shape": tuple(dims), "dtype": dtype}

    # Override from parameters if provided
    if input_name and input_shape:
        shape_dict = {input_name: input_shape}
        dtype_dict = {input_name: input_dtype}
        input_info = {input_name: {"shape": input_shape, "dtype": input_dtype}}

    # Convert to Relax IR
    ir_mod = from_onnx(onnx_model, shape_dict, dtype_dict)

    return ir_mod, input_info, output_info


def parse_weights(model_path: str) -> Dict[str, np.ndarray]:
    """Extract weights from ONNX model.

    Parameters
    ----------
    model_path : str
        Path to ONNX model file.

    Returns
    -------
    weights : Dict[str, np.ndarray]
        Dictionary of weights {name: numpy array}.
    """
    # Load ONNX model
    onnx_model = onnx.load(model_path)

    # Extract weights from initializers
    weights = {}
    for initializer in onnx_model.graph.initializer:
        # Sanitize name for C compatibility
        name = initializer.name.replace(':', '_').replace('/', '_')
        data = onnx.numpy_helper.to_array(initializer)
        weights[name] = data

    return weights


def get_model_info(model_path: str) -> Dict:
    """Get comprehensive model information without conversion.

    Parameters
    ----------
    model_path : str
        Path to ONNX model file.

    Returns
    -------
    info : Dict
        Model information including inputs, outputs, nodes, and weights.
    """
    onnx_model = onnx.load(model_path)

    dtype_map = {
        1: "float32",
        2: "uint8",
        3: "int8",
        6: "int32",
        7: "int64"
    }

    # Extract input info
    inputs = {}
    initializer_names = {init.name for init in onnx_model.graph.initializer}
    for inp in onnx_model.graph.input:
        if inp.name in initializer_names:
            continue

        dims = [d.dim_value if d.dim_value else 1 for d in inp.type.tensor_type.shape.dim]
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, "float32")
        inputs[inp.name] = {"shape": tuple(dims), "dtype": dtype}

    # Extract output info
    outputs = {}
    for out in onnx_model.graph.output:
        dims = [d.dim_value if d.dim_value else 1 for d in out.type.tensor_type.shape.dim]
        dtype = dtype_map.get(out.type.tensor_type.elem_type, "float32")
        outputs[out.name] = {"shape": tuple(dims), "dtype": dtype}

    # Count weights and total size
    total_weight_size = 0
    weight_count = len(onnx_model.graph.initializer)
    for initializer in onnx_model.graph.initializer:
        data = onnx.numpy_helper.to_array(initializer)
        total_weight_size += data.nbytes

    return {
        "model_path": model_path,
        "inputs": inputs,
        "outputs": outputs,
        "num_nodes": len(onnx_model.graph.node),
        "num_weights": weight_count,
        "total_weight_size_bytes": total_weight_size,
        "total_weight_size_mb": total_weight_size / (1024 * 1024)
    }


def parse_model(model_path: str,
                input_name: str = None,
                input_shape: Tuple = None,
                input_dtype: str = "float32") -> Tuple[tvm.IRModule, Dict, Dict, Dict]:
    """All-in-one model parsing function.

    Parameters
    ----------
    model_path : str
        Path to ONNX model file.
    input_name : str, optional
        Override input name.
    input_shape : Tuple, optional
        Override input shape.
    input_dtype : str
        Input data type.

    Returns
    -------
    ir_mod : tvm.IRModule
        Relax IR module.
    input_info : Dict
        Input information.
    output_info : Dict
        Output information.
    weights : Dict[str, np.ndarray]
        Model weights.
    """
    # Parse ONNX to Relax IR
    ir_mod, input_info, output_info = parse_onnx(
        model_path,
        input_name=input_name,
        input_shape=input_shape,
        input_dtype=input_dtype
    )

    # Extract weights
    weights = parse_weights(model_path)

    return ir_mod, input_info, output_info, weights

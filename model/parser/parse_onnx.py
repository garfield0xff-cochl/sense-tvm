import onnx
from onnx import numpy_helper
import numpy as np
import os


def parse_onnx_layers(onnx_path):
    """Parse ONNX model and extract basic information"""
    model = onnx.load(onnx_path)
    graph = model.graph

    input_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]

    weights = {}
    for init in graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    return input_shape, output_shape, graph, weights


def extract_onnx_weights(onnx_path, output_dir="../bin/sense_params"):
    """Extract and save ONNX weights to binary files"""

    # Load ONNX model
    model = onnx.load(onnx_path)
    graph = model.graph

    os.makedirs(output_dir, exist_ok=True)

    weights = {}
    for init in graph.initializer:
        name = init.name
        weight = numpy_helper.to_array(init)
        weights[name] = weight

    # Map ONNX weight names to layer names
    saved_count = 0

    # Process each weight tensor
    for name, weight in weights.items():
        # Skip non-weight tensors (like padding constants, min/max values)
        if 'pad_const' in name or 'min__' in name or 'max__' in name:
            continue

        # Determine layer type and save with appropriate name
        saved_name = None

        # Determine which submodel this weight belongs to
        model_prefix = ""
        if 'model_2/model_1/m_main_1' in name:
            model_prefix = "m1_"
        elif 'model_2/model/m_main_0' in name:
            model_prefix = "m0_"

        # First Conv layer
        if 'Conv1' in name and 'Conv_1' not in name:
            if 'weights' in name:
                saved_name = f'{model_prefix}conv1_weight.bin'
                print(f"  {model_prefix}Conv1 weight: {weight.shape} -> {saved_name}")
            elif 'bias' in name:
                saved_name = f'{model_prefix}conv1_bias.bin'
                print(f"  {model_prefix}Conv1 bias: {weight.shape} -> {saved_name}")

        # Inverted residual blocks
        elif 'block_' in name:
            # Extract block number from name
            parts = name.split('block_')
            if len(parts) > 1:
                block_info = parts[1].split('_')
                if len(block_info) >= 2:
                    block_num = block_info[0]
                    layer_type = '_'.join(block_info[1:])

                    # Expand layer
                    if 'expand' in layer_type and 'Conv2D' in name:
                        if 'weights' in name:
                            saved_name = f'{model_prefix}blocks_{block_num}_expand_conv_weight.bin'
                        elif 'bias' in name:
                            saved_name = f'{model_prefix}blocks_{block_num}_expand_conv_bias.bin'

                    # Depthwise layer
                    elif 'depthwise' in layer_type:
                        if 'weights' in name:
                            saved_name = f'{model_prefix}blocks_{block_num}_dw_conv_weight.bin'
                        elif 'bias' in name:
                            saved_name = f'{model_prefix}blocks_{block_num}_dw_conv_bias.bin'

                    # Project layer
                    elif 'project' in layer_type and 'Conv2D' in name:
                        if 'weights' in name:
                            saved_name = f'{model_prefix}blocks_{block_num}_project_conv_weight.bin'
                        elif 'bias' in name:
                            saved_name = f'{model_prefix}blocks_{block_num}_project_conv_bias.bin'

        # Expanded conv (first layer in some models, block 0)
        elif 'expanded_conv' in name:
            if 'depthwise' in name:
                if 'weights' in name:
                    saved_name = f'{model_prefix}blocks_0_dw_conv_weight.bin'
                elif 'bias' in name:
                    saved_name = f'{model_prefix}blocks_0_dw_conv_bias.bin'
            elif 'project' in name:
                if 'weights' in name:
                    saved_name = f'{model_prefix}blocks_0_project_conv_weight.bin'
                elif 'bias' in name:
                    saved_name = f'{model_prefix}blocks_0_project_conv_bias.bin'

        # Last conv layer (Conv_1)
        elif 'Conv_1' in name:
            if 'weights' in name or 'weight' in name.lower():
                saved_name = f'{model_prefix}conv_last_weight.bin'
                print(f"  {model_prefix}Last conv weight: {weight.shape} -> {saved_name}")
            elif 'bias' in name:
                saved_name = f'{model_prefix}conv_last_bias.bin'
                print(f"  {model_prefix}Last conv bias: {weight.shape} -> {saved_name}")

        # Fully connected / Logit layer
        elif 'logit' in name or 'Logit' in name:
            if 'MatMul' in name or 'weight' in name.lower() or 'kernel' in name.lower():
                saved_name = f'{model_prefix}fc_weight.bin'
                # FC weight might need transpose
                if len(weight.shape) == 2:
                    # ONNX typically stores as [in_features, out_features]
                    # TVM expects [out_features, in_features]
                    print(f"  {model_prefix}FC weight (original): {weight.shape}")
                    weight = weight.T  # Transpose
                    print(f"  {model_prefix}FC weight (transposed): {weight.shape} -> {saved_name}")
            elif 'bias' in name or 'Bias' in name:
                saved_name = f'{model_prefix}fc_bias.bin'
                print(f"  {model_prefix}FC bias: {weight.shape} -> {saved_name}")

        # Save weight if name was determined
        if saved_name:
            output_path = os.path.join(output_dir, saved_name)
            weight.astype(np.float32).tofile(output_path)
            saved_count += 1
        else:
            # Debug: print unmatched weights
            if len(name) < 100:  # Only print short names
                print(f"  [skip] {name}: {weight.shape}")

    print()
    print(f"  extracted {saved_count} weight files to {output_dir}/")
    print()

    return saved_count
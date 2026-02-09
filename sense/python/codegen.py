# SPDX-License-Identifier: Apache-2.0
"""
Sense C Code Generator

Generates standalone C code from TVM Relax IR for embedded deployment.
Based on tvm_native_c_codegen/scripts/generate_full_model.py
"""

import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class SenseCodeGenerator:
    """Generate standalone C code from Relax IR."""

    def __init__(self, ir_content: str, weights_dir: Path, weight_files: List[str]):
        """Initialize code generator.

        Parameters
        ----------
        ir_content : str
            Relax IR content as string.
        weights_dir : Path
            Directory containing weight binary files.
        weight_files : List[str]
            List of weight file names.
        """
        self.ir_content = ir_content
        self.weights_dir = weights_dir
        self.weight_files = weight_files
        self.operations = []
        self.tensor_shapes = {}
        self.lv_to_const = {}

    def parse_ir(self):
        """Parse IR to extract operations and tensor information."""
        content = self.ir_content

        # Extract tensor shapes: alloc123: R.Tensor((1, 24, 64, 96), dtype="float32")
        for match in re.finditer(r'(alloc\d*): R\.Tensor\(\(([^)]+)\)', content):
            name = match.group(1)
            try:
                dims = [int(x.strip()) for x in match.group(2).split(',')]
                self.tensor_shapes[name] = dims
            except:
                pass

        # Build lv -> Constant index mapping
        for m in re.finditer(
            r'(lv\d+).*?R\.call_packed\("vm\.builtin\.reshape",\s*'
            r'metadata\["relax\.expr\.Constant"\]\[(\d+)\]',
            content
        ):
            self.lv_to_const[m.group(1)] = int(m.group(2))

        # Extract all operations from cls.xxx(...) calls
        for match in re.finditer(r'cls\.([a-z_0-9]+)\(', content):
            op_name = match.group(1)
            start = match.end()

            # Find balanced closing paren
            depth = 1
            end = start
            while depth > 0 and end < len(content):
                if content[end] == '(':
                    depth += 1
                elif content[end] == ')':
                    depth -= 1
                end += 1
            args_raw = content[start:end-1]

            # Parse each argument
            args = []
            for arg in args_raw.split(','):
                arg = arg.strip()
                if 'metadata' in arg:
                    idx = re.search(r'\[(\d+)\]', arg)
                    args.append(('weight', int(idx.group(1)) if idx else 0))
                elif arg.startswith('alloc'):
                    args.append(('alloc', arg))
                elif arg.startswith('input'):
                    args.append(('input', arg))
                elif 'R.const' in arg:
                    val = re.search(r'R\.const\(([\d.]+)', arg)
                    args.append(('const', float(val.group(1)) if val else 0.0))
                elif arg.startswith('lv'):
                    lv_name = arg.split(':')[0].split(')')[0].strip()
                    if lv_name in self.lv_to_const:
                        args.append(('weight', self.lv_to_const[lv_name]))
                    else:
                        args.append(('lv', arg))
                else:
                    args.append(('other', arg))

            self.operations.append({'op': op_name, 'args': args, 'raw': args_raw})

        return self

    def _order_weights(self) -> List[str]:
        """Order weights according to model structure.

        Based on generate_full_model.py weight ordering pattern.
        Returns ordered list of weight file names.
        """
        ordered = []

        # Branch 0 (m_main_1) - Constant[0-103]
        # Conv1 [0-1]
        ordered.extend([
            "model_2_model_1_m_main_1_Conv1_Conv2D_weights_fused_bn.bin",
            "model_2_model_1_m_main_1_Conv1_Conv2D_bias_fused_bn.bin",
        ])
        # expanded_conv [2-5]
        ordered.extend([
            "model_2_model_1_m_main_1_expanded_conv_depthwise_depthwise_weights_fused_bn.bin",
            "model_2_model_1_m_main_1_expanded_conv_depthwise_depthwise_bias_fused_bn.bin",
            "model_2_model_1_m_main_1_expanded_conv_project_Conv2D_weights_fused_bn.bin",
            "model_2_model_1_m_main_1_expanded_conv_project_Conv2D_bias_fused_bn.bin",
        ])
        # block_1 ~ block_16 [6-101] (16 blocks * 6 weights = 96)
        for b in range(1, 17):
            ordered.extend([
                f"model_2_model_1_m_main_1_block_{b}_expand_Conv2D_weights_fused_bn.bin",
                f"model_2_model_1_m_main_1_block_{b}_expand_Conv2D_bias_fused_bn.bin",
                f"model_2_model_1_m_main_1_block_{b}_depthwise_depthwise_weights_fused_bn.bin",
                f"model_2_model_1_m_main_1_block_{b}_depthwise_depthwise_bias_fused_bn.bin",
                f"model_2_model_1_m_main_1_block_{b}_project_Conv2D_weights_fused_bn.bin",
                f"model_2_model_1_m_main_1_block_{b}_project_Conv2D_bias_fused_bn.bin",
            ])
        # Conv_1 [102-103]
        ordered.extend([
            "model_2_model_1_m_main_1_Conv_1_Conv2D_weights_fused_bn.bin",
            "model_2_model_1_m_main_1_Conv_1_Conv2D_bias_fused_bn.bin",
        ])

        # Branch 1 (m_main_0) - Constant[104-207]
        # Conv1 [104-105]
        ordered.extend([
            "model_2_model_m_main_0_Conv1_Conv2D_weights_fused_bn.bin",
            "model_2_model_m_main_0_Conv1_Conv2D_bias_fused_bn.bin",
        ])
        # expanded_conv [106-109]
        ordered.extend([
            "model_2_model_m_main_0_expanded_conv_depthwise_depthwise_weights_fused_bn.bin",
            "model_2_model_m_main_0_expanded_conv_depthwise_depthwise_bias_fused_bn.bin",
            "model_2_model_m_main_0_expanded_conv_project_Conv2D_weights_fused_bn.bin",
            "model_2_model_m_main_0_expanded_conv_project_Conv2D_bias_fused_bn.bin",
        ])
        # block_1 ~ block_16 [110-205] (16 blocks * 6 weights = 96)
        for b in range(1, 17):
            ordered.extend([
                f"model_2_model_m_main_0_block_{b}_expand_Conv2D_weights_fused_bn.bin",
                f"model_2_model_m_main_0_block_{b}_expand_Conv2D_bias_fused_bn.bin",
                f"model_2_model_m_main_0_block_{b}_depthwise_depthwise_weights_fused_bn.bin",
                f"model_2_model_m_main_0_block_{b}_depthwise_depthwise_bias_fused_bn.bin",
                f"model_2_model_m_main_0_block_{b}_project_Conv2D_weights_fused_bn.bin",
                f"model_2_model_m_main_0_block_{b}_project_Conv2D_bias_fused_bn.bin",
            ])
        # Conv_1 [206-207]
        ordered.extend([
            "model_2_model_m_main_0_Conv_1_Conv2D_weights_fused_bn.bin",
            "model_2_model_m_main_0_Conv_1_Conv2D_bias_fused_bn.bin",
        ])

        # FC weights - Constant[208-211]
        ordered.extend([
            "model_2_model_1_m_main_1_logit_MatMul_ReadVariableOp_0.bin",  # [208]
            "model_2_model_1_m_main_1_logit_BiasAdd_ReadVariableOp_0.bin",  # [209]
            "model_2_model_m_main_0_logit_MatMul_ReadVariableOp_0.bin",  # [210]
            "model_2_model_m_main_0_logit_BiasAdd_ReadVariableOp_0.bin",  # [211]
        ])

        return ordered

    def generate_c_code(self, model_name: str = "sense_model",
                       input_shape: Tuple = (1, 128, 192, 1),
                       output_shape: Tuple = (1, 863)) -> str:
        """Generate complete C code with all operation handling.

        Parameters
        ----------
        model_name : str
            Model name for generated code.
        input_shape : Tuple
            Input tensor shape.
        output_shape : Tuple
            Output tensor shape.

        Returns
        -------
        str
            Generated C code.
        """
        op_names = sorted(set(op['op'] for op in self.operations))
        input_size = 1
        for d in input_shape:
            input_size *= d
        output_size = 1
        for d in output_shape:
            output_size *= d

        # Reorder weights according to model structure
        ordered_weights = self._order_weights()
        num_weights = 220  # Fixed size for dual-branch MobileNetV2

        code = f'''/**
 * Sense Standalone Model - Generated C Code
 * Model: {model_name}
 * Operations: {len(self.operations)}
 * Input: {input_shape}
 * Output: {output_shape}
 *
 * Generated by Sense Code Generator
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#include "tvm/ffi/c_api.h"
#include "tvm/runtime/c_backend_api.h"

extern int tvm_workspace_init(void);
extern void tvm_workspace_cleanup(void);
extern void tvm_workspace_reset(void);

/* TVM operators */
'''
        for op in op_names:
            code += f'extern int32_t __tvm_ffi_{op}(void* self, void* args, int32_t num_args, void* result);\n'

        code += f'''
/*========================================
 * Helpers
 *========================================*/
static double get_time_ms(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}}

static void init_tensor(DLTensor* t, float* data, int ndim, int64_t* shape) {{
    memset(t, 0, sizeof(DLTensor));
    t->data = data;
    t->device.device_type = kDLCPU;
    t->ndim = ndim;
    t->dtype.code = 2;
    t->dtype.bits = 32;
    t->dtype.lanes = 1;
    t->shape = shape;
}}

static TVMFFIAny make_arg(DLTensor* t) {{
    TVMFFIAny a; a.type_index = 0; a.v_ptr = t; return a;
}}

static float* load_weights(const char* fn, size_t* n) {{
    FILE* f = fopen(fn, "rb");
    if (!f) {{ *n = 0; return NULL; }}
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n = sz / 4;
    float* d = (float*)malloc(sz);
    fread(d, 4, *n, f);
    fclose(f);
    return d;
}}

#define POOL_SIZE (300*1024*1024)
static float* g_pool = NULL;
static size_t g_pool_off = 0;

static float* pool_alloc(size_t n) {{
    size_t a = ((n*4+63)&~63ULL)/4;
    if (g_pool_off + a > POOL_SIZE/4) return NULL;
    float* p = g_pool + g_pool_off;
    g_pool_off += a;
    return p;
}}
static void pool_reset(void) {{ g_pool_off = 0; }}

/*========================================
 * Weights
 *========================================*/
#define NUM_W {num_weights}
static float* g_w[NUM_W];
static size_t g_ws[NUM_W];

static const char* W_FILES[] = {{
'''
        # Generate weight files array with ordered weights
        avail = set(self.weight_files)
        code += '    // Branch 0 (m_main_1) - Constant[0-103]\n'
        for i, wf in enumerate(ordered_weights):
            if i == 104:
                code += '    // Branch 1 (m_main_0) - Constant[104-207]\n'
            elif i == 208:
                code += '    // FC weights - Constant[208-211]\n'
            if wf in avail:
                code += f'    "{wf}",  // [{i}]\n'
            else:
                code += f'    NULL,  // [{i}] missing: {wf}\n'
        code += '    NULL\n};\n\n'

        code += '''static int load_weights_all(void) {
    printf("Loading weights...\\n");
    size_t tot = 0; int ld = 0;
    for (int i = 0; i < NUM_W && W_FILES[i]; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", "weights", W_FILES[i]);
        g_w[i] = load_weights(path, &g_ws[i]);
        if (g_w[i]) { tot += g_ws[i]*4; ld++; }
        else { g_w[i] = (float*)calloc(4096,4); g_ws[i] = 4096; }
    }
    for (int i = ld; i < NUM_W; i++) {
        if (!g_w[i]) { g_w[i] = (float*)calloc(4096,4); g_ws[i] = 4096; }
    }
    printf("  Loaded %d files (%.2f MB)\\n", ld, tot/1048576.0);
    return 0;
}
static void free_weights_all(void) {
    for (int i = 0; i < NUM_W; i++) { free(g_w[i]); g_w[i] = NULL; }
}

'''

        # Generate model_forward function with detailed operation handling
        code += f'''/*========================================
 * Model Forward - {len(self.operations)} Operations
 *========================================*/
int model_forward(float* input, float* output) {{
'''

        # Generate buffer declarations
        code += f'    static float* buf[{len(self.operations) + 10}];\n'
        code += f'    static int64_t sh[{len(self.operations) + 10}][4];\n'
        code += f'    static DLTensor T[{len(self.operations) * 2 + 10}];\n'
        code += '    TVMFFIAny A[10];\n'
        code += '    int ti = 0;\n'
        code += f'    memset(buf, 0, sizeof(buf));\n'
        code += f'    memset(T, 0, sizeof(T));\n\n'

        # Scalar tensors
        code += '    /* Scalars */\n'
        code += '    float* p_zero = pool_alloc(1);\n'
        code += '    float* p_six = pool_alloc(1);\n'
        code += '    *p_zero = 0.0f;\n'
        code += '    *p_six = 6.0f;\n\n'
        code += '    int64_t sz[1] = {1};\n'
        code += '    DLTensor Tzero, Tsix;\n'
        code += '    init_tensor(&Tzero, p_zero, 1, sz);\n'
        code += '    init_tensor(&Tsix, p_six, 1, sz);\n\n'

        # Input tensor
        input_shape_str = ', '.join(str(d) for d in input_shape)
        code += '    /* Input */\n'
        code += f'    int64_t s_in[{len(input_shape)}] = {{{input_shape_str}}};\n'
        code += '    DLTensor Tin;\n'
        code += f'    init_tensor(&Tin, input, {len(input_shape)}, s_in);\n\n'

        # Process each operation with detailed handling
        alloc_map = {}
        buf_idx = 0
        tensor_idx = 0
        weight_idx = 0
        matmul_count = 0
        add39_count = 0

        for op_idx, op in enumerate(self.operations):
            op_name = op['op']
            args = op['args']

            # Find output tensor
            out_name = None
            for atype, aval in reversed(args):
                if atype == 'alloc':
                    out_name = aval
                    break

            # Get shape
            if out_name and out_name in self.tensor_shapes:
                shape = self.tensor_shapes[out_name]
            else:
                shape = [1, 24, 64, 96]

            buf_size = 1
            for s in shape:
                buf_size *= s
            ndim = len(shape)

            code += f'    /* Op {op_idx+1}: {op_name} */\n'
            code += f'    buf[{buf_idx}] = pool_alloc({buf_size});\n'

            if ndim == 4:
                code += f'    sh[{buf_idx}][0]={shape[0]}; sh[{buf_idx}][1]={shape[1]}; sh[{buf_idx}][2]={shape[2]}; sh[{buf_idx}][3]={shape[3]};\n'
            elif ndim == 2:
                code += f'    sh[{buf_idx}][0]={shape[0]}; sh[{buf_idx}][1]={shape[1]};\n'
            else:
                code += f'    sh[{buf_idx}][0]={shape[0] if shape else 1};\n'

            code += f'    init_tensor(&T[ti], buf[{buf_idx}], {ndim}, sh[{buf_idx}]); ti++;\n'

            if out_name:
                alloc_map[out_name] = (buf_idx, tensor_idx)

            # Generate operation call based on type (from generate_full_model.py)
            if op_name == 'concatenate':
                code += '    A[0] = make_arg(&Tin);\n'
                code += '    A[1] = make_arg(&Tin);\n'
                code += '    A[2] = make_arg(&Tin);\n'
                code += f'    A[3] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_concatenate(NULL, A, 4, NULL);\n'

            elif op_name == 'transpose':
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += f'    A[1] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_transpose(NULL, A, 2, NULL);\n'

            elif op_name.startswith('conv2d'):
                # input, weight, output
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                    elif atype == 'lv':
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'

                # Weight tensor
                w_idx = weight_idx
                for atype, aval in args:
                    if atype == 'weight':
                        w_idx = aval
                        break

                code += '    {\n'
                code += f'        static int64_t ws[4] = {{1,1,1,1}};\n'
                code += f'        static DLTensor Tw;\n'
                code += f'        init_tensor(&Tw, g_w[{w_idx % num_weights}], 4, ws);\n'
                code += '        A[1] = make_arg(&Tw);\n'
                code += '    }\n'
                code += f'    A[2] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_{op_name}(NULL, A, 3, NULL);\n'
                weight_idx += 1

            elif op_name.startswith('add'):
                has_weight = any(atype == 'weight' for atype, _ in args)
                has_lv = any(atype == 'lv' for atype, _ in args)

                if has_weight or has_lv:
                    # Bias add
                    in_name = None
                    for atype, aval in args:
                        if atype == 'alloc' and aval != out_name:
                            in_name = aval
                            break
                    if in_name and in_name in alloc_map:
                        _, in_ti = alloc_map[in_name]
                        code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                    else:
                        code += f'    A[0] = make_arg(&T[ti-2]);\n'

                    # Get bias index
                    bias_const_idx = None
                    for atype, aval in args:
                        if atype == 'weight':
                            bias_const_idx = aval
                            break

                    if op_name == 'add39':
                        if bias_const_idx is not None:
                            b_idx = bias_const_idx
                        else:
                            b_idx = 209 if add39_count == 0 else 211
                        add39_count += 1
                        code += '    {\n'
                        code += f'        static int64_t bs[1] = {{863}};\n'
                        code += f'        static DLTensor Tb;\n'
                        code += f'        init_tensor(&Tb, g_w[{b_idx}], 1, bs);\n'
                        code += '        A[1] = make_arg(&Tb);\n'
                        code += '    }\n'
                    else:
                        if bias_const_idx is not None:
                            b_idx = bias_const_idx
                        else:
                            b_idx = weight_idx
                            weight_idx += 1
                        code += '    {\n'
                        code += f'        static int64_t bs[4] = {{1,1,1,1}};\n'
                        code += f'        static DLTensor Tb;\n'
                        code += f'        init_tensor(&Tb, g_w[{b_idx}], 4, bs);\n'
                        code += '        A[1] = make_arg(&Tb);\n'
                        code += '    }\n'
                    code += f'    A[2] = make_arg(&T[ti-1]);\n'
                else:
                    # Residual add
                    inputs = []
                    for atype, aval in args:
                        if atype == 'alloc' and aval != out_name:
                            inputs.append(aval)

                    if len(inputs) >= 2:
                        if inputs[0] in alloc_map:
                            _, ti0 = alloc_map[inputs[0]]
                            code += f'    A[0] = make_arg(&T[{ti0}]);\n'
                        else:
                            code += f'    A[0] = make_arg(&T[ti-3]);\n'
                        if inputs[1] in alloc_map:
                            _, ti1 = alloc_map[inputs[1]]
                            code += f'    A[1] = make_arg(&T[{ti1}]);\n'
                        else:
                            code += f'    A[1] = make_arg(&T[ti-2]);\n'
                    else:
                        code += f'    A[0] = make_arg(&T[ti-3]);\n'
                        code += f'    A[1] = make_arg(&T[ti-2]);\n'
                    code += f'    A[2] = make_arg(&T[ti-1]);\n'

                code += f'    __tvm_ffi_{op_name}(NULL, A, 3, NULL);\n'

            elif op_name.startswith('maximum'):
                # max(input, 0)
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += '    A[1] = make_arg(&Tzero);\n'
                code += f'    A[2] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_{op_name}(NULL, A, 3, NULL);\n'

            elif op_name.startswith('minimum'):
                # min(input, 6)
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += '    A[1] = make_arg(&Tsix);\n'
                code += f'    A[2] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_{op_name}(NULL, A, 3, NULL);\n'

            elif op_name.startswith('pad'):
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += f'    A[1] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_{op_name}(NULL, A, 2, NULL);\n'

            elif op_name == 'mean':
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += f'    A[1] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_mean(NULL, A, 2, NULL);\n'

            elif op_name == 'matmul':
                # MatMul weight indices from ordered W_FILES array
                # First matmul: g_w[208] (m_main_1_logit_MatMul)
                # Second matmul: g_w[210] (m_main_0_logit_MatMul)
                matmul_weight_idx = 208 if matmul_count == 0 else 210
                matmul_count += 1

                # Create 2D view of mean output
                code += '    {\n'
                code += '        /* Reshape mean output from (1,1280,1,1) to (1,1280) for matmul */\n'
                code += '        static int64_t rs[2] = {1, 1280};\n'
                code += '        static DLTensor Tr;\n'
                code += f'        init_tensor(&Tr, buf[{buf_idx-1}], 2, rs);\n'
                code += '        A[0] = make_arg(&Tr);\n'
                code += '    }\n'
                code += '    {\n'
                code += f'        static int64_t ms[2] = {{1280, 863}};\n'
                code += f'        static DLTensor Tm;\n'
                code += f'        init_tensor(&Tm, g_w[{matmul_weight_idx}], 2, ms);\n'
                code += '        A[1] = make_arg(&Tm);\n'
                code += '    }\n'
                code += f'    A[2] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_matmul(NULL, A, 3, NULL);\n'

            elif op_name == 'tir_sigmoid':
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += f'    A[1] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_tir_sigmoid(NULL, A, 2, NULL);\n'

            elif op_name == 'multiply':
                # multiply hardcodes *0.5 internally
                in_name = None
                for atype, aval in args:
                    if atype == 'alloc' and aval != out_name:
                        in_name = aval
                        break
                if in_name and in_name in alloc_map:
                    _, in_ti = alloc_map[in_name]
                    code += f'    A[0] = make_arg(&T[{in_ti}]);\n'
                else:
                    code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += f'    A[1] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_multiply(NULL, A, 2, NULL);\n'

            else:
                # Generic operation (fallback)
                code += f'    A[0] = make_arg(&T[ti-2]);\n'
                code += f'    A[1] = make_arg(&T[ti-1]);\n'
                code += f'    __tvm_ffi_{op_name}(NULL, A, 2, NULL);\n'

            buf_idx += 1
            tensor_idx += 1
            code += '\n'

        # Copy final output
        code += f'''    /* Copy output */
    memcpy(output, buf[{buf_idx-1}], {output_size} * sizeof(float));
    return 0;
}}

'''

        # Benchmark and main
        code += f'''/*========================================
 * Benchmark
 *========================================*/
void benchmark(int runs, int json, const char* input_file, const char* output_file) {{
    float* in = (float*)malloc({input_size}*4);
    float* out = (float*)malloc({output_size}*4);

    if (input_file) {{
        FILE* f = fopen(input_file, "rb");
        if (f) {{
            fread(in, 4, {input_size}, f);
            fclose(f);
        }} else {{
            fprintf(stderr, "Warning: Could not read input file, using random\\n");
            srand(42);
            for (int i = 0; i < {input_size}; i++) in[i] = (float)rand()/RAND_MAX;
        }}
    }} else {{
        srand(42);
        for (int i = 0; i < {input_size}; i++) in[i] = (float)rand()/RAND_MAX;
    }}

    for (int i = 0; i < 5; i++) {{ pool_reset(); model_forward(in, out); }}

    double* t = (double*)malloc(runs*8);
    for (int i = 0; i < runs; i++) {{
        pool_reset();
        double s = get_time_ms();
        model_forward(in, out);
        t[i] = get_time_ms() - s;
    }}

    if (output_file) {{
        FILE* f = fopen(output_file, "wb");
        if (f) {{
            fwrite(out, 4, {output_size}, f);
            fclose(f);
        }}
    }}

    double sum=0, mn=t[0], mx=t[0];
    for (int i = 0; i < runs; i++) {{ sum += t[i]; if (t[i]<mn) mn=t[i]; if (t[i]>mx) mx=t[i]; }}
    double avg = sum/runs;
    double var = 0;
    for (int i = 0; i < runs; i++) var += (t[i]-avg)*(t[i]-avg);
    double std = sqrt(var/runs);

    if (json) {{
        printf("{{\\n");
        printf("  \\"runtime\\": \\"Sense Standalone\\",\\n");
        printf("  \\"runs\\": %d,\\n", runs);
        printf("  \\"avg_ms\\": %.6f,\\n", avg);
        printf("  \\"std_ms\\": %.6f,\\n", std);
        printf("  \\"min_ms\\": %.6f,\\n", mn);
        printf("  \\"max_ms\\": %.6f,\\n", mx);
        printf("  \\"throughput\\": %.2f,\\n", 1000.0/avg);
        printf("  \\"output_sample\\": [");
        for (int i = 0; i < 20 && i < {output_size}; i++) {{ printf("%.8f", out[i]); if (i<19 && i<{output_size}-1) printf(", "); }}
        printf("],\\n");
        printf("  \\"success\\": true\\n");
        printf("}}\\n");
    }} else {{
        printf("\\nSense Standalone Benchmark\\n");
        printf("Runs: %d, Avg: %.4f ms, Std: %.4f ms\\n", runs, avg, std);
        printf("Min/Max: %.4f / %.4f ms, Throughput: %.2f inf/s\\n", mn, mx, 1000.0/avg);
        printf("Output: "); for (int i=0;i<10 && i<{output_size};i++) printf("%.6f ",out[i]); printf("\\n");
    }}
    free(in); free(out); free(t);
}}

int main(int argc, char** argv) {{
    int runs=100, json=0;
    const char* input_file = NULL;
    const char* output_file = NULL;
    for (int i=1;i<argc;i++) {{
        if (!strcmp(argv[i],"--runs")&&i+1<argc) runs=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--json")) json=1;
        else if (!strcmp(argv[i],"--input")&&i+1<argc) input_file=argv[++i];
        else if (!strcmp(argv[i],"--output")&&i+1<argc) output_file=argv[++i];
    }}
    if (!json) printf("Sense Standalone Model ({len(self.operations)} ops)\\n");
    if (tvm_workspace_init()) {{ fprintf(stderr,"Init fail\\n"); return 1; }}
    g_pool = (float*)malloc(POOL_SIZE);
    if (!g_pool) {{ fprintf(stderr,"Pool fail\\n"); return 1; }}
    if (load_weights_all()) {{ fprintf(stderr,"Weight fail\\n"); return 1; }}
    benchmark(runs, json, input_file, output_file);
    free_weights_all();
    free(g_pool);
    tvm_workspace_cleanup();
    if (!json) printf("Done!\\n");
    return 0;
}}
'''

        return code


def generate_standalone_c(ir_path: Path, weights_dir: Path, output_path: Path,
                          model_name: str = "sense_model",
                          input_shape: Tuple = (1, 128, 192, 1),
                          output_shape: Tuple = (1, 863)) -> bool:
    """Generate standalone C code from IR file.

    Parameters
    ----------
    ir_path : Path
        Path to Relax IR file.
    weights_dir : Path
        Directory containing weight files.
    output_path : Path
        Output path for generated C code.
    model_name : str
        Model name.
    input_shape : Tuple
        Input tensor shape.
    output_shape : Tuple
        Output tensor shape.

    Returns
    -------
    bool
        True if successful.
    """
    try:
        # Read IR file
        with open(ir_path, 'r') as f:
            ir_content = f.read()

        # Get weight files
        weight_files = sorted([f for f in os.listdir(weights_dir) if f.endswith('.bin')])

        # Generate C code
        codegen = SenseCodeGenerator(ir_content, weights_dir, weight_files)
        codegen.parse_ir()
        c_code = codegen.generate_c_code(model_name, input_shape, output_shape)

        # Write output
        with open(output_path, 'w') as f:
            f.write(c_code)

        print(f"Generated C code: {output_path}")
        print(f"  Operations: {len(codegen.operations)}")
        print(f"  Weights: {len(weight_files)}")
        print(f"  Size: {len(c_code)} bytes")

        return True

    except Exception as e:
        print(f"Error generating C code: {e}")
        import traceback
        traceback.print_exc()
        return False

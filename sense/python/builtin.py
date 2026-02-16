from typing import Tuple, List


def default_header() -> str:
    header = '''#include <stdio.h>
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
'''
    return header

def generate_weights_runtime(weight_size: int, weight_count: int) -> str:
    code = f'''
#define UNIFIED_WEIGHTS_SIZE {weight_size}
#define UNIFIED_WEIGHTS_COUNT {weight_count}

static float __attribute__((aligned(64))) g_weights[UNIFIED_WEIGHTS_COUNT];

static int load_weights(const char* path) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{
        fprintf(stderr, "Failed to open: %s\\n", path);
        return -1;
    }}

    size_t read = fread(g_weights, sizeof(float), UNIFIED_WEIGHTS_COUNT, f);
    fclose(f);

    if (read != UNIFIED_WEIGHTS_COUNT) {{
        fprintf(stderr, "Weight size mismatch: expected %d, got %zu\\n", UNIFIED_WEIGHTS_COUNT, read);
        return -1;
    }}

    return 0;
}}
'''
    return code


def generate_storage_buffers(storage_sizes: list) -> str:
    total_storage = sum(storage_sizes)
    total_mb = total_storage / (1024 * 1024)
    num_buffers = len([s for s in storage_sizes if s > 0])

    code = f'''
/*========================================
 * Storage Buffers ({num_buffers} buffers, {total_mb:.2f} MB total)
 *========================================*/
'''

    # Declare static arrays
    for storage_id, size in enumerate(storage_sizes):
        if size > 0:
            size_floats = (size + 3) // 4
            size_mb = size / (1024 * 1024)
            code += f'static float __attribute__((aligned(64))) g_storage_{storage_id}[{size_floats}];  /* {size_mb:.2f} MB */\n'

    return code


def generate_helpers() -> str:
    code = '''
static void init_tensor(DLTensor* t, float* data, int ndim, int64_t* shape) {
    memset(t, 0, sizeof(DLTensor));
    t->data = data;
    t->device.device_type = kDLCPU;
    t->ndim = ndim;
    t->dtype.code = 2;
    t->dtype.bits = 32;
    t->dtype.lanes = 1;
    t->shape = shape;
}

static TVMFFIAny make_arg(DLTensor* t) {
    TVMFFIAny a;
    a.type_index = 0;
    a.v_ptr = t;
    return a;
}

#ifdef DEBUG_INTERMEDIATE
static void dump_tensor(const char* name, DLTensor* t, int max_elems) {
    float* data = (float*)t->data;
    int64_t total = 1;
    for (int i = 0; i < t->ndim; i++) {
        total *= t->shape[i];
    }
    int to_print = (total < max_elems) ? total : max_elems;

    printf("[%s] shape=(", name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld%s", t->shape[i], i < t->ndim - 1 ? "," : "");
    }
    printf(") values=[");
    for (int i = 0; i < to_print; i++) {
        printf("%.6f%s", data[i], i < to_print - 1 ? "," : "");
    }
    printf("%s]\\n", to_print < total ? "..." : "");
    fflush(stdout);
}
#endif
'''
    return code


def generate_inference_skeleton(model_name: str,
                                  input_shape: tuple,
                                  output_shape: tuple,
                                  num_storages: int,
                                  total_storage_mb: float,
                                  num_tensors: int,
                                  num_operations: int) -> str:
    code = f'''
int {model_name}_inference(float* input, float* output) {{
    /* Input: {input_shape}, Output: {output_shape} */
    /* Storages: {num_storages}, Total: {total_storage_mb:.2f} MB */
    /* Tensors: {num_tensors}, Operations: {num_operations} */

    /* TODO: Tensor allocations and operations */

    return 0;
}}
'''
    return code


def generate_reshape(output_var: str, input_var: str, new_shape: Tuple[int, ...]) -> str:
    """
    Generate C code for vm.builtin.reshape

    TVM implementation: data.CreateView(new_shape, data->dtype)
    C equivalent: Create new DLTensor with same data pointer, different shape
    """
    ndim = len(new_shape)
    shape_str = ', '.join(str(s) for s in new_shape)

    code = f"    /* reshape: {input_var} -> {output_var} */\n"
    code += f"    int64_t {output_var}_shape_data[4] = {{{shape_str}}};\n"
    code += f"    init_tensor(&{output_var}_tensor, (float*){input_var}_tensor.data, {ndim}, {output_var}_shape_data);\n"

    return code


def generate_match_shape(input_var: str, expected_shape: Tuple[int, ...]) -> str:
    """
    Generate C code for vm.builtin.match_shape

    Validates tensor shape (usually skipped in optimized code)
    """
    return f"    /* match_shape validation skipped for {input_var} */\n"


def generate_check_tensor_info(input_var: str, expected_ndim: int, expected_dtype: str) -> str:
    """
    Generate C code for vm.builtin.check_tensor_info

    Validates tensor dtype (usually skipped in optimized code)
    """
    return f"    /* check_tensor_info validation skipped for {input_var} */\n"


def generate_main(model_name: str, input_size: int, output_size: int) -> str:
    """
    Generate main function for standalone execution
    """
    code = f'''
/*========================================
 * Main Function
 *========================================*/
int main(int argc, char** argv) {{
    if (argc != 3) {{
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\\n", argv[0]);
        return 1;
    }}

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    /* Load weights */
    printf("Loading weights...\\n");
    if (load_weights("lib/weights.bin") != 0) {{
        fprintf(stderr, "Failed to load weights\\n");
        return 1;
    }}
    printf("Weights loaded successfully\\n");

    /* Allocate buffers */
    float* input = (float*)malloc({input_size} * sizeof(float));
    float* output = (float*)malloc({output_size} * sizeof(float));

    if (!input || !output) {{
        fprintf(stderr, "Failed to allocate memory\\n");
        free(input);
        free(output);
        return 1;
    }}

    /* Read input */
    FILE* f_in = fopen(input_path, "rb");
    if (!f_in) {{
        fprintf(stderr, "Failed to open input file: %s\\n", input_path);
        free(input);
        free(output);
        return 1;
    }}

    size_t read_count = fread(input, sizeof(float), {input_size}, f_in);
    fclose(f_in);

    if (read_count != {input_size}) {{
        fprintf(stderr, "Expected to read {input_size} floats, got %zu\\n", read_count);
        free(input);
        free(output);
        return 1;
    }}

    printf("Input loaded: {input_size} floats\\n");

    /* Initialize TVM workspace */
    if (tvm_workspace_init() != 0) {{
        fprintf(stderr, "Failed to initialize TVM workspace\\n");
        free(input);
        free(output);
        return 1;
    }}

    /* Run inference */
    printf("Running inference...\\n");
    clock_t start = clock();

    int ret = {model_name}_inference(input, output);

    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    if (ret != 0) {{
        fprintf(stderr, "Inference failed with code: %d\\n", ret);
        tvm_workspace_cleanup();
        free(input);
        free(output);
        return 1;
    }}

    printf("Inference completed in %.2f ms\\n", elapsed);

    /* Write output */
    FILE* f_out = fopen(output_path, "wb");
    if (!f_out) {{
        fprintf(stderr, "Failed to open output file: %s\\n", output_path);
        tvm_workspace_cleanup();
        free(input);
        free(output);
        return 1;
    }}

    fwrite(output, sizeof(float), {output_size}, f_out);
    fclose(f_out);

    printf("Output saved: {output_size} floats\\n");

    /* Print first few outputs for sanity check */
    printf("First 10 outputs: ");
    for (int i = 0; i < 10 && i < {output_size}; i++) {{
        printf("%.6f ", output[i]);
    }}
    printf("\\n");

    /* Cleanup */
    tvm_workspace_cleanup();
    free(input);
    free(output);

    return 0;
}}
'''
    return code


BUILTIN_CODEGEN = {
    'header': default_header,
    'weights': generate_weights_runtime,
    'storage_buffers': generate_storage_buffers,
    'helpers': generate_helpers,
    'main': generate_main,
    'reshape': generate_reshape,
    'match_shape': generate_match_shape,
    'check_tensor_info': generate_check_tensor_info,
}

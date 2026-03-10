# SPDX-License-Identifier: Apache-2.0
"""MCU source code generation: C code templates and builders."""

from typing import Dict


def generate_mcu_header() -> str:
    """Generate MCU-optimized header (no workspace - it's in lib0.c)

    Returns
    -------
        str : C header code
    """
    return '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

/* Workspace management functions from lib0.c */
extern int tvm_workspace_init(void);
extern void tvm_workspace_cleanup(void);
extern void tvm_workspace_reset(void);

'''


def generate_lib0_workspace_patch(workspace_size: int) -> str:
    """Generate workspace implementation to be inserted into lib0.c

    Parameters
    ----------
        workspace_size : Workspace size in bytes

    Returns
    -------
        str : C code for workspace management
    """
    workspace_mb = workspace_size / (1024 * 1024)

    return f'''
/* Workspace management - static allocation */
#define WORKSPACE_SIZE {workspace_size}  /* {workspace_mb:.2f} MB */
static char g_workspace[WORKSPACE_SIZE] __attribute__((aligned(64)));
static size_t g_workspace_offset = 0;

TVM_DLL void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                        uint64_t nbytes, int dtype_code,
                                        int dtype_bits) {{
    size_t aligned = (nbytes + 63) & ~63ULL;  /* 64-byte alignment */
    if (g_workspace_offset + aligned > WORKSPACE_SIZE) {{
        return NULL;
    }}
    void* ptr = g_workspace + g_workspace_offset;
    g_workspace_offset += aligned;
    return ptr;
}}

TVM_DLL int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {{
    g_workspace_offset = 0;
    return 0;
}}

int tvm_workspace_init(void) {{
    g_workspace_offset = 0;
    return 0;
}}

void tvm_workspace_cleanup(void) {{
    g_workspace_offset = 0;
}}

void tvm_workspace_reset(void) {{
    memset(g_workspace, 0, WORKSPACE_SIZE);
    g_workspace_offset = 0;
}}

'''


def generate_weights_section(total_size_bytes: int, total_floats: int) -> str:
    """Generate weights storage and loading function

    Parameters
    ----------
        total_size_bytes : Total weight size in bytes
        total_floats : Total number of floats

    Returns
    -------
        str : C code for weights
    """
    total_mb = total_size_bytes / (1024 * 1024)

    return f'''
/*========================================
 * Weights - Static Allocation ({total_mb:.2f} MB)
 *========================================*/
#define UNIFIED_WEIGHTS_SIZE {total_size_bytes}
#define UNIFIED_WEIGHTS_COUNT {total_floats}

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


def generate_storage_buffers_section(storage_sizes: list) -> str:
    """Generate static storage buffers

    Parameters
    ----------
        storage_sizes : List of storage sizes in bytes

    Returns
    -------
        str : C code for storage buffers
    """
    total_storage = sum(storage_sizes)
    num_buffers = len([s for s in storage_sizes if s > 0])
    total_mb = total_storage / (1024 * 1024)

    code = f'''
/*========================================
 * Storage Buffers - Static Allocation ({num_buffers} buffers, {total_mb:.2f} MB total)
 *========================================*/
'''

    for storage_id, size in enumerate(storage_sizes):
        if size > 0:
            size_floats = (size + 3) // 4
            size_mb = size / (1024 * 1024)
            code += f'static float __attribute__((aligned(64))) g_storage_{storage_id}[{size_floats}];  /* {size_mb:.2f} MB */\n'

    code += '\n'
    return code


def generate_debug_helper() -> str:
    """Generate debug helper function for tensor dumping

    Returns
    -------
        str : C code for debug helper
    """
    return '''
#ifdef DEBUG_INTERMEDIATE
static void dump_tensor_data(const char* name, float* data, int size, int max_elems) {
    int to_print = (size < max_elems) ? size : max_elems;
    printf("[%s] values=[", name);
    for (int i = 0; i < to_print; i++) {
        printf("%.6f%s", data[i], i < to_print - 1 ? "," : "");
    }
    printf("%s]\\n", to_print < size ? "..." : "");
    fflush(stdout);
}
#endif

'''


def generate_main_function(model_name: str, input_size: int, output_size: int) -> str:
    """Generate main function with goto-based error handling

    Parameters
    ----------
        model_name : Model name for inference function
        input_size : Total input elements
        output_size : Total output elements

    Returns
    -------
        str : C code for main function
    """
    return f'''
/*========================================
 * Main Function
 *========================================*/
int main(int argc, char** argv) {{
    int ret = 0;
    float *input = NULL, *output = NULL;
    FILE *f_in = NULL, *f_out = NULL;
    size_t read_count = 0;
    clock_t start = 0;
    clock_t end = 0;
    double elapsed = 0.0;
    int inference_ret = 0;

    if (argc < 3) {{
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> [unused]\\n", argv[0]);
        return 1;
    }}

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    /* Load weights */
    printf("Loading weights...\\n");
    if (load_weights("weights.bin") != 0) {{
        fprintf(stderr, "Failed to load weights\\n");
        ret = 1;
        goto cleanup;
    }}
    printf("Weights loaded successfully\\n");

    /* Allocate buffers */
    input = (float*)malloc({input_size} * sizeof(float));
    output = (float*)malloc({output_size} * sizeof(float));

    if (!input || !output) {{
        fprintf(stderr, "Failed to allocate memory\\n");
        ret = 1;
        goto cleanup;
    }}

    /* Read input */
    f_in = fopen(input_path, "rb");
    if (!f_in) {{
        fprintf(stderr, "Failed to open input file: %s\\n", input_path);
        ret = 1;
        goto cleanup;
    }}

    read_count = fread(input, sizeof(float), {input_size}, f_in);
    fclose(f_in);
    f_in = NULL;

    if (read_count != {input_size}) {{
        fprintf(stderr, "Expected to read {input_size} floats, got %zu\\n", read_count);
        ret = 1;
        goto cleanup;
    }}

    printf("Input loaded: {input_size} floats\\n");

    /* Initialize workspace */
    if (tvm_workspace_init() != 0) {{
        fprintf(stderr, "Failed to initialize workspace\\n");
        ret = 1;
        goto cleanup;
    }}

    /* Run inference */
    printf("Running inference...\\n");
    start = clock();
    inference_ret = {model_name}_inference(input, output);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    if (inference_ret != 0) {{
        fprintf(stderr, "Inference failed with code: %d\\n", inference_ret);
        ret = 1;
        goto cleanup;
    }}

    printf("Inference completed in %.2f ms\\n", elapsed);

    /* Write output */
    f_out = fopen(output_path, "wb");
    if (!f_out) {{
        fprintf(stderr, "Failed to open output file: %s\\n", output_path);
        ret = 1;
        goto cleanup;
    }}

    fwrite(output, sizeof(float), {output_size}, f_out);
    fclose(f_out);
    f_out = NULL;

    printf("Output saved: {output_size} floats\\n");

    /* Print first few outputs */
    printf("First 10 outputs: ");
    for (int i = 0; i < 10 && i < {output_size}; i++) {{
        printf("%.6f ", output[i]);
    }}
    printf("\\n");

cleanup:
    if (f_in) fclose(f_in);
    if (f_out) fclose(f_out);
    if (input) free(input);
    if (output) free(output);
    tvm_workspace_cleanup();

    return ret;
}}
'''


def generate_function_declarations(func_declarations: Dict[str, str], op_names: list) -> str:
    """Generate forward declarations for lib0.c functions

    Parameters
    ----------
        func_declarations : Dictionary of function declarations from lib0.c
        op_names : List of operation names

    Returns
    -------
        str : C code for function declarations
    """
    code = '/* Forward declarations from lib0.c - direct function calls */\n'
    code += '#ifdef __cplusplus\nextern "C" {\n#endif\n'
    for op in op_names:
        if op == 'reshape':
            # Reshape functions handled differently
            pass
        elif op in func_declarations:
            # Use actual signature from lib0.c
            code += func_declarations[op] + '\n'
        else:
            # Fallback: generic 3-arg signature
            code += f'extern int32_t {op}(float* input1, float* input2, float* output);\n'
    code += '#ifdef __cplusplus\n}\n#endif\n\n'
    return code

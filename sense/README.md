# SENSE

SENSE is the user-facing interface layer for the SENSE-TVM pipeline. It loads a model configuration, translates the input model, runs optimization and compilation, and emits deployable output artifacts for the selected target.

## Architecture

<img src="../readme/sense.png" alt="SENSE Architecture" width="360" />

The SENSE flow is centered around the `Sense` class in [`sense.py`](/Users/gyujinkim/Desktop/cochl/sense-tvm/sense/sense.py). The pipeline is executed in a fixed order:

1. `translate`: Load the input model and convert it into TVM Relax IR with input, output, and weight metadata.
2. `optimize_graph`: Select and run Relax passes based on the configured backend, hardware, and optimization level.
3. `compile`: Build the optimized module through TVM `relax.build` and the registered TIR pipeline.
4. `export`: Save intermediate artifacts such as TIR text, weight metadata, and exported library files.
5. `codegen`: Generate backend-specific standalone sources, metadata, and build files.

In practice, `sense/main.py` loads `settings/*.json`, validates the configuration, runs `Sense.execute()`, and optionally compares the generated result against ONNX Runtime when `--validate` is enabled.

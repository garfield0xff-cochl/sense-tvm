import time
import json
import tempfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict
import tvm
from tvm import relax


class Sense:
    def __init__(self, config: Dict = None):
        """The class definition support Sense Multi AI Backend

        Parameters
        ----------
            config : 
                common    : output path, target hardware, optimization level
                optimizer : operation fusion,
                export    : ir graph, metadata
        """
        self.config = config or {}
        self.model_path = None
        self.ir_mod = None
        self.input_info = {}
        self.output_info = {}
        self.weights = {}
        self.weight_order = []
        self.compiled_module = None

        output_dir = self.config.get("output_dir", "./bin")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"  TVM: {tvm.__version__}")
        print(f"  Output: {output_dir}")

    def parse(self, model_path: str):
        """Load ONNX model and convert to TVM Relax IR with metadata

        Extracts:
            ir_mod        : TVM Relax IR module (converted from ONNX)
            input_info    : Input tensor metadata (name, shape, dtype)
            output_info   : Output tensor metadata (name, shape, dtype)
            weights       : Model weights as numpy arrays {name: ndarray}
            weight_order  : Weight names in TVM Constant usage order (for matching)
        """
        print(f"\n[Parse]")

        from .parser import parse

        self.model_path = model_path
        self.ir_mod, self.input_info, self.output_info, self.weights, self.weight_order = parse(model_path)

        return self

    def ir_pass(self):
        """Apply TVM optimization passes to Relax IR
        """
        print(f"\n[IR Pass]")
        t_start = time.perf_counter()

        # Default build pipeline passes (from python/tvm/relax/pipeline.py:default_build_pipeline)
        from tvm.relax import backend
        passes = [
            backend.DispatchSampling(),           # Dispatch sampling ops to target-specific implementations
            backend.DispatchSortScan(),           # Dispatch sort/scan ops to target-specific implementations
            relax.transform.LegalizeOps(),        # Lower high-level Relax ops to executable lower-level ops
            relax.transform.RewriteDataflowReshape(),  # Rewrite reshape ops in dataflow blocks for optimization
            relax.transform.ToNonDataflow(),      # Convert dataflow blocks to normal bindings (simplify control flow)
            relax.transform.RemovePurityChecking(),    # Remove purity checking annotations (not needed at runtime)
            relax.transform.CallTIRRewrite(),     # Rewrite call_tir ops into VM-executable form
            relax.transform.StaticPlanBlockMemory(),   # Plan static memory allocation at compile time
            relax.transform.RewriteCUDAGraph(),   # Rewrite code for CUDA graph optimization
            relax.transform.LowerAllocTensor(),   # Lower alloc_tensor ops to VM memory allocation instructions
            relax.transform.KillAfterLastUse(),   # Insert memory deallocation after last use (memory optimization)
            relax.transform.LowerRuntimeBuiltin(),     # Lower to VM runtime builtin function calls
            relax.transform.ComputePrimValue(),   # Compute PrimValue type calculations to concrete values
            relax.transform.VMShapeLower(),       # Lower dynamic shape computations to VM-executable form
            relax.transform.AttachGlobalSymbol(), # Attach global symbol attributes (enable runtime function lookup)
        ]

        opt_level = self.config.get("opt_level", 3)
        with tvm.transform.PassContext(opt_level=opt_level):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        return self

    def compile(self, target: str = "c"):
        """Compile optimized IR to target backend code

        Parameters
        ----------
            target : Compilation target ("c", "llvm", "cuda", etc.)
        """
        print(f"\n[Compile]")
        t_start = time.perf_counter()

        target_str = target or self.config.get("target", "c")

        if target_str == "c":
            tvm_target = tvm.target.Target("c -keys=cpu")
        else:
            tvm_target = tvm.target.Target(target_str)

        print(f"  Target: {tvm_target}")

        opt_level = self.config.get("opt_level", 3)
        with tvm.transform.PassContext(opt_level=opt_level):
            self.compiled_module = relax.build(self.ir_mod, target=tvm_target)

        return self

    def export(self, name: str = "sense_model"):
        """Export compiled module and IR to files
        """
        print(f"\n[Export]")
        t_start = time.perf_counter()

        output_dir = Path(self.config.get("output_dir", "./bin"))
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok= True)

        if self.config.get("save_ir", True):
            ir_path =  metadata_dir / f"{name}_ir.txt"
            with open(ir_path, 'w') as f:
                f.write(self.ir_mod.script(show_meta=True))
            print(f"  IR: {ir_path}")

        if self.compiled_module:
            tar_path = output_dir / f"{name}.tar"
            self.compiled_module.export_library(str(tar_path))
            print(f"  TAR: {tar_path}")

            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(tmpdir)

                tmpdir_path = Path(tmpdir)
                c_files = list(tmpdir_path.rglob("*.c"))

                if c_files:
                    lib_dir = output_dir / "lib"
                    lib_dir.mkdir(parents=True, exist_ok=True)

                    for c_file in c_files:
                        if c_file.name.startswith('._'):
                            continue
                        dest = lib_dir / c_file.name
                        shutil.copy(c_file, dest)
                        print(f"    - {c_file.name} ({dest.stat().st_size / 1024:.2f} KB)")

        print(f"  Time: {time.perf_counter() - t_start:.2f}s")
        return self

    def codegen(self, name: str = "sense_model"):
        """Generate standalone C code without VM dependency

        Output:
            - lib/{name}.c: Standalone executable C code
            - lib/weights.bin: Unified weight binary
            - metadata/weights.json: Weight offset map

        Parameters
        ----------
            name : Model name for generated files
        """
        from .codegen import main_entry_codegen

        output_dir = Path(self.config.get("output_dir", "./bin"))
        save_metadata = self.config.get("save_metadata", True)

        input_name = list(self.input_info.keys())[0]
        input_shape = self.input_info[input_name]["shape"]
        output_name = list(self.output_info.keys())[0]
        output_shape = self.output_info[output_name]["shape"]

        main_entry_codegen(
            self.ir_mod,
            input_shape,
            output_shape,
            self.weights,
            self.weight_order,
            output_dir,
            name,
            save_metadata
        )

        return self

    def execute(self, model_path: str, target: str = "c", name: str = "sense_model"):
        """Run complete compilation pipeline

        Pipeline:
            parse → ir_pass → compile → export → codegen

        Parameters
        ----------
            model_path : Path to ONNX model
            target : Compilation target (default: "c")
            name : Output model name        
        """
        return (
            self.parse(model_path)
                .ir_pass()
                .compile(target=target)
                .export(name=name)
                .codegen(name=name)
        )

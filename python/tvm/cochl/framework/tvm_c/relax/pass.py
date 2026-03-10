"""tvm_c-specific Relax optimization passes."""

import tvm
from tvm import relax, tir

from .transform import FusePadConv, FuseRelu6

def get_sense_main_passes():
    """Get Relax optimization passes for tvm_c targets."""
    return [
        relax.transform.RemoveUnusedOutputs(),          # Remove outputs that are not used by the main function.
        relax.transform.EliminateCommonSubexpr(),       # Eliminate common subexpressions to reduce duplicated computation.
        relax.transform.DecomposeOpsForInference(),     # Decompose high-level ops into inference-friendly forms.
        relax.transform.FoldConstant(),                 # Fold constant expressions to reduce runtime work.
        FusePadConv(),                                  # Fuse pad -> conv2d where padding can be absorbed into conv attrs.
        FuseRelu6(),                                    # Fuse max/min clip pattern into a single clip op.
        relax.backend.DispatchSampling(),               # Dispatch sampling ops to target-specific implementations.
        relax.backend.DispatchSortScan(),               # Dispatch sort/scan ops to target-specific implementations.
        relax.transform.LegalizeOps(),                  # Legalize high-level Relax ops to lower-level forms.
        relax.transform.RewriteDataflowReshape(),       # Rewrite reshape ops in dataflow blocks for optimization.
        relax.transform.ToNonDataflow(),                # Convert dataflow blocks to normal bindings (simplify control flow).
        relax.transform.RemovePurityChecking(),         # Remove purity checking annotations (not needed at runtime).
        relax.transform.CallTIRRewrite(),               # Rewrite call_tir ops into VM-executable form.
        relax.transform.StaticPlanBlockMemory(),        # Plan static memory allocation at compile time.
        relax.transform.RewriteCUDAGraph(),             # Rewrite code for CUDA graph optimization (no-op on CPU).
        relax.transform.LowerAllocTensor(),             # Lower alloc_tensor ops to VM memory allocation instructions.
        relax.transform.KillAfterLastUse(),             # Insert memory deallocation after last use.
        relax.transform.LowerRuntimeBuiltin(),          # Lower to VM runtime builtin function calls.
        relax.transform.ComputePrimValue(),             # Compute PrimValue calculations to concrete values.
        relax.transform.VMShapeLower(),                 # Lower dynamic shape computations to VM-executable form.
        relax.transform.AttachGlobalSymbol(),           # Attach global symbol attributes for runtime lookup.
        relax.transform.RemoveUnusedOutputs(),          # Final cleanup of unused outputs after VM lowering.
    ]


def get_unpacked_passes():
    """Get TIR pipeline that splits host/device and uses unpacked API."""

    @tvm.transform.module_pass(opt_level=0, name="unpacked_tir_pipeline")
    def unpacked_tir_pipeline(mod: tvm.IRModule, ctx: tvm.transform.PassContext):
        """Custom TIR pipeline using MakeUnpackedAPI for MCU targets."""
        _ = ctx
        passes = [
            tir.transform.CanonicalizeLoop(),
            tir.transform.LowerInitBlock(),
            tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tir.transform.ConvertBlocksToOpaque(),
            tir.transform.CompactBufferAllocation(),
            tir.transform.LowerMatchBuffer(),
            tir.transform.Simplify(),
            tir.transform.LowerOpaqueBlock(),
            tir.transform.FlattenBuffer(),
            tir.transform.NarrowDataType(32),
            tir.transform.Simplify(),
            tir.transform.RemoveNoOp(),
            tir.transform.AnnotateDeviceRegions(),
            tir.transform.SplitHostDevice(),
            tir.transform.MakeUnpackedAPI(),  # Unpacked API for MCU
            tir.transform.LowerDeviceKernelLaunch(),
        ]
        return tvm.ir.transform.Sequential(passes)(mod)

    return unpacked_tir_pipeline

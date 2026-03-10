"""ncnn-specific Relax optimization passes."""

from tvm import relax


def get_sense_main_passes():
    """Get Relax optimization passes for ncnn targets.

    Keep ops at a relatively high level suitable for ncnn translation.
    Avoid legalization/VM-lowering passes here; codegen/bridge should handle that separately.
    """
    return [
        relax.transform.RemoveUnusedOutputs(),          # Drop unused outputs to simplify the graph.
        relax.transform.EliminateCommonSubexpr(),       # Remove duplicate subexpressions.
        relax.transform.FoldConstant(),                 # Fold constants early.
        relax.transform.RewriteDataflowReshape(),       # Canonicalize reshape patterns.
        relax.transform.ToNonDataflow(),                # Simplify dataflow blocks.
        relax.transform.RemovePurityChecking(),         # Remove purity checks not needed for codegen.
        relax.transform.FoldConstant(),                 # Fold constants again after cleanup.
        relax.transform.RemoveUnusedOutputs(),          # Final cleanup.
    ]

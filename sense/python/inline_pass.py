# SPDX-License-Identifier: Apache-2.0
"""
InlineOps Transform Pass

Custom TVM module_pass for inlining simple operations.
Follows TVM transform API convention.

Usage:
    from sense.python.inline_pass import InlineOps
    ir_mod = InlineOps()(ir_mod)
"""

import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class SimpleOpsFuser(PyExprMutator):
    """Fuse simple elementwise operations.

    This mutator traverses Relax IR and fuses patterns like:
    - add + maximum → add_relu
    - maximum + minimum → relu6
    """

    def __init__(self, mod):
        super().__init__(mod)
        self.mod_ = mod

    def visit_var_binding_(self, binding):
        """Visit each binding and potentially fuse operations."""
        # Default behavior: just visit recursively
        return super().visit_var_binding_(binding)


def InlineOps() -> tvm.transform.Pass:
    """Inline simple elementwise operations.

    Returns a proper TVM module_pass that can be used in pipeline.

    Returns
    -------
    fpass : tvm.transform.Pass
        The transformation pass.

    Example
    -------
    .. code-block:: python

        from sense.python.inline_pass import InlineOps

        # Use in a pipeline
        passes = [
            relax.transform.FuseOps(),
            InlineOps(),  # Custom pass
            relax.transform.FoldConstant(),
        ]
        ir_mod = tvm.transform.Sequential(passes)(ir_mod)
    """

    @tvm.transform.module_pass(opt_level=0, name="InlineOps")
    def _transform(mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """Transform function that modifies IRModule."""

        print(f"      [InlineOps] Processing {len(mod.functions)} functions...")

        # Apply fusion using mutator
        fuser = SimpleOpsFuser(mod)

        # Transform each Relax function
        new_functions = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function):
                # Apply fusion to this function
                new_func = fuser.visit_expr(func)
                new_functions[gvar] = new_func
            else:
                # Keep TIR functions unchanged
                new_functions[gvar] = func

        # Return new IRModule
        new_mod = tvm.IRModule(new_functions, mod.attrs, mod.global_infos)

        print(f"      [InlineOps] Completed (placeholder - no actual fusion yet)")

        return new_mod

    return _transform

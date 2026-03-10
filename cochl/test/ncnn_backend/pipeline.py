# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import tvm

from tvm.cochl.core.translate import translate_onnx


def build_ir_mod(onnx_path, *, apply_mem_passes: bool = True):
    ir_mod, _, _, _, _ = translate_onnx(onnx_path)
    relax_mod = importlib.import_module("tvm.cochl.framework.ncnn.relax.pass")
    relax_passes = relax_mod.get_sense_main_passes()
    if relax_passes:
        with tvm.transform.PassContext(opt_level=3):
            ir_mod = tvm.transform.Sequential(relax_passes)(ir_mod)

    if apply_mem_passes:
        mem_passes = [
            tvm.relax.transform.StaticPlanBlockMemory(),
            tvm.relax.transform.LowerAllocTensor(),
            tvm.relax.transform.KillAfterLastUse(),
            tvm.relax.transform.LowerRuntimeBuiltin(),
            tvm.relax.transform.ComputePrimValue(),
            tvm.relax.transform.VMShapeLower(),
            tvm.relax.transform.AttachGlobalSymbol(),
            tvm.relax.transform.RemoveUnusedOutputs(),
        ]
        with tvm.transform.PassContext(opt_level=3):
            ir_mod = tvm.transform.Sequential(mem_passes)(ir_mod)
    return ir_mod

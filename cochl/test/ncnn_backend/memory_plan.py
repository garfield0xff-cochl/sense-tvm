# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import tvm

from tvm.cochl.core.translate import translate_onnx
from tvm.cochl.framework.relax.standalone_packer import StandalonePacker
import re


def build_tvmc_ir_mod(onnx_path):
    ir_mod, *_ = translate_onnx(onnx_path)
    mod = importlib.import_module("tvm.cochl.framework.tvm_c.relax.pass")
    passes = mod.get_sense_main_passes()
    with tvm.transform.PassContext(opt_level=3):
        ir_mod = tvm.transform.Sequential(passes)(ir_mod)
    return ir_mod


def build_plan(onnx_path=None, *, ir_mod=None):
    if ir_mod is None:
        if onnx_path is None:
            raise ValueError("build_plan requires onnx_path or ir_mod")
        ir_mod = build_tvmc_ir_mod(onnx_path)
    text = ir_mod.script(show_meta=True)
    parser = StandalonePacker(text)
    parser.pack()
    # extract scalar for unary multiply if present
    scalar = None
    m = re.search(r"T_multiply[^\n]*= .*?\* T\.float32\(([^)]+)\)", text)
    if m:
        try:
            scalar = float(m.group(1))
        except ValueError:
            scalar = None
    parser.scalar_values = {"multiply": scalar} if scalar is not None else {}
    return parser

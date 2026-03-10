# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import tvm
from tvm import tir

from tvm.cochl.framework.ncnn.codegen.sources import NEON_SOURCES
from tvm.cochl.framework.ncnn.codegen.match import call_extern_symbol


def build_call_extern_module(entries: list[dict]) -> tvm.IRModule:
    f32_ptr = tvm.ir.PointerType(tvm.ir.PrimType("float32"))
    i32_ptr = tvm.ir.PointerType(tvm.ir.PrimType("int32"))
    a = tir.Var("a", f32_ptr)
    b = tir.Var("b", f32_ptr)
    c = tir.Var("c", f32_ptr)
    d = tir.Var("d", f32_ptr)
    e = tir.Var("e", f32_ptr)
    f = tir.Var("f", f32_ptr)
    ia = tir.Var("ia", i32_ptr)
    ib = tir.Var("ib", i32_ptr)

    buf_a = tir.decl_buffer((1,), "float32", data=a)
    buf_b = tir.decl_buffer((1,), "float32", data=b)
    buf_c = tir.decl_buffer((1,), "float32", data=c)
    buf_d = tir.decl_buffer((1,), "float32", data=d)
    buf_e = tir.decl_buffer((1,), "float32", data=e)
    buf_f = tir.decl_buffer((1,), "float32", data=f)
    buf_ia = tir.decl_buffer((1,), "int32", data=ia)
    buf_ib = tir.decl_buffer((1,), "int32", data=ib)

    stmts = []
    for entry in entries:
        symbol_info = call_extern_symbol(entry)
        if symbol_info is None:
            continue
        symbol, kind = symbol_info
        if kind == "conv":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        buf_c.data,
                        buf_d.data,
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                    )
                )
            )
        elif kind == "binary":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        buf_c.data,
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                    )
                )
            )
        elif kind == "binary_broadcast":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        buf_c.data,
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                    )
                )
            )
        elif kind == "sigmoid":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        tir.IntImm("int32", 1),
                    )
                )
            )
        elif kind == "matmul":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        buf_c.data,
                        buf_d.data,
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                    )
                )
            )
        elif kind == "squeeze":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        buf_ia.data,
                        tir.IntImm("int32", 1),
                        buf_ib.data,
                        tir.IntImm("int32", 1),
                    )
                )
            )
        elif kind == "permute":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        buf_ia.data,
                        tir.IntImm("int32", 1),
                        buf_ib.data,
                    )
                )
            )
        elif kind == "pad":
            pad_top = int(entry.get("attrs", {}).get("pad_top", 0))
            pad_left = int(entry.get("attrs", {}).get("pad_left", 0))
            pad_bottom = int(entry.get("attrs", {}).get("pad_bottom", 0))
            pad_right = int(entry.get("attrs", {}).get("pad_right", 0))
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", pad_top),
                        tir.IntImm("int32", pad_left),
                        tir.IntImm("int32", pad_bottom),
                        tir.IntImm("int32", pad_right),
                        tir.FloatImm("float32", 0.0),
                    )
                )
            )
        elif kind == "reduction_mean":
            stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "int32",
                        symbol,
                        buf_a.data,
                        buf_b.data,
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                        tir.IntImm("int32", 1),
                    )
                )
            )

    body = tir.SeqStmt(stmts) if stmts else tir.Evaluate(tir.IntImm("int32", 0))
    prim = tir.PrimFunc(
        [a, b, c, d, e, f, ia, ib],
        body,
        ret_type=None,
    ).with_attr("global_symbol", "main").with_attr("tir.noalias", True)

    return tvm.IRModule({"main": prim})


def collect_neon_sources(symbols: set[str]) -> list[Path]:
    sources = []
    seen = set()
    for sym in symbols:
        src = NEON_SOURCES.get(sym)
        if src and src.exists():
            if src not in seen:
                sources.append(src)
                seen.add(src)
    return sources


def write_lib0_with_impl(
    lib0_path: Path, sources: list[Path], wrappers: str, symbols: set[str]
) -> None:
    lib0_path.parent.mkdir(parents=True, exist_ok=True)
    call_extern_src = ""
    if lib0_path.exists():
        call_extern_src = lib0_path.read_text(encoding="utf-8", errors="ignore")
    parts = []
    parts.append("/* Generated lib0.c (call_extern + implementations) */\n")
    parts.append('#include "cochl/include/target/nchw/ncnn.h"\n\n')
    parts.append(wrappers)
    # Skip TVM call_extern C output for standalone validation build.
    emitted_mat = False
    emitted_option = False
    for src in sources:
        lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
        filtered_lines = []
        skip_mat = False
        skip_opt = False
        for line in lines:
            if not emitted_mat and line.strip() == "struct Mat":
                emitted_mat = True
                filtered_lines.append(line)
                continue
            if emitted_mat and line.strip() == "struct Mat":
                skip_mat = True
                continue
            if not emitted_option and line.strip() == "struct Option":
                emitted_option = True
                filtered_lines.append(line)
                continue
            if emitted_option and line.strip() == "struct Option":
                skip_opt = True
                continue
            if skip_mat:
                if line.strip() == "};":
                    skip_mat = False
                continue
            if skip_opt:
                if line.strip() == "};":
                    skip_opt = False
                continue
            filtered_lines.append(line)
        parts.append(f"/* ===== {src} ===== */\n")
        parts.append("\n".join(filtered_lines))
        parts.append("\n\n")
    text = "".join(parts)
    text = text.replace("TVM_DLL ", "")
    filtered = []
    for line in text.splitlines():
        if line.startswith('#include "tvm/') or line.startswith("#include <tvm/"):
            continue
        if line.startswith("int32_t "):
            skip = False
            for sym in symbols:
                if sym in line:
                    skip = True
                    break
            if skip:
                continue
        filtered.append(line)
    lib0_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

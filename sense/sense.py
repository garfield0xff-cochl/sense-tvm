from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import json
import tvm
from tvm import relax
import tarfile

try:
    from .config import SenseConfig, create_default_config
except ImportError:
    from config import SenseConfig, create_default_config

from tvm.cochl.core import translate
from tvm.cochl import registry


@dataclass
class ParseResult:
    ir_mod: Any
    input_info: Dict[str, Any]
    output_info: Dict[str, Any]
    weights: Dict[str, Any]
    weight_order: list[str]


class Sense:
    def __init__(self, config: Union[SenseConfig, Dict] = None):
        """Initialize Sense with a config object or dict."""

        # Handle both SenseConfig object and dict for backward compatibility
        if isinstance(config, SenseConfig):
            self.sense_config = config
            self.config = config.to_dict()
        elif isinstance(config, dict):
            self.sense_config = SenseConfig.from_dict(config)
            self.config = config
        else:
            self.sense_config = create_default_config()
            self.config = self.sense_config.to_dict()

        self.model_path = None
        self.parse_result: ParseResult | None = None
        self.compiled = None

    def translate(self, model_path: str) -> "Sense":
        """Translate model into IR and related metadata."""
        self.model_path = model_path
        self.parse_result = ParseResult(*translate.translate(model_path))
        self.ir_mod = self.parse_result.ir_mod
        return self

    def optimize_graph(self) -> "Sense":
        if self.ir_mod is None:
            return self
        passes = registry.get_relax_passes(
            backend=self.sense_config.build_option.backend,
            hardware=self.sense_config.hardware.device,
            custom_passes=self.sense_config.optimizer.custom_passes,
        )
        if not passes:
            return self
        opt_level = int(self.sense_config.build_option.opt_level)
        with tvm.transform.PassContext(opt_level=opt_level):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)
        self.parse_result = ParseResult(
            ir_mod=self.ir_mod,
            input_info=self.parse_result.input_info,
            output_info=self.parse_result.output_info,
            weights=self.parse_result.weights,
            weight_order=self.parse_result.weight_order,
        )
        return self

    def export(self) -> "Sense":
        """Export metadata based on configuration."""
        if self.sense_config.export.save_tir:
            self._save_ir_mod()
        if self.sense_config.export.save_weight_manifest:
            self._save_weight_manifest()
        self._export_library()
        return self

    def codegen(self) -> "Sense":
        if not self.parse_result or self.ir_mod is None:
            return self
        codegen = registry.get_codegen(self.sense_config.build_option.backend)
        output_dir = self._hardware_output_dir()
        input_info = self.parse_result.input_info
        output_info = self.parse_result.output_info
        if not input_info or not output_info:
            return self
        input_name, input_meta = next(iter(input_info.items()))
        output_name, output_meta = next(iter(output_info.items()))
        codegen(
            self.ir_mod,
            input_name=input_name,
            input_shape=input_meta.get("shape", ()),
            output_shape=output_meta.get("shape", ()),
            weights=self.parse_result.weights,
            weight_order=self.parse_result.weight_order,
            output_dir=output_dir,
            model_name=self.sense_config.export.model_name,
            save_metadata=self.sense_config.export.save_weight_manifest,
            model_path=self.model_path,
        )
        makefile_generator = registry.get_makefile_generator(
            self.sense_config.build_option.backend
        )
        makefile_generator(
            output_dir=output_dir,
            model_name=self.sense_config.export.model_name,
            debug={
                "trace_operation_delay": self.sense_config.debug.measure_execution_time,
                "dump_ir_tensor_data": self.sense_config.debug.dump_operator_outputs,
            },
        )
        return self

    def compile(self) -> "Sense":
        if self.ir_mod is None:
            return self
        tvm_target = tvm.target.Target("c")
        tir_pipeline = registry.get_tir_pipeline(self.sense_config.build_option.backend)
        if tir_pipeline is not None:
            self.compiled = relax.build(
                self.ir_mod,
                target=tvm_target,
                tir_pipeline=tir_pipeline,
            )
        else:
            self.compiled = relax.build(self.ir_mod, target=tvm_target)
        return self

    def _metadata_dir(self) -> Path:
        return self._hardware_output_dir() / "metadata"

    def _hardware_output_dir(self) -> Path:
        return (
            Path(self.sense_config.export.output_dir)
            / self.sense_config.hardware.device
        )

    def _save_ir_mod(self) -> None:
        if not self.parse_result:
            return
        meta_dir = self._metadata_dir()
        meta_dir.mkdir(parents=True, exist_ok=True)
        ir_path = meta_dir / "tir.txt"
        ir_mod = self.parse_result.ir_mod
        try:
            content = ir_mod.astext(show_meta_data=False)
        except Exception:
            content = str(ir_mod)
        ir_path.write_text(content, encoding="utf-8")

    def _save_weight_manifest(self) -> None:
        if not self.parse_result:
            return
        meta_dir = self._metadata_dir()
        meta_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = meta_dir / "weight_manifest.json"
        weights = self.parse_result.weights
        weight_order = self.parse_result.weight_order
        items = []
        for name in weight_order:
            weight = weights.get(name)
            if weight is None:
                continue
            shape = getattr(weight, "shape", None)
            dtype = getattr(weight, "dtype", None)
            size = getattr(weight, "size", None)
            items.append(
                {
                    "name": name,
                    "shape": list(shape) if shape is not None else None,
                    "dtype": str(dtype) if dtype is not None else None,
                    "size": int(size) if size is not None else None,
                }
            )
        payload = {
            "weight_order": list(weight_order),
            "weights": items,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _export_library(self) -> None:
        if self.compiled is None:
            return
        output_dir = self._hardware_output_dir()
        lib_dir = output_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        tar_path = lib_dir / "lib.tar"
        self.compiled.export_library(str(tar_path))
        lib0_path = lib_dir / "lib0.c"
        if lib0_path.exists():
            content = lib0_path.read_text(encoding="utf-8", errors="ignore")
            if "Workspace management - static allocation" in content:
                return
        with tarfile.open(tar_path, "r") as tar:
            member = tar.getmember("lib0.c") if "lib0.c" in tar.getnames() else None
            if member is None:
                return
            member.name = "lib0.c"
            tar.extract(member, path=lib_dir)


    def execute(self) -> "Sense":
        (
            self.translate(self.sense_config.model_path)
            .optimize_graph()
            .compile()
            .export()
            .codegen()
        )
        return self

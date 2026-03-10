"""Unified registry for pass/codegen selection by hardware/backend."""

from typing import Callable, Iterable, List, Optional

# Supported hardware and backend names.
SUPPORTED_HARDWARE = {"rpi2"}
SUPPORTED_BACKENDS = {"c", "ncnn"}

# Supported custom pass presets.
SUPPORTED_CUSTOM_PASS = {"sense_onnx_main", "sense_onnx_pre"}

# Hardware -> architecture mapping.
HARDWARE_ARCH = {
    "rpi2": "armv7l",
}

# Backend -> Relax pass getter for each hardware.
RELAX_PASS_REGISTRY_BY_HW = {
    "rpi2": {
        "c": ("tvm.cochl.framework.tvm_c.relax.pass", "get_sense_main_passes"),
        "ncnn": ("tvm.cochl.framework.ncnn.relax.pass", "get_sense_main_passes"),
    },
}

CODEGEN_REGISTRY = {
    "c": ("tvm.cochl.framework.tvm_c.codegen.codegen", "_codegen_impl"),
    "ncnn": ("tvm.cochl.framework.ncnn.codegen.codegen", "_codegen_impl"),
}
TIR_PIPELINE_REGISTRY = {
    "c": ("tvm.cochl.framework.tvm_c.relax.pass", "get_unpacked_passes"),
}
MAKEFILE_REGISTRY = {
    "c": ("tvm.cochl.framework.tvm_c.codegen.makefile", "generate_makefile"),
    "ncnn": ("tvm.cochl.framework.ncnn.codegen.makefile", "generate_makefile"),
}


def _import_registry_function(registry: dict, backend: str, kind: str):
    """Load a function from registry by backend."""
    entry = registry.get(backend)
    if entry is None:
        return None

    module_name, func_name = entry
    try:
        module = __import__(module_name, fromlist=[func_name])
        return getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load {kind} '{module_name}.{func_name}': {e}")


def get_architecture(hardware: str) -> str:
    """Resolve architecture string from hardware name."""
    if hardware not in HARDWARE_ARCH:
        supported = sorted(HARDWARE_ARCH.keys())
        raise ValueError(
            f"Unsupported hardware '{hardware}'. Supported: {', '.join(supported)}"
        )
    return HARDWARE_ARCH[hardware]


def _custom_pass_sense_onnx_main() -> list:
    return []


def _custom_pass_sense_onnx_pre() -> list:
    return []


CUSTOM_PASS_PRESETS: dict[str, Callable[[], list]] = {
    "sense_onnx_main": _custom_pass_sense_onnx_main,
    "sense_onnx_pre": _custom_pass_sense_onnx_pre,
}


def _normalize_custom_passes(custom_passes: Optional[Iterable[str]]) -> List[str]:
    if not custom_passes:
        return []
    if isinstance(custom_passes, str):
        return [custom_passes]
    return list(custom_passes)


def _resolve_custom_passes(custom_passes: Optional[Iterable[str]]) -> Optional[list]:
    names = _normalize_custom_passes(custom_passes)
    if not names:
        return None

    passes: list = []
    for name in names:
        if name not in SUPPORTED_CUSTOM_PASS:
            supported = sorted(SUPPORTED_CUSTOM_PASS)
            raise ValueError(
                f"Unsupported custom_pass '{name}'. Supported: {', '.join(supported)}"
            )
        preset = CUSTOM_PASS_PRESETS.get(name)
        if preset is None:
            continue
        passes.extend(preset())
    return passes


def get_relax_passes(
    backend: str,
    hardware: Optional[str] = None,
    custom_passes: Optional[Iterable[str]] = None,
) -> Optional[list]:
    """Get Relax optimization passes for specified target backend.

    Priority:
    1) custom_passes (if provided)
    2) hardware-specific backend passes
    3) None
    """
    custom = _resolve_custom_passes(custom_passes)
    if custom is not None:
        return custom

    if hardware:
        hw_registry = RELAX_PASS_REGISTRY_BY_HW.get(hardware, {})
        pass_getter = _import_registry_function(hw_registry, backend, "Relax passes")
        if pass_getter is None:
            return None
        return pass_getter()
    return None


def get_codegen(backend: str):
    """Get codegen function for specified target backend."""
    codegen_func = _import_registry_function(CODEGEN_REGISTRY, backend, "codegen")
    if codegen_func is None:
        supported = sorted(SUPPORTED_BACKENDS)
        raise NotImplementedError(
            f"Backend '{backend}' does not have standalone codegen support.\n"
            f"Supported backends: {', '.join(supported)}"
        )
    return codegen_func


def get_tir_pipeline(backend: str):
    """Get TIR pipeline function for specified target backend."""
    pipeline_getter = _import_registry_function(TIR_PIPELINE_REGISTRY, backend, "TIR pipeline")
    if pipeline_getter is None:
        return None
    return pipeline_getter()


def register_codegen(backend: str, module_name: str, func_name: str):
    """Register a custom codegen function."""
    CODEGEN_REGISTRY[backend] = (module_name, func_name)


def get_makefile_generator(backend: str):
    """Get Makefile generator function for specified target backend."""
    makefile_func = _import_registry_function(MAKEFILE_REGISTRY, backend, "Makefile generator")
    if makefile_func is None:
        supported = sorted(MAKEFILE_REGISTRY.keys())
        raise NotImplementedError(
            f"Backend '{backend}' does not have Makefile generator support.\n"
            f"Supported backends: {', '.join(supported)}"
        )
    return makefile_func


def register_tir_pipeline(backend: str, module_name: str, func_name: str):
    """Register a custom TIR pipeline."""
    TIR_PIPELINE_REGISTRY[backend] = (module_name, func_name)


def register_makefile_generator(backend: str, module_name: str, func_name: str):
    """Register a custom Makefile generator."""
    MAKEFILE_REGISTRY[backend] = (module_name, func_name)

"""Configuration management for Sense compiler"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class HardwareConfig:
    """Hardware target configuration

    Attributes:
        device: Hardware target name (e.g., cpu, rpi2)
    """
    device: str = "cpu"

    def __post_init__(self):
        """Validate hardware configuration"""
        valid_devices = [
            "rpi2",
        ]
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. Must be one of {valid_devices}")

    def is_rpi2(self) -> bool:
        """Check if device is MCU (microcontroller)"""
        return self.device == "rpi2"

@dataclass
class BuildOptionConfig:
    """Build options for compilation

    Attributes:
        backend: Target backend (c, llvm, cuda, etc.)
        opt_level: Optimization level (0-3)
        target_str: Custom TVM target string (overrides backend if provided)
    """
    backend: str = "c"
    opt_level: int = 3
    target_str: Optional[str] = None

    def __post_init__(self):
        """Validate build option configuration"""
        if self.opt_level not in range(4):
            raise ValueError(f"Invalid opt_level: {self.opt_level}. Must be 0-3")

        valid_backends = ["c", "ncnn"]
        if not self.target_str and self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {valid_backends}")

    def get_tvm_target_string(self) -> str:
        """Generate TVM target string"""
        if self.target_str:
            return self.target_str

        if self.backend in ["c"]:
            return "c -keys=cpu"
        return self.backend


@dataclass
class OptimizerConfig:
    """Optimizer configuration for future extensions

    Attributes:
        custom_passes: Custom optimization passes (empty list uses default pipeline)
    """
    custom_passes: list[str] = field(default_factory=list)


@dataclass
class DebugConfig:
    """Debug configuration

    Attributes:
        dump_operator_outputs: Enable dumping per-operator outputs
        measure_execution_time: Enable per-op timing trace
    """
    dump_operator_outputs: bool = False
    measure_execution_time: bool = False


@dataclass
class ExportConfig:
    """Export configuration

    Attributes:
        output_dir: Output directory for compiled artifacts
        model_name: Output model name
        save_tir: Save TIR modules
        save_weight_manifest: Save weight manifest files
    """
    output_dir: str = "./bin"
    model_name: str = "sense_model"
    save_tir: bool = True
    save_weight_manifest: bool = True

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class SenseConfig:
    """Main configuration for Sense compiler

    Attributes:
        model_path: Path to input model
        hardware: Hardware configuration
        build_option: Build option configuration
        optimizer: Optimizer configuration
        export: Export configuration
        debug: Debug configuration
    """
    model_path: str = ""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    build_option: BuildOptionConfig = field(default_factory=BuildOptionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            "model_path": self.model_path,
            "hardware": self.hardware.device,
            "build_option": {
                "backend": self.build_option.backend,
                "opt_level": self.build_option.opt_level,
                "target_str": self.build_option.target_str,
            },
            "optimizer": {
                "custom_passes": self.optimizer.custom_passes,
            },
            "export": {
                "output_dir": self.export.output_dir,
                "model_name": self.export.model_name,
                "save_tir": self.export.save_tir,
                "save_weight_manifest": self.export.save_weight_manifest,
            },
            "debug": {
                "dump_operator_outputs": self.debug.dump_operator_outputs,
                "measure_execution_time": self.debug.measure_execution_time,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SenseConfig":
        """Create SenseConfig from dictionary"""
        model_path = config_dict.get("model_path")
        hardware_value = config_dict.get("hardware", "cpu")
        if isinstance(hardware_value, dict):
            hardware_device = hardware_value.get("device", "cpu")
        else:
            hardware_device = hardware_value
        hardware = HardwareConfig(device=hardware_device)

        build_dict = config_dict.get("build_option", {})
        build_option = BuildOptionConfig(
            backend=build_dict.get("backend", "c"),
            opt_level=build_dict.get("opt_level", 3),
            target_str=build_dict.get("target_str"),
        )

        optimizer_dict = config_dict.get("optimizer", {})
        optimizer = OptimizerConfig(
            custom_passes=optimizer_dict.get("custom_passes", []),
        )

        debug_dict = config_dict.get("debug", {})
        debug = DebugConfig(
            dump_operator_outputs=debug_dict.get("dump_operator_outputs", False),
            measure_execution_time=debug_dict.get("measure_execution_time", False),
        )

        export_dict = config_dict.get("export", {})
        export = ExportConfig(
            output_dir=export_dict.get("output_dir", "./bin"),
            model_name=export_dict.get("model_name", "sense_model"),
            save_tir=export_dict.get("save_tir", True),
            save_weight_manifest=export_dict.get("save_weight_manifest", True),
        )

        return cls(
            model_path=model_path,
            hardware=hardware,
            build_option=build_option,
            optimizer=optimizer,
            export=export,
            debug=debug,
        )

    def validate(self) -> bool:
        """Validate entire configuration"""
        try:
            # Validation is done in __post_init__ of each component
            if not self.model_path:
                raise ValueError("model_path is empty. Set 'model_path' in config file.")
            return True
        except ValueError as e:
            print(f"Configuration validation failed: {e}")
            return False

    def __str__(self) -> str:
        """Human-readable configuration summary"""
        return (
            "SenseConfig:\n"
            f"  Model Path: {self.model_path or 'unset'}\n"
            f"  Hardware: {self.hardware.device}\n"
            f"  Backend: {self.build_option.backend} (opt_level={self.build_option.opt_level})\n"
            f"  Output: {self.export.output_dir}\n"
            f"  Model: {self.export.model_name}"
        )


def create_default_config() -> SenseConfig:
    """Create default configuration with sensible defaults"""
    return SenseConfig(
        hardware=HardwareConfig(device="cpu"),
        build_option=BuildOptionConfig(backend="c", opt_level=3),
        optimizer=OptimizerConfig(custom_passes=[]),
        export=ExportConfig(output_dir="./bin", model_name="sense_model"),
    )

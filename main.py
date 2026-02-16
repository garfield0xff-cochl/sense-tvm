#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sense2.python.sense import Sense

config = {
    "output_dir": "./bin_sense2",
    "optimizer": {
        "apply_default_pipeline": True
    },
    "save_ir": True,
    "opt_level": 3
}

print("=" * 70)
print("Sense2 Test")
print("=" * 70)

sense = Sense(config)

sense.execute(
    model_path="sense_onnx/model_main_17.onnx",
    target="c",
    name="sense_model"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

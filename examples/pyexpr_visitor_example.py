#!/usr/bin/env python3
"""
PyExprVisitor Simple Examples

Demonstrates how to use TVM Relax PyExprVisitor to traverse and analyze IR.
"""

import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor, visitor


# Example 1: Count different node types
@visitor
class NodeCounter(PyExprVisitor):
    """Count different types of nodes in Relax IR."""

    def __init__(self):
        super().__init__()
        self.counts = {
            'call': 0,
            'var': 0,
            'constant': 0,
            'tuple': 0,
            'function': 0
        }

    def visit_call_(self, call):
        """Called when visiting Call nodes."""
        self.counts['call'] += 1
        # Continue visiting arguments
        for arg in call.args:
            self.visit_expr(arg)

    def visit_var_(self, var):
        """Called when visiting Var nodes."""
        self.counts['var'] += 1

    def visit_constant_(self, const):
        """Called when visiting Constant nodes."""
        self.counts['constant'] += 1

    def visit_tuple_(self, tup):
        """Called when visiting Tuple nodes."""
        self.counts['tuple'] += 1
        for field in tup.fields:
            self.visit_expr(field)

    def visit_function_(self, func):
        """Called when visiting Function nodes."""
        self.counts['function'] += 1
        # Visit function body
        self.visit_expr(func.body)


# Example 2: Collect all Constants
@visitor
class ConstantCollector(PyExprVisitor):
    """Collect all Constant nodes and their data."""

    def __init__(self):
        super().__init__()
        self.constants = []
        self.constant_shapes = []

    def visit_constant_(self, const):
        """Collect constant data."""
        try:
            data = const.data.numpy()
            self.constants.append(data)
            self.constant_shapes.append(data.shape)
        except:
            # Some constants might not be convertible
            pass

    def visit_call_(self, call):
        """Visit call arguments to find constants."""
        for arg in call.args:
            self.visit_expr(arg)

    def visit_tuple_(self, tup):
        """Visit tuple fields."""
        for field in tup.fields:
            self.visit_expr(field)


# Example 3: Find specific operations
@visitor
class OpFinder(PyExprVisitor):
    """Find all calls to specific operations."""

    def __init__(self, target_op_name):
        super().__init__()
        self.target_op = target_op_name
        self.found_calls = []

    def visit_call_(self, call):
        """Check if this call matches target operation."""
        op_name = None

        # Get operation name
        if isinstance(call.op, relax.ExternFunc):
            op_name = call.op.global_symbol
        elif isinstance(call.op, relax.GlobalVar):
            op_name = call.op.name_hint

        # Record if matches
        if op_name and self.target_op in op_name:
            self.found_calls.append({
                'op_name': op_name,
                'num_args': len(call.args)
            })

        # Continue visiting arguments
        for arg in call.args:
            self.visit_expr(arg)


# Example 4: Extract operation sequence
@visitor
class OperationSequence(PyExprVisitor):
    """Extract sequence of operations in execution order."""

    def __init__(self):
        super().__init__()
        self.sequence = []

    def visit_binding_block_(self, block):
        """Visit bindings in order."""
        for binding in block.bindings:
            self.visit_binding(binding)

    def visit_var_binding_(self, binding):
        """Record operation from binding."""
        # Visit the value (usually a Call)
        if isinstance(binding.value, relax.Call):
            call = binding.value
            op_name = "unknown"

            if isinstance(call.op, relax.ExternFunc):
                op_name = call.op.global_symbol
            elif isinstance(call.op, relax.GlobalVar):
                op_name = call.op.name_hint

            self.sequence.append({
                'var': binding.var.name_hint,
                'op': op_name
            })

        # Continue visiting the value
        self.visit_expr(binding.value)


def example_usage():
    """Demonstrate PyExprVisitor usage with a simple Relax IR."""

    print("=" * 60)
    print("PyExprVisitor Examples")
    print("=" * 60)

    # Create simple Relax function
    from tvm.script import ir as I
    from tvm.script import relax as R

    @I.ir_module
    class SimpleModule:
        @R.function
        def main(x: R.Tensor((10, 20), "float32")) -> R.Tensor:
            with R.dataflow():
                # Some operations
                y = R.add(x, x)
                z = R.multiply(y, R.const(2.0, "float32"))
                R.output(z)
            return z

    print("\nExample Module:")
    print(SimpleModule.script())

    # Get the main function
    main_func = SimpleModule["main"]

    # Example 1: Count nodes
    print("\n" + "=" * 60)
    print("Example 1: Count Node Types")
    print("=" * 60)
    counter = NodeCounter()
    counter.visit_expr(main_func)
    print(f"Node counts: {counter.counts}")

    # Example 2: Collect constants
    print("\n" + "=" * 60)
    print("Example 2: Collect Constants")
    print("=" * 60)
    collector = ConstantCollector()
    collector.visit_expr(main_func)
    print(f"Found {len(collector.constants)} constants")
    for i, shape in enumerate(collector.constant_shapes):
        print(f"  Constant[{i}]: shape={shape}")

    # Example 3: Find specific operations
    print("\n" + "=" * 60)
    print("Example 3: Find Operations")
    print("=" * 60)
    finder = OpFinder("add")
    finder.visit_expr(main_func)
    print(f"Found {len(finder.found_calls)} 'add' operations:")
    for call_info in finder.found_calls:
        print(f"  - {call_info['op_name']} ({call_info['num_args']} args)")

    # Example 4: Extract operation sequence
    print("\n" + "=" * 60)
    print("Example 4: Operation Sequence")
    print("=" * 60)
    seq_visitor = OperationSequence()
    seq_visitor.visit_expr(main_func)
    print(f"Operation sequence ({len(seq_visitor.sequence)} ops):")
    for i, op_info in enumerate(seq_visitor.sequence):
        print(f"  {i}: {op_info['var']} = {op_info['op']}")


def example_with_real_model():
    """Example with real ONNX model."""
    print("\n" + "=" * 60)
    print("Example 5: Real Model Analysis")
    print("=" * 60)

    from sense.python.parser import parse_model

    # Parse a simple model
    model_path = "sense_onnx/model_main_17.onnx"
    ir_mod, input_info, output_info, weights = parse_model(model_path)

    print(f"Model: {model_path}")
    print(f"  Inputs: {list(input_info.keys())}")
    print(f"  Outputs: {list(output_info.keys())}")

    # Get main function
    main_func = ir_mod["main"]

    # Count operations
    counter = NodeCounter()
    counter.visit_expr(main_func)
    print(f"\nNode statistics:")
    print(f"  Calls: {counter.counts['call']}")
    print(f"  Variables: {counter.counts['var']}")
    print(f"  Constants: {counter.counts['constant']}")

    # Collect constants
    collector = ConstantCollector()
    collector.visit_expr(main_func)
    print(f"\nConstants found: {len(collector.constants)}")
    print(f"  Sample shapes: {collector.constant_shapes[:5]}")

    # Find conv2d operations
    finder = OpFinder("conv2d")
    finder.visit_expr(main_func)
    print(f"\nConv2D operations: {len(finder.found_calls)}")

    # Extract sequence
    seq_visitor = OperationSequence()
    seq_visitor.visit_expr(main_func)
    print(f"\nOperation sequence: {len(seq_visitor.sequence)} operations")
    print(f"  First 5 ops:")
    for i, op_info in enumerate(seq_visitor.sequence[:5]):
        print(f"    {i}: {op_info['var']} = {op_info['op']}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Run simple examples
    example_usage()

    # Run real model example
    try:
        example_with_real_model()
    except Exception as e:
        print(f"\nReal model example failed: {e}")
        import traceback
        traceback.print_exc()

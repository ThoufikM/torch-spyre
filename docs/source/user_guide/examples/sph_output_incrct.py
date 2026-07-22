import argparse

import torch

from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import (
    declare_tensor_dim,
    name_tensor_dims,
)
import torch_spyre._inductor.span_overflow_hint_analysis as span_analysis


SHAPE = (32, 64, 1024, 2049)
DTYPE = torch.float16
DEVICE = torch.device("spyre")

torch.manual_seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a Spyre add operation with configurable tiling and span size."
    )

    parser.add_argument(
        "--max-span-bytes",
        type=int,
        default=None,
        help=(
            "Override span_overflow_hint_analysis.MAX_SPAN_BYTES. "
            "Example: --max-span-bytes 8192"
        ),
    )

    parser.add_argument(
        "--num-tiles",
        type=int,
        default=5,
        help="Number of tiles for the M dimension. Default: 5",
    )

    return parser.parse_args()


def configure_max_span_bytes(max_span_bytes):
    if max_span_bytes is None:
        print(
            "Using default MAX_SPAN_BYTES:",
            span_analysis.MAX_SPAN_BYTES,
        )
        return

    if max_span_bytes <= 0:
        raise ValueError("--max-span-bytes must be greater than zero")

    span_analysis.MAX_SPAN_BYTES = max_span_bytes

    print(
        "Overridden MAX_SPAN_BYTES:",
        span_analysis.MAX_SPAN_BYTES,
    )


def declare_dimensions():
    declare_tensor_dim("N", SHAPE[0])
    declare_tensor_dim("M", SHAPE[1])
    declare_tensor_dim("H", SHAPE[2])
    declare_tensor_dim("W", SHAPE[3])


def create_add_function(num_tiles):
    def add_tensors(a, b):
        # M has size 20.
        # For num_tiles=5, each tile covers 4 elements along M.
        with spyre_hint(tiles={"M": num_tiles}):
            result = a + b

        return result

    return add_tensors


def main():
    args = parse_arguments()

    configure_max_span_bytes(args.max_span_bytes)

    if args.num_tiles <= 0:
        raise ValueError("--num-tiles must be greater than zero")

    if SHAPE[1] % args.num_tiles != 0:
        raise ValueError(
            f"M dimension size {SHAPE[1]} is not divisible "
            f"by tile count {args.num_tiles}"
        )

    declare_dimensions()

    add_tensors = create_add_function(args.num_tiles)

    # Create CPU inputs.
    a_cpu = torch.randn(SHAPE, dtype=DTYPE)
    b_cpu = torch.randn(SHAPE, dtype=DTYPE)

    # CPU reference output.
    cpu_output = add_tensors(a_cpu, b_cpu)

    print("\nCPU Output:")
    print(cpu_output)

    # Move the same inputs to Spyre.
    a_spyre = a_cpu.to(DEVICE)
    b_spyre = b_cpu.to(DEVICE)

    # Attach symbolic dimension names.
    name_tensor_dims(a_spyre, ["N", "M", "H", "W"])
    name_tensor_dims(b_spyre, ["N", "M", "H", "W"])

    # MAX_SPAN_BYTES must be configured before torch.compile
    # and before the compiled function is executed.
    compiled_add = torch.compile(
        add_tensors,
        backend="inductor",
    )

    # Execute on Spyre.
    spyre_output = compiled_add(a_spyre, b_spyre)
    spyre_output_cpu = spyre_output.cpu()

    print("\nSpyre Output:")
    print(spyre_output_cpu)

    # Compare in float32 for clearer difference calculations.
    cpu_float = cpu_output.float()
    spyre_float = spyre_output_cpu.float()

    absolute_difference = torch.abs(cpu_float - spyre_float)

    max_difference = absolute_difference.max().item()
    mean_difference = absolute_difference.mean().item()

    absolute_tolerance = 1e-2
    relative_tolerance = 1e-2

    outputs_match = torch.allclose(
        cpu_float,
        spyre_float,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )

    values_exceeding_absolute_tolerance = torch.count_nonzero(
        absolute_difference > absolute_tolerance
    ).item()

    print("\nComparison:")
    print("Outputs Match:", outputs_match)
    print("Maximum absolute difference:", max_difference)
    print("Mean absolute difference:", mean_difference)
    print(
        "Values exceeding absolute tolerance:",
        values_exceeding_absolute_tolerance,
    )
    print("Total values:", cpu_output.numel())

    print("\nTiling Information:")
    print("Shape:", SHAPE)
    print("Tiled dimension: M")
    print("M dimension size:", SHAPE[1])
    print("Number of tiles:", args.num_tiles)
    print("Elements per tile:", SHAPE[1] // args.num_tiles)

    print("\nSpan Configuration:")
    print("MAX_SPAN_BYTES:", span_analysis.MAX_SPAN_BYTES)


if __name__ == "__main__":
    main()
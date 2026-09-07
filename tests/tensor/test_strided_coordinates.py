# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Address-level regressions for strided views and physical gap dimensions."""

import itertools

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import FixedLayout
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.pass_utils import (
    compute_strided_view_restickify_target,
    device_coordinates,
    strided_stick_host_dim,
)
from torch_spyre._inductor.views import compute_coordinates, normalize_coordinates
from torch_spyre._inductor.errors import Unsupported


@pytest.mark.parametrize("width,start,step", [(128, 0, 3), (130, 0, 3), (128, 1, 5)])
def test_partial_stride_does_not_cross_host_dimension(width, start, step):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    count = len(range(start, width, step))
    coords = compute_coordinates(
        [8, width], [width, 1], {row: 8, col: count}, width * row + step * col + start
    )
    assert coords == [row, step * col + start]


@pytest.mark.parametrize(
    "step,extent,start", [(2, 128, 0), (2, 128, 1), (3, 192, 1), (5, 320, 63)]
)
def test_affine_gap_preserves_physical_addresses(step, extent, start):
    row, col, lane = sympy.symbols("row col lane", integer=True, nonnegative=True)
    count = (128 - start + step - 1) // step
    ranges = {row: 2, col: count, lane: 64}
    synthetic = itertools.count()
    terms = normalize_coordinates(
        ranges, [2, extent, 64], [row, step * col + start, lane],
        lambda: sympy.Symbol(f"gap{next(synthetic)}"), compare_value=int,
    )
    flat = sympy.S.Zero
    stride = 1
    for term in reversed(terms):
        coord = term.offset
        if term.var is not None:
            coord += term.num * sympy.floor(sympy.Mod(term.var, term.mod) / term.den)
        flat += stride * coord
        stride *= term.dim_size
    assert stride == 2 * extent * 64
    for row_value, col_value, lane_value in itertools.product(
        range(2), range(count), (0, 31, 63)
    ):
        values = {var: 0 for var in flat.free_symbols}
        values.update({row: row_value, col: col_value, lane: lane_value})
        expected = (row_value * extent + step * col_value + start) * 64 + lane_value
        assert int(flat.subs(values)) == expected


def _slice_layouts(step, dtype=torch.float16):
    size = [2, 128, 8, 128]
    layout = FixedLayout(torch.device("cpu"), dtype, size, [131072, 1024, 128, 1])
    source = SpyreTensorLayout(size, dtype)
    b, h, seq, col = sympy.symbols("b h seq col", integer=True, nonnegative=True)
    count = len(range(0, 128, step))
    variables = (b, h, seq, col)
    ranges = (2, 8, 128, count)
    read = MemoryDep(
        "input", 131072 * b + 128 * h + 1024 * seq + step * col, variables, ranges
    )
    write = MemoryDep(
        "output",
        8 * 128 * count * b + 128 * count * h + count * seq + col,
        variables,
        ranges,
    )
    target = SpyreTensorLayout(
        list(ranges), [8 * 128 * count, 128 * count, count, 1], dtype, [0, 1, 3, 2]
    )
    return layout, source, read, target, write


@pytest.mark.parametrize("step", [2, 3, 4, 5, 8, 16, 32])
def test_restickifies_full_producer_before_strided_read(step):
    layout, source, read, output, write = _slice_layouts(step)
    assert strided_stick_host_dim(source, layout, read) == 3
    target = compute_strided_view_restickify_target(source, layout, read, output, write)
    assert target is not None
    assert target.stride_map[-1] == 1024
    target_stick = device_coordinates(target, read, None)[-1]
    assert target_stick == sympy.Mod(read.var_names[2], 64)
    # The old stick has room for complete step groups and complete sticks.
    old_axis = list(target.stride_map).index(1)
    assert target.device_size[old_axis] >= 128
    assert target.device_size[old_axis] % step == 0
    assert target.device_size[old_axis] % 64 == 0


def test_whole_stick_stride_keeps_existing_sparse_path():
    layout, source, read, output, write = _slice_layouts(64)
    assert strided_stick_host_dim(source, layout, read) is None
    target = compute_strided_view_restickify_target(source, layout, read, output, write)
    assert target is None


def test_strided_restickify_rejects_unsupported_device_format():
    layout, source, read, output, write = _slice_layouts(3, torch.float32)
    target = compute_strided_view_restickify_target(source, layout, read, output, write)
    assert target is None


@pytest.mark.parametrize("step", [3, 5, 6])
def test_nonunit_quotient_remainder_preserves_addresses(step):
    row, lane = sympy.symbols("row lane", integer=True, nonnegative=True)
    ranges = {row: 64, lane: 64}
    synthetic = itertools.count()
    terms = normalize_coordinates(
        ranges, [step, 64, 64],
        [sympy.floor(step * row / 64), sympy.Mod(step * row, 64), lane],
        lambda: sympy.Symbol(f"gap{next(synthetic)}"), compare_value=int,
    )
    flat = sympy.S.Zero
    stride = 1
    for term in reversed(terms):
        coord = term.offset
        if term.var is not None:
            coord += term.num * sympy.floor(sympy.Mod(term.var, term.mod) / term.den)
        flat += stride * coord
        stride *= term.dim_size
    assert stride == step * 64 * 64
    for row_value in range(64):
        values = {var: 0 for var in flat.free_symbols}
        values.update({row: row_value, lane: 17})
        assert int(flat.subs(values)) == step * row_value * 64 + 17


def test_nonunit_coalescing_keeps_physical_stick_boundary():
    row = sympy.Symbol("row", integer=True, nonnegative=True)
    # The quotient/remainder pair ends at the hardware stick. It cannot be
    # reshaped away; this access needs the explicit producer relayout instead.
    with pytest.raises(AssertionError, match="Unsupported coordinate expression"):
        normalize_coordinates(
            {row: 64}, [3, 64],
            [sympy.floor(3 * row / 64), sympy.Mod(3 * row, 64)],
            lambda: sympy.Symbol("gap"), compare_value=int,
        )


def test_nondivisible_affine_gap_requires_padded_allocation():
    col, lane = sympy.symbols("col lane", integer=True, nonnegative=True)
    with pytest.raises(Unsupported, match="must be padded to a multiple of step 3"):
        normalize_coordinates(
            {col: 43, lane: 64}, [128, 64], [3 * col, lane],
            lambda: sympy.Symbol("gap"), compare_value=int,
        )

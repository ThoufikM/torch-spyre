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

"""Strided reads must preserve values across physical stick boundaries."""

import pytest
import torch

from utils_inductor import _compile_and_run


@pytest.mark.parametrize(
    "width,start,step",
    [(128, 0, step) for step in (2, 3, 4, 5, 8, 16, 32, 64)]
    + [(128, 1, 2), (128, 1, 3), (128, 63, 3), (130, 0, 3), (130, 1, 5), (96, 0, 3)],
)
def test_transpose_strided_slice_add(width, start, step):
    # Exactly representable values make incorrect addresses visible without
    # introducing a tolerance that could hide a misplaced element.
    shape = (2, 128, 8, width)
    x = (torch.arange(2 * 128 * 8 * width) % 251).reshape(shape).to(torch.float16)

    def fn(x):
        y = x.transpose(1, 2)[..., start::step]
        return y + y

    actual = _compile_and_run(fn, (x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)


@pytest.mark.parametrize("step", [3, 5, 6])
def test_nonunit_fraction_across_adjacent_device_dimensions(step):
    x = (
        (torch.arange(2 * step * 64 * 64) % 251)
        .reshape(2, step, 64, 64)
        .to(torch.float16)
    )

    def fn(x):
        # Flattening adjacent non-stick dimensions followed by this slice
        # produces floor(step*i/64), (step*i)%64 on the original buffer.
        y = x.flatten(1, 2)[:, ::step, :]
        return y + y

    actual = _compile_and_run(fn, (x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_strided_slice_relayout_preserves_input(dtype):
    x = (torch.arange(2 * 128 * 8 * 130) % 97).reshape(2, 128, 8, 130).to(dtype)
    device_x = x.to("spyre")

    def fn(x):
        y = x.transpose(1, 2)[..., 1::3]
        return y + y

    actual = _compile_and_run(fn, (device_x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    torch.testing.assert_close(device_x.cpu(), x, rtol=0, atol=0)

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


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 7, 8, 16, 32, 64])
def test_strided_slice_scalar_add(step):
    x = (torch.arange(2 * 128 * 8 * 128) % 251).reshape(2, 128, 8, 128).half()

    def fn(x):
        return x[..., ::step] + 1

    actual = _compile_and_run(fn, (x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)


@pytest.mark.parametrize("step,start", [(3, 0), (3, 1), (5, 0), (5, 63)])
def test_nonunit_fraction_across_three_device_dimensions(step, start):
    x = (torch.arange(2 * step * 8 * 8 * 64) % 251).reshape(2, step, 8, 8, 64).half()

    def fn(x):
        return x.flatten(1, 3)[:, start::step, :] + 1

    actual = _compile_and_run(fn, (x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)


@pytest.mark.parametrize("step", [3, 5, 7])
def test_nonunit_fraction_with_periodic_consumer(step):
    x = (torch.arange(step * 8 * 8 * 64) % 251).reshape(step, 8, 8, 64).half()

    def fn(x):
        return x.flatten(0, 2)[::step, :].repeat(2, 1) + 1

    device_x = x.to("spyre")
    actual = _compile_and_run(fn, (device_x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    torch.testing.assert_close(device_x.cpu(), x, rtol=0, atol=0)


@pytest.mark.parametrize("step,start", [(3, 0), (3, 1), (5, 0), (5, 63)])
def test_nonstick_strided_slice_partial_span(step, start):
    x = (torch.arange(2 * 128 * 64) % 251).reshape(2, 128, 64).half()

    def fn(x):
        return x[:, start::step, :] + 1

    device_x = x.to("spyre")
    actual = _compile_and_run(fn, (device_x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    torch.testing.assert_close(device_x.cpu(), x, rtol=0, atol=0)


@pytest.mark.parametrize("step,start", [(3, 0), (3, 1), (5, 0), (5, 63)])
def test_fractional_digits_partial_span(step, start):
    x = (torch.arange(2 * 2 * 64 * 64) % 251).reshape(2, 2, 64, 64).half()

    def fn(x):
        return x.flatten(1, 2)[:, start::step, :] + 1

    device_x = x.to("spyre")
    actual = _compile_and_run(fn, (device_x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    torch.testing.assert_close(device_x.cpu(), x, rtol=0, atol=0)


@pytest.mark.xfail(
    strict=True,
    raises=torch._inductor.exc.InductorError,
    reason="#1353: a 1-D strided stick has no alternate logical stick axis",
)
@pytest.mark.parametrize("step,start", [(2, 0), (3, 0), (5, 1)])
def test_one_dimensional_strided_slice(step, start):
    x = torch.arange(128).half()

    def fn(x):
        return x[start::step] + 1

    actual = _compile_and_run(fn, (x,), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)


@pytest.mark.parametrize("step", [2, 3, 5])
def test_strided_slice_add_independent_operand(step):
    x = (torch.arange(2 * 128 * 8 * 128) % 251).reshape(2, 128, 8, 128).half()
    count = len(range(0, 128, step))
    bias = (torch.arange(2 * 128 * 8 * count) % 97).reshape(2, 128, 8, count).half()

    def fn(x, bias):
        return x[..., ::step] + bias

    actual = _compile_and_run(fn, (x, bias), torch.device("spyre"))
    torch.testing.assert_close(actual, fn(x, bias), rtol=0, atol=0)

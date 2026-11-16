# Copyright 2026 The HuggingFace Inc. team.
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
"""
Integration of the "flash-linear-attention" kernels, reachable through the `kernels` package or the original package.
This interface can be used by passing `linear_attn_implementation="kernels-community/fla"`, which directly reaches
for the Hub kernel through `kernels`. You can also use `linear_attn_implementation="fla"`, which reaches for the
original package and falls back to the Hub kernel if the package is not installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..utils import logging
from ..utils.import_utils import is_flash_linear_attention_available, is_kernels_available


if TYPE_CHECKING:
    from ..cache_utils import Cache


logger = logging.get_logger(__name__)

# The name under which the Hub kernel is requested, like in flash attention
FLA_HUB_KERNEL = "kernels-community/fla"

# `globals()` is not compatible with dynamo, hence we define the loaded kernels in global scope ourselves
_loaded_implementation = None
_fla_conv_fn = None
_fla_conv_update_fn = None


def _load_hub_fla_conv() -> tuple[Callable, Callable]:
    """`causal_conv1d` and `causal_conv1d_update` from the Hub kernel (`kernels-community/fla`)."""
    from .hub_kernels import lazy_load_kernel

    if not is_kernels_available():
        raise ImportError(
            f'`linear_attn_implementation="{FLA_HUB_KERNEL}"` needs the `kernels` package to load the Hub kernel. '
            "Install it with `pip install kernels`."
        )
    kernel = lazy_load_kernel("fla")
    if kernel is None:
        raise ImportError(
            f"Could not load the `{FLA_HUB_KERNEL}` Hub kernel. Check the warnings above, your connection to the Hub, "
            "and that `USE_HUB_KERNELS` is not set to disable the Hub kernels."
        )
    return kernel.modules.convolution.causal_conv1d, kernel.modules.convolution.causal_conv1d_update


def lazy_import_fla_conv(implementation: str) -> tuple[Callable, Callable]:
    """`causal_conv1d` and `causal_conv1d_update` from `flash-linear-attention`: from the Hub kernel for
    `"kernels-community/fla"`, from the package for `"fla"`. The fallback from the package to the Hub kernel is
    resolved beforehand, when the implementation is set on the model."""
    global _loaded_implementation, _fla_conv_fn, _fla_conv_update_fn
    # Only (re)load if another implementation is requested
    if implementation != _loaded_implementation:
        if implementation == FLA_HUB_KERNEL:
            _fla_conv_fn, _fla_conv_update_fn = _load_hub_fla_conv()
        elif is_flash_linear_attention_available():
            from fla.modules.convolution import causal_conv1d, causal_conv1d_update

            _fla_conv_fn, _fla_conv_update_fn = causal_conv1d, causal_conv1d_update
        else:
            raise ImportError(
                '`linear_attn_implementation="fla"` needs the `flash-linear-attention` package. Install it with `pip '
                "install flash-linear-attention`, or install `kernels` to use the Hub kernel with "
                f'`linear_attn_implementation="{FLA_HUB_KERNEL}"`.'
            )
        # Only mark the implementation as loaded once the import succeeded, so that a failed load is retried
        _loaded_implementation = implementation
    return _fla_conv_fn, _fla_conv_update_fn


def fla_causal_conv1d_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    cache: Cache | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """`"fla"` and `"kernels-community/fla"` implementation of the `CausalConv1dInterface`, with the
    `flash-linear-attention` kernels. Packed sequences (`cu_seq_lens_q`) are kept apart."""
    conv_fn, conv_update_fn = lazy_import_fla_conv(module.config._linear_attn_implementation)
    layer_idx = module.layer_idx
    weight, bias, activation = module.conv1d.weight.squeeze(1), module.conv1d.bias, module.activation
    conv_state = cache.get_conv_state(layer_idx) if cache is not None else None

    # Single-token decode: the update kernel advances the conv state in place
    if conv_state is not None and hidden_states.shape[1] == 1 and not cache.is_recording_past(layer_idx):
        hidden_states, _ = conv_update_fn(hidden_states, conv_state, weight=weight, bias=bias, activation=activation)
        return hidden_states

    # Packed sequences are kept apart with `cu_seq_lens_q` to match the `fla` library. An older variation of this arg
    # was `seq_idx`, which carries the same information but is not favored. It is supported with a warning.
    cu_seqlens, seq_idx = kwargs.get("cu_seq_lens_q"), kwargs.get("seq_idx")
    if cu_seqlens is None and seq_idx is not None:
        logger.warning_once(
            "`seq_idx` was passed without `cu_seq_lens_q`: converting it to `cu_seq_lens_q` for the convolution. Pass "
            "`cu_seq_lens_q` to avoid this warning."
        )
        _, seq_lens = torch.unique_consecutive(seq_idx.flatten(), return_counts=True)
        cu_seqlens = nn.functional.pad(seq_lens.cumsum(0), (1, 0)).to(torch.int32)
    elif cu_seqlens is None:
        raise ValueError("`cu_seq_lens_q` is required when `seq_idx` is not passed.")

    output, _ = conv_fn(
        hidden_states,
        weight=weight,
        bias=bias,
        initial_state=None if conv_state is None else conv_state[..., -module.conv_kernel_size :],
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    # The cache keeps the last inputs as the next conv state. This comes after the convolution, which reads the current
    # conv state that the update overwrites in place
    if cache is not None:
        cache.update_conv_state(hidden_states.transpose(1, 2), layer_idx, conv_kernel_size=module.conv_kernel_size)
    return output

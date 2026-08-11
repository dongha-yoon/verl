# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Load already TP-local tensors without applying vLLM's TP slice twice."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType

import torch

_LOCAL_METHODS = {
    "load_column_parallel_weight": "_load_local_weight",
    "load_row_parallel_weight": "_load_local_weight",
    "load_merged_column_weight": "_load_local_fused_weight",
    "load_qkv_weight": "_load_local_fused_weight",
}


def _get_vocab_parallel_embedding_type() -> type[torch.nn.Module]:
    """Import lazily so this adapter remains importable without vLLM installed."""

    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    return VocabParallelEmbedding


def _vocab_parallel_weight_ids(
    models: list[torch.nn.Module],
    vocab_parallel_embedding_type: type[torch.nn.Module],
) -> set[int]:
    """Find weights owned by vLLM vocabulary-parallel layers."""

    return {
        id(module.weight)
        for model in models
        for module in model.modules()
        if isinstance(module, vocab_parallel_embedding_type) and isinstance(module.weight, torch.nn.Parameter)
    }


def _copy_exact(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy one already-local tensor without applying another TP partition.

    Args:
        target: vLLM parameter storage local to the current TP rank.
        source: Rank-local checkpoint tensor for the same parameter.

    Raises:
        ValueError: If the local shapes differ. A scalar source may populate a
            singleton target because vLLM represents some scalar parameters as
            one-element tensors.
    """

    if target.ndim == 1 and target.numel() == 1 and source.ndim == 0:
        source = source.reshape(1)
    if target.shape != source.shape:
        raise ValueError(f"TP-local weight shape mismatch: target={target.shape}, source={source.shape}")
    target.copy_(source)


def _load_local_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor, **_kwargs) -> None:
    """Implement vLLM's basic weight-loader contract for a rank-local tensor.

    Args:
        param: Destination parameter owned by the current vLLM worker.
        loaded_weight: Tensor already partitioned for this worker's TP rank.
        **_kwargs: Loader metadata accepted for signature compatibility. Basic
            local copies do not need it.
    """

    _copy_exact(param.data, loaded_weight)


def _load_local_fused_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor, **kwargs) -> None:
    """Load one component of an already-TP-local fused parameter.

    Args:
        param: Destination fused parameter owned by the current vLLM worker.
        loaded_weight: Rank-local component to copy.
        **kwargs: vLLM loader metadata. ``shard_offset`` and ``shard_size``
            select the component within a merged-column or QKV parameter.
    """

    output_dim = getattr(param, "output_dim", None)
    offset, size = kwargs.get("shard_offset"), kwargs.get("shard_size")
    if output_dim is None or offset is None or size is None:
        _copy_exact(param.data, loaded_weight)
        return
    if getattr(param, "packed_dim", None) == output_dim:
        size, offset = param.adjust_shard_indexes_for_packing(shard_offset=offset, shard_size=size)
    _copy_exact(param.data.narrow(output_dim, offset, size), loaded_weight)


def _load_local_vocab_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Load TP-local vocabulary rows and clear any vLLM padding rows.

    Args:
        param: Destination embedding or language-model-head parameter.
        loaded_weight: Unpadded vocabulary rows local to this TP rank.

    Raises:
        ValueError: If the tensors do not have compatible ranks and dimensions.
    """

    output_dim = getattr(param, "output_dim", None)
    if output_dim is None or param.ndim != loaded_weight.ndim:
        raise ValueError("invalid TP-local vocabulary parameter")
    for dim, (target_size, source_size) in enumerate(zip(param.shape, loaded_weight.shape, strict=True)):
        if (dim == output_dim and source_size > target_size) or (dim != output_dim and source_size != target_size):
            raise ValueError(f"TP-local vocabulary shape mismatch: {param.shape} vs {loaded_weight.shape}")
    param.data.zero_()
    param.data.narrow(output_dim, 0, loaded_weight.shape[output_dim]).copy_(loaded_weight)


@contextmanager
def use_local_tp_weight_loaders(models: Iterator[torch.nn.Module]):
    """Temporarily replace vLLM's global-tensor loaders with local-copy loaders.

    Args:
        models: Main and optional draft models receiving the same rank-local
            update.

    Yields:
        Control after installing local loaders. Every overwritten parameter
        attribute is restored when the context exits, including on failure.
    """

    models = list(models)
    vocab_parallel_embedding_type = _get_vocab_parallel_embedding_type()
    vocab_parallel_weight_ids = _vocab_parallel_weight_ids(models, vocab_parallel_embedding_type)
    saved = []
    seen = set()
    for model in models:
        for _, param in model.named_parameters(remove_duplicate=False):
            # Tied weights can appear under multiple names or across model views;
            # patch and restore each Parameter object exactly once.
            if id(param) in seen:
                continue
            seen.add(id(param))
            methods = {}
            for method_name, replacement_name in _LOCAL_METHODS.items():
                if hasattr(param, method_name):
                    # Record whether the method was an instance override. Removing
                    # our temporary override later reveals class-defined methods.
                    methods[method_name] = (method_name in param.__dict__, param.__dict__.get(method_name))
                    setattr(param, method_name, MethodType(globals()[replacement_name], param))
            saved.append(
                (
                    param,
                    hasattr(param, "is_sharded_weight"),
                    getattr(param, "is_sharded_weight", None),
                    hasattr(param, "weight_loader"),
                    getattr(param, "weight_loader", None),
                    methods,
                )
            )
            param.is_sharded_weight = True
            if id(param) in vocab_parallel_weight_ids:
                param.weight_loader = _load_local_vocab_weight
    try:
        yield
    finally:
        for param, had_sharded, old_sharded, had_loader, old_loader, methods in reversed(saved):
            if had_sharded:
                param.is_sharded_weight = old_sharded
            else:
                delattr(param, "is_sharded_weight")
            if had_loader:
                param.weight_loader = old_loader
            elif hasattr(param, "weight_loader"):
                delattr(param, "weight_loader")
            for method_name, (had_override, old_override) in methods.items():
                if had_override:
                    setattr(param, method_name, old_override)
                else:
                    delattr(param, method_name)

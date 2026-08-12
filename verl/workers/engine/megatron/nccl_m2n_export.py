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

"""No-gather conversion of local Megatron TP shards to Hugging Face shards."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from verl.models.transformers.hf_dense_decoder_tp import infer_dense_decoder_tp_shard_dim

__all__ = ["MegatronNCCLM2NWeight", "expert_name", "export_local_nccl_m2n_weights"]


@dataclass(frozen=True)
class MegatronNCCLM2NWeight:
    """One HF-named local weight plus its MBridge TP metadata."""

    name: str
    tensor: torch.Tensor
    global_shape: torch.Size
    destination_shard_dim: int | None
    source_shard_dim: int | None
    source_shard_rank: int
    source_shard_size: int
    # Expert stacks have shape [local experts, *local HF tensor shape].
    # The name is a format string with one slot for the global expert ID.
    expert_ids: tuple[int, ...] | None = None


def _global_shape(local_shape: Sequence[int], shard_dim: int | None, shard_size: int) -> torch.Size:
    """Recover a global shape from one evenly sharded local shape."""

    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    shape = list(local_shape)
    if shard_dim is not None:
        if not 0 <= shard_dim < len(shape):
            raise ValueError(f"shard_dim={shard_dim} is invalid for local shape {tuple(shape)}")
        shape[shard_dim] *= shard_size
    return torch.Size(shape)


def _mapped_hf_names(bridge: Any, mcore_name: str) -> list[str]:
    """Normalize the vanilla MBridge private name mapper to a list of names."""

    mapper = getattr(bridge, "_weight_name_mapping_mcore_to_hf", None)
    if mapper is None:
        raise RuntimeError("the installed MBridge does not expose _weight_name_mapping_mcore_to_hf()")
    mapped = mapper(mcore_name)
    names = [mapped] if isinstance(mapped, str) else list(mapped)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"invalid HF name mapping for {mcore_name!r}: {mapped!r}")
    return names


def expert_name(name: str) -> tuple[str, int]:
    """Return an HF expert-name template and its concrete expert ID."""

    match = re.search(r"(?<=\.experts\.)\d+(?=\.)", name)
    if match is None:
        raise ValueError(f"expected a numbered HF expert name: {name}")
    return name[: match.start()] + "{}" + name[match.end() :], int(match[0])


def _split_local_qkv(bridge: Any, name: str, tensor: torch.Tensor, tp_size: int) -> list[torch.Tensor]:
    """Split one TP-local grouped-QKV tensor without gathering the other TP lanes."""

    config = bridge.hf_config
    attention_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    hidden_size = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // attention_heads) or hidden_size // attention_heads)
    if min(attention_heads, kv_heads, head_dim, tp_size) <= 0:
        raise ValueError(f"invalid QKV dimensions for {name}")
    if attention_heads % kv_heads:
        raise ValueError(f"num_attention_heads={attention_heads} is not divisible by num_key_value_heads={kv_heads}")
    if kv_heads % tp_size:
        raise ValueError(f"num_key_value_heads={kv_heads} is not divisible by TP={tp_size} for {name}")

    local_kv_heads = kv_heads // tp_size
    q_rows_per_kv_head = head_dim * attention_heads // kv_heads
    rows_per_local_kv_head = q_rows_per_kv_head + 2 * head_dim
    expected_rows = local_kv_heads * rows_per_local_kv_head
    if tensor.ndim == 0 or tensor.shape[0] != expected_rows:
        raise ValueError(f"unexpected local QKV shape for {name}: first dimension must be {expected_rows}")

    tail = tuple(tensor.shape[1:])
    qkv = tensor.reshape(local_kv_heads, rows_per_local_kv_head, *tail)
    return [
        qkv[:, :q_rows_per_kv_head].reshape(-1, *tail).contiguous(),
        qkv[:, q_rows_per_kv_head : q_rows_per_kv_head + head_dim].reshape(-1, *tail).contiguous(),
        qkv[:, q_rows_per_kv_head + head_dim :].reshape(-1, *tail).contiguous(),
    ]


def _convert_local_weight(
    bridge: Any, mcore_name: str, tensor: torch.Tensor, tp_size: int
) -> tuple[list[str], list[torch.Tensor]]:
    """Convert one local dense-Qwen MCore tensor to local HF tensor(s)."""

    names = _mapped_hf_names(bridge, mcore_name)
    if "self_attention.linear_qkv." in mcore_name and "layer_norm" not in mcore_name:
        if len(names) != 3:
            raise ValueError(f"expected Q/K/V names for {mcore_name}, got {names}")
        return names, _split_local_qkv(bridge, mcore_name, tensor, tp_size)

    if "linear_fc1.weight" in mcore_name or "linear_fc1.bias" in mcore_name:
        if len(names) != 2:
            raise ValueError(f"expected gate/up names for {mcore_name}, got {names}")
        if tensor.ndim == 0 or tensor.shape[0] % 2:
            raise ValueError(f"gate/up tensor for {mcore_name} must have an even first dimension")
        gate, up = tensor.chunk(2, dim=0)
        return names, [gate.contiguous(), up.contiguous()]

    if len(names) != 1:
        raise NotImplementedError(f"unsupported local MBridge conversion for {mcore_name}: {names}")
    if "embedding.word_embeddings.weight" in mcore_name or "output_layer.weight" in mcore_name:
        padded_vocab_size = getattr(bridge, "padded_vocab_size", None)
        vocab_size = getattr(bridge, "vocab_size", None)
        if padded_vocab_size != vocab_size:
            raise ValueError(
                "NCCL M2N local export requires padded_vocab_size to equal vocab_size; "
                f"got {padded_vocab_size} and {vocab_size}"
            )
    return names, [tensor.detach().contiguous()]


@torch.no_grad()
def export_local_nccl_m2n_weights(
    bridge: Any, models: list[torch.nn.Module]
) -> Generator[MegatronNCCLM2NWeight, None, None]:
    """Yield final HF-named TP-local shards without a Megatron TP gather.

    This consumes the eight-field records produced by the separate vanilla
    MBridge ``export_weights_without_gather`` work. The exporter must yield the
    same ordered record sequence on every participating DP/TP rank.
    """

    export = getattr(bridge, "export_weights_without_gather", None)
    if export is None:
        raise RuntimeError("the installed MBridge does not expose export_weights_without_gather()")

    experts: dict[str, list[tuple[int, torch.Tensor, int, int, int, int, int | None]]] = defaultdict(list)
    for record in export(models):
        try:
            fields = tuple(record)
        except TypeError as exc:
            raise TypeError("MBridge local-export records must be iterable") from exc
        if len(fields) != 8:
            raise ValueError(f"unexpected MBridge local-export record with {len(fields)} fields")

        mcore_name, tp_rank, tp_size, ep_rank, ep_size, is_tp_sharded, shard_dim, tensor = fields
        if not isinstance(mcore_name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("MBridge local-export records require a string name and torch.Tensor payload")
        tp_rank, tp_size, ep_rank, ep_size = map(int, (tp_rank, tp_size, ep_rank, ep_size))
        source_dim = int(shard_dim) if bool(is_tp_sharded) else None
        source_size = tp_size if source_dim is not None else 1
        names, tensors = _convert_local_weight(bridge, mcore_name, tensor, source_size)
        for name, local_tensor in zip(names, tensors, strict=True):
            if ".experts." in name:
                if tp_size <= 0 or not 0 <= tp_rank < tp_size:
                    raise ValueError(f"invalid expert TP coordinate rank={tp_rank}, size={tp_size} for {mcore_name}")
                template, expert_id = expert_name(name)
                # MBridge reports the MCore matrix axis. Record the converted
                # HF axis after prepending the local-expert dimension.
                if name.endswith(("gate_proj.weight", "up_proj.weight")):
                    matrix_dim = 1
                elif name.endswith("down_proj.weight"):
                    matrix_dim = 2
                else:
                    raise NotImplementedError(f"unsupported expert projection: {name}")
                experts[template].append(
                    (
                        expert_id,
                        local_tensor.detach(),
                        ep_rank,
                        max(1, ep_size),
                        tp_rank,
                        tp_size,
                        matrix_dim if tp_size > 1 else None,
                    )
                )
                continue
            if bool(is_tp_sharded):
                if tp_size <= 0 or not 0 <= tp_rank < tp_size:
                    raise ValueError(f"invalid TP coordinate rank={tp_rank}, size={tp_size} for {mcore_name}")
            elif tp_rank != 0 or tp_size < 0:
                raise ValueError(f"invalid replicated TP coordinate rank={tp_rank}, size={tp_size} for {mcore_name}")
            if ep_rank != 0 or ep_size not in (0, 1):
                raise ValueError(f"expert placement attached to a dense weight: {name}")
            destination_dim = infer_dense_decoder_tp_shard_dim(name)
            if destination_dim != source_dim:
                raise NotImplementedError(
                    f"trainer and rollout TP placements differ for {name}: {source_dim} vs {destination_dim}"
                )
            yield MegatronNCCLM2NWeight(
                name=name,
                tensor=local_tensor.detach(),
                global_shape=_global_shape(local_tensor.shape, source_dim, source_size),
                destination_shard_dim=destination_dim,
                source_shard_dim=source_dim,
                source_shard_rank=tp_rank if source_dim is not None else 0,
                source_shard_size=source_size,
            )

    num_experts = int(getattr(bridge.hf_config, "num_experts", 0))
    for name in sorted(experts):
        records = sorted(experts[name], key=lambda item: item[0])
        _, first, ep_rank, ep_size, tp_rank, source_size, source_dim = records[0]
        if num_experts <= 0 or num_experts % ep_size:
            raise ValueError(f"invalid expert count {num_experts} for EP{ep_size}")
        count = num_experts // ep_size
        expert_ids = tuple(item[0] for item in records)
        expected_ids = tuple(range(ep_rank * count, (ep_rank + 1) * count))
        if expert_ids != expected_ids:
            raise ValueError(f"incomplete local expert ownership for {name}: {expert_ids}")
        if any(
            item[2:] != records[0][2:] or item[1].shape != first.shape or item[1].dtype != first.dtype
            for item in records
        ):
            raise ValueError(f"inconsistent expert shards for {name}")
        # Stack bytes because torch.stack does not implement every checkpoint
        # scale dtype, notably E8M0.
        stack = torch.stack([item[1].contiguous().view(torch.uint8) for item in records]).view(first.dtype)
        global_shape = list(_global_shape(stack.shape, source_dim, source_size))
        global_shape[0] = num_experts
        yield MegatronNCCLM2NWeight(
            name=name,
            tensor=stack,
            global_shape=torch.Size(global_shape),
            destination_shard_dim=None,
            source_shard_dim=source_dim,
            source_shard_rank=tp_rank,
            source_shard_size=source_size,
            expert_ids=expert_ids,
        )

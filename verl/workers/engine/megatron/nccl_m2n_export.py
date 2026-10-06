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

"""Shared descriptors for owner-local Megatron-to-Hugging Face weight export."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

__all__ = ["MegatronNCCLM2NWeight", "expert_name"]


@dataclass(frozen=True)
class MegatronNCCLM2NWeight:
    """One HF-named local weight plus its shard and expert ownership metadata."""

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


def expert_name(name: str) -> tuple[str, int]:
    """Return an HF expert-name template and its concrete expert ID."""

    match = re.search(r"(?<=\.experts\.)\d+(?=\.)", name)
    if match is None:
        raise ValueError(f"expected a numbered HF expert name: {name}")
    return name[: match.start()] + "{}" + name[match.end() :], int(match[0])

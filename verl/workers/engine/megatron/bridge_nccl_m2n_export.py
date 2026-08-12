# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-local Megatron-Bridge export that preserves checkpoint conversion.

Only construction of Bridge's parameter directory is collective. Conversion
uses the model's real mappings with PP broadcasts and EP/TP tensor gathers
replaced by local identities. Each expert family therefore contains exactly
the experts owned by this EP rank.
"""

import copy
from collections import defaultdict

import torch

from .nccl_m2n_export import MegatronNCCLM2NWeight, expert_name


class _LocalGroup:
    """Expose group geometry but fail if a mapping attempts communication."""

    def __init__(self, size=1, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank

    def __getattr__(self, name):
        raise RuntimeError(f"local conversion attempted process-group operation {name}")


def make_local_mapping(mapping, module):
    """Run Bridge's TP1/ETP1 transforms with PP/EP communication removed."""

    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    from megatron.core.utils import get_pg_rank, get_pg_size

    # AutoMapping must resolve before copying; otherwise first conversion would
    # create a child mapping holding the real process groups.
    if hasattr(mapping, "_detect_parallelism_type") and getattr(mapping, "_mapping", None) is None:
        kind = mapping._detect_parallelism_type(module)
        mapping._mapping = mapping._get_or_create_mapping(kind)
        mapping._detected_type = kind
    local = copy.copy(mapping)
    local.pp_group = _LocalGroup()
    for attribute in ("ep_group", "_tp_group", "_etp_group"):
        group = getattr(mapping, attribute, None)
        setattr(local, attribute, _LocalGroup(get_pg_size(group), get_pg_rank(group)))
    local.gather_from_tp_ranks = lambda tensor: [tensor]
    local.gather_from_ep_ranks = lambda tensor, module, name: {str(name): tensor}
    local.gather_from_ep_ranks_scale = lambda tensor, module, name: {
        str(name): tensor.unsqueeze(0).squeeze().unsqueeze(-1)
    }
    for attribute, child in vars(mapping).items():
        if isinstance(child, MegatronParamMapping):
            setattr(local, attribute, make_local_mapping(child, module))
    return local


class BridgeNCCLM2NExport:
    """Cache owner-local Bridge conversion tasks and export their current values."""

    def __init__(self, bridge, modules, *, ep_rank, ep_size, num_experts):
        self.bridge = bridge
        self.ep_rank, self.ep_size = ep_rank, ep_size
        self.num_experts = num_experts
        if ep_size <= 0 or not 0 <= ep_rank < ep_size or num_experts % ep_size:
            raise ValueError(f"invalid expert ownership: EP rank {ep_rank}/{ep_size}, {num_experts} experts")
        self.dense = []
        self.experts = defaultdict(list)

        # Every trainer calls this once. The only exchange is the parameter
        # directory across PP; no tensor value is gathered or broadcast.
        tasks = bridge.get_conversion_tasks(modules)
        self.model_bridge = bridge._model_bridge
        for task in tasks:
            if task is None or task.param_weight is None:
                continue
            if task.weight_dtype is not None or getattr(task.mapping, "is_grouped_export", False):
                raise ValueError("local Bridge export requires ordinary checkpoint-format mapping tasks")
            # The pinned Bridge's etp_size property references a nonexistent
            # etp_group attribute; the actual process group is _etp_group.
            if task.mapping.tp_size != 1 or task.mapping._etp_group.size() != 1:
                raise NotImplementedError("local Megatron-Bridge mappings currently require TP1/ETP1")
            local_mapping = make_local_mapping(task.mapping, task.megatron_module)
            if task.mapping.is_expert:
                names = task.mapping.hf_param
                first_name = names if isinstance(names, str) else next(iter(names.values()))
                template, expert_id = expert_name(first_name)
                self.experts[template].append((expert_id, task, local_mapping))
            else:
                self.dense.append((task, local_mapping))

        self.dense.sort(key=lambda pair: pair[0].global_param_name)
        expected = list(range(ep_rank * (num_experts // ep_size), (ep_rank + 1) * (num_experts // ep_size)))
        for family, records in self.experts.items():
            records.sort(key=lambda record: record[0])
            if [record[0] for record in records] != expected:
                raise ValueError(f"incomplete local expert ownership for {family}: expected {expected}")

    def convert(self, task, local_mapping):
        converted = local_mapping.megatron_to_hf(task.param_weight.detach(), task.megatron_module)
        # Preserve the complete model-specific hook, including MXFP4/FP8 scales.
        return self.model_bridge.maybe_modify_converted_hf_weight(task, converted, self.bridge.hf_pretrained.state)

    @torch.no_grad()
    def weights(self):
        for task, local_mapping in self.dense:
            for name, tensor in self.convert(task, local_mapping).items():
                yield MegatronNCCLM2NWeight(name, tensor.detach().contiguous(), tensor.shape, None, None, 0, 1)

        for family in sorted(self.experts):
            outputs = defaultdict(list)
            for expert_id, task, local_mapping in self.experts[family]:
                for name, tensor in self.convert(task, local_mapping).items():
                    template, output_id = expert_name(name)
                    if output_id != expert_id:
                        raise ValueError(f"local conversion changed expert ownership: {name}, expected {expert_id}")
                    outputs[template].append(tensor.detach())
            for template in sorted(outputs):
                tensors = outputs.pop(template)
                if len(tensors) != self.num_experts // self.ep_size:
                    raise ValueError(f"incomplete expert output {template}")
                shape, dtype = tensors[0].shape, tensors[0].dtype
                if any(tensor.shape != shape or tensor.dtype != dtype for tensor in tensors):
                    raise ValueError(f"inconsistent expert output {template}")
                # Stack bytes: torch does not implement stack/cat for every
                # checkpoint scale dtype, in particular E8M0.
                packed = torch.stack([tensor.contiguous().view(torch.uint8) for tensor in tensors])
                count = len(tensors)
                yield MegatronNCCLM2NWeight(
                    name=template,
                    tensor=packed.view(dtype).reshape(count, *shape),
                    global_shape=torch.Size((self.num_experts, *shape)),
                    destination_shard_dim=None,
                    source_shard_dim=None,
                    source_shard_rank=0,
                    source_shard_size=1,
                    expert_ids=tuple(range(self.ep_rank * count, (self.ep_rank + 1) * count)),
                )

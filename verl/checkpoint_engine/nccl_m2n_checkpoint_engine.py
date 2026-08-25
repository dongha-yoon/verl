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

"""NCCL M2N checkpoint backend for layout-aware weight redistribution."""

from __future__ import annotations

import atexit
import logging
import math
import os
import socket
from dataclasses import dataclass, replace
from typing import Any, AsyncGenerator, Generator

import ray
import torch
import zmq

try:
    from nccl.core import Communicator, UniqueId, get_unique_id
    from nccl.core.interop.torch import empty as nccl_empty
    from nccl.m2n import DistTensor, Handle, Mesh, Replicate, Shard

    _NCCL_IMPORT_ERROR = None
except ImportError as exc:  # Optional experimental dependency.
    _NCCL_IMPORT_ERROR = exc

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.checkpoint_engine.reshard_layout import (
    LocalWeightDesc,
    ReshardLayout,
    build_reshard_layouts,
    local_weight_desc_from_shard_api,
)
from verl.models.transformers.hf_dense_decoder_tp import infer_dense_decoder_tp_shard_dim
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _require_nccl() -> None:
    if _NCCL_IMPORT_ERROR is not None:
        raise ImportError(
            "checkpoint_engine.backend='nccl_m2n' requires nccl-extensions==0.1.0; "
            "install `verl[nccl_m2n]` when CUDA Python and NCCL are provided by the environment, "
            "or install `nccl-extensions[cu12]==0.1.0` or `nccl-extensions[cu13]==0.1.0` "
            "to pull the matching CUDA runtime"
        ) from _NCCL_IMPORT_ERROR


@dataclass(frozen=True)
class NCCLM2NMasterMetadata:
    """Bootstrap data created by the trainer master.

    Attributes:
        unique_id: Serialized NCCL unique ID shared by all source and destination ranks.
        zmq_ip: IP address of the master's weight-metadata publisher.
        zmq_port: TCP port of the master's weight-metadata publisher.
        source_dp: Size of the replicated dimension of the source mesh.
        source_shard_size: Size of the sharded dimension of the source mesh.
        destination_dp: Size of the replicated dimension of the destination mesh.
        destination_shard_size: Size of the sharded dimension of the destination mesh.
        source_ep: Expert-parallel size of the source stage.
        source_etp: Expert tensor-parallel size of the source stage.
        destination_ep: Expert-parallel size of the destination mesh.
    """

    unique_id: bytes
    zmq_ip: str
    zmq_port: int
    source_dp: int
    source_shard_size: int
    destination_dp: int
    destination_shard_size: int
    source_ep: int = 1
    source_etp: int = 1
    destination_ep: int = 1


@dataclass(frozen=True)
class NCCLM2NLocalWeight(LocalWeightDesc):
    """Rank-local tensor with optional MoE ownership metadata.

    ``source_shard_rank`` records the local lane within the source tensor-
    parallel dimension. ``expert_ids`` identifies the experts stacked on this
    source rank; when present, ``name`` is a format string containing one
    integer placeholder for the destination expert ID.
    """

    source_shard_rank: int = 0
    expert_ids: tuple[int, ...] | None = None


def _window_view(window: torch.Tensor, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Return a typed tensor view backed by the shared byte window."""

    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes > window.numel():
        raise ValueError(f"local tensor needs {nbytes} bytes but the NCCL M2N window has {window.numel()}")
    return window[:nbytes].view(dtype).view(shape)


def _placement_objects(layout: ReshardLayout) -> list[Any]:
    """Translate a reshard layout into NCCL M2N placement objects."""

    return [Replicate() if dim is None else Shard(dim) for dim in layout.placements]


def _nccl_stream(stream: Any) -> Any:
    """Use the raw handle for torch streams that lack ``__cuda_stream__``."""

    # NCCL4Py accepts integer handles on every supported release. Torch 2.9
    # exposes ``cuda_stream`` but no longer implements the older protocol that
    # NCCL4Py otherwise probes for non-cuda.core stream objects.
    raw_stream = getattr(stream, "cuda_stream", None)
    return int(raw_stream) if raw_stream is not None else stream


def _allocate_destination(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Allocate one destination rank's output tensor on its current CUDA device.

    Args:
        shape: Shape of the destination rank-local tensor.
        dtype: Data type of the destination rank-local tensor.

    Returns:
        A newly allocated CUDA tensor with ``shape`` and ``dtype``.
    """

    device = torch.device("cuda", torch.cuda.current_device())
    return torch.empty(shape, dtype=dtype, device=device)


class _NCCLM2NChannel(CheckpointEngine):
    """Transfer source shards directly into destination rank-local shards.

    Source and destination ranks form separate two-dimensional meshes with shapes
    ``(source_dp, source_shard_size)`` and
    ``(destination_dp, destination_shard_size)``. The first dimension contains
    replicated model copies; the second is the rank-local sharding dimension.

    Args:
        bucket_size: Common checkpoint-engine bucket size in bytes. NCCL M2N
            manages its staging pool internally, so this value is retained only
            for checkpoint-engine interface compatibility.
        is_master: Whether this source rank creates the NCCL unique ID and publishes
            per-weight metadata. Exactly one source rank must be the master.
        source_dp: Number of replicated groups in the source mesh.
        source_shard_size: Number of ranks in the source mesh's sharded dimension.
        destination_dp: Number of replicated groups in the destination mesh.
        destination_shard_size: Number of ranks in the destination mesh's generic
            sharded dimension. A consumer such as vLLM may interpret this as TP.
    """

    wire_format = "rank_local_named_tensors"
    topic = "nccl_m2n_metadata"
    ready_topic_prefix = "nccl_m2n_ready:"

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        source_dp: int | None = None,
        source_shard_size: int | None = None,
        destination_dp: int | None = None,
        destination_shard_size: int | None = None,
    ) -> None:
        topology = {
            "source_dp": source_dp,
            "source_shard_size": source_shard_size,
            "destination_dp": destination_dp,
            "destination_shard_size": destination_shard_size,
        }
        missing = [name for name, value in topology.items() if value is None]
        if missing:
            raise ValueError(f"NCCL M2N topology requires explicit values for {', '.join(missing)}")

        self.bucket_size = int(bucket_size)
        self.is_master = bool(is_master)
        self.source_dp = int(source_dp)
        self.source_shard_size = int(source_shard_size)
        self.destination_dp = int(destination_dp)
        self.destination_shard_size = int(destination_shard_size)
        if (
            min(
                self.bucket_size,
                self.source_dp,
                self.source_shard_size,
                self.destination_dp,
                self.destination_shard_size,
            )
            <= 0
        ):
            raise ValueError("NCCL M2N sizes must be positive")

        self.source_world_size = self.source_dp * self.source_shard_size
        self.reshard_world_size = self.source_world_size + self.destination_dp * self.destination_shard_size
        self.rank: int | None = None
        self.role: str | None = None
        self._master_metadata: NCCLM2NMasterMetadata | None = None
        self._comm = self._handle = self._window = self._window_tensor = self._transfer_stream = None
        self._socket = self._zmq_context = None
        self._closed = False
        if self.is_master:
            self._start_metadata_server()
        atexit.register(self.close)

    def _start_metadata_server(self) -> None:
        ip = ray.util.get_node_ip_address().strip("[]")
        port, _ = get_free_port(ip)
        context = zmq.Context()
        # XPUB exposes subscription notifications, allowing rank zero to wait
        # until every destination's metadata subscription has reached this socket.
        socket = context.socket(zmq.XPUB)
        socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        address = f"tcp://[{ip}]:{port}" if is_valid_ipv6_address(ip) else f"tcp://{ip}:{port}"
        if is_valid_ipv6_address(ip):
            socket.setsockopt(zmq.IPV6, 1)
        socket.bind(address)
        self._zmq_context, self._socket = context, socket
        self._metadata_ip, self._metadata_port = ip, port

    def _connect_metadata_client(self, metadata: NCCLM2NMasterMetadata) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        address = (
            f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            if is_valid_ipv6_address(metadata.zmq_ip)
            else f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"
        )
        if is_valid_ipv6_address(metadata.zmq_ip):
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(address)
        # Subscribe to the common topic first. XPUB observes subscriptions from
        # one connection in order, so seeing the rank-specific readiness topic
        # proves that the common subscription is active at the publisher.
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        # ZMQ matches prefixes; the trailing separator keeps rank 8 from matching rank 80.
        socket.setsockopt_string(zmq.SUBSCRIBE, f"{self.ready_topic_prefix}{self.rank}:")
        self._zmq_context, self._socket = context, socket

    def _wait_for_metadata_subscribers(self) -> None:
        """Wait for every destination subscription and acknowledge each one."""

        expected = set(range(self.source_world_size, self.reshard_world_size))
        ready: set[int] = set()
        prefix = self.ready_topic_prefix.encode()
        while ready != expected:
            event = self._socket.recv()
            if not event:
                raise RuntimeError("received an empty NCCL M2N subscription event")
            subscription = event[1:]
            if not subscription.startswith(prefix):
                continue
            try:
                if not subscription.endswith(b":"):
                    raise ValueError("missing readiness topic terminator")
                rank = int(subscription[len(prefix) : -1].decode("ascii"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid NCCL M2N readiness subscription: {subscription!r}") from exc
            if rank not in expected:
                raise RuntimeError(f"unexpected NCCL M2N destination readiness rank {rank}")
            if event[0] == 1:
                ready.add(rank)
            elif event[0] == 0:
                ready.discard(rank)
            else:
                raise RuntimeError(f"invalid NCCL M2N subscription action {event[0]}")
        for rank in sorted(expected):
            self._socket.send_string(f"{self.ready_topic_prefix}{rank}:")

    def _wait_for_metadata_publisher(self) -> None:
        """Wait until rank zero acknowledges this destination's subscriptions."""

        expected = f"{self.ready_topic_prefix}{self.rank}:"
        received = self._socket.recv_string()
        if received != expected:
            raise RuntimeError(f"unexpected NCCL M2N readiness acknowledgement {received!r}; expected {expected!r}")

    def prepare(self) -> NCCLM2NMasterMetadata | None:
        """Create the communicator bootstrap metadata on the master source rank.

        Returns:
            Cached bootstrap metadata on the master, or ``None`` on every other rank.
        """

        if not self.is_master:
            return None
        _require_nccl()
        if self._master_metadata is None:
            self._master_metadata = NCCLM2NMasterMetadata(
                bytes(get_unique_id()),
                self._metadata_ip,
                self._metadata_port,
                self.source_dp,
                self.source_shard_size,
                self.destination_dp,
                self.destination_shard_size,
            )
        return self._master_metadata

    @classmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[Any]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build per-worker communicator arguments for source and destination ranks.

        Args:
            trainer_world_size: Number of trainer workers in the source mesh.
            rollout_world_size: Number of rollout workers in the destination mesh.
            metadata: Results of ``prepare`` ordered with trainer workers first.

        Returns:
            Trainer and rollout keyword-argument maps for ``init_process_group``.
            Every value is a per-worker list suitable for worker-group dispatch.

        Raises:
            ValueError: If there is not exactly one trainer master or either worker
                count does not match the configured mesh dimensions.
        """

        masters = [item for item in metadata[:trainer_world_size] if isinstance(item, NCCLM2NMasterMetadata)]
        if len(masters) != 1:
            raise ValueError(f"NCCL M2N requires exactly one trainer master, got {len(masters)}")
        master = masters[0]
        source_size = master.source_dp * master.source_shard_size
        destination_size = master.destination_dp * master.destination_shard_size
        world_size = source_size + destination_size
        if trainer_world_size != source_size or rollout_world_size != destination_size:
            raise ValueError(
                f"NCCL M2N requires exactly {source_size} trainer and {destination_size} rollout ranks, "
                f"got {trainer_world_size} and {rollout_world_size}"
            )
        return (
            {
                "rank": list(range(source_size)),
                "world_size": [world_size] * trainer_world_size,
                "master_metadata": [master] * trainer_world_size,
                "role": ["source"] * source_size,
            },
            {
                "rank": list(range(source_size, world_size)),
                "world_size": [world_size] * rollout_world_size,
                "master_metadata": [master] * rollout_world_size,
                "role": ["destination"] * rollout_world_size,
            },
        )

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: NCCLM2NMasterMetadata,
        role: str,
        shared_tensor: torch.Tensor | None = None,
    ) -> None:
        """Create the combined NCCL M2N communicator and runtime handle.

        Args:
            rank: This worker's rank in the source-plus-destination communicator.
            world_size: Total number of source and destination ranks.
            master_metadata: Bootstrap and mesh metadata returned by the master.
            role: This worker's role, either ``"source"`` or ``"destination"``.

        Raises:
            ValueError: If the rank, role, world size, or topology is invalid.
            RuntimeError: If an existing communicator has a different topology or
                communicator bootstrap fails.
        """

        self.rank, self.role = int(rank), role
        if self.rank < 0:
            raise ValueError(f"NCCL M2N rank must be non-negative, got {self.rank}")
        if role not in {"source", "destination"}:
            raise ValueError(f"NCCL M2N role must be source or destination, got {role!r}")

        _require_nccl()
        if world_size != self.reshard_world_size:
            raise ValueError(f"runtime world_size={world_size}, expected {self.reshard_world_size}")
        expected = (self.source_dp, self.source_shard_size, self.destination_dp, self.destination_shard_size)
        actual = (
            master_metadata.source_dp,
            master_metadata.source_shard_size,
            master_metadata.destination_dp,
            master_metadata.destination_shard_size,
        )
        if actual != expected:
            raise ValueError(f"NCCL M2N topology {actual} does not match local config {expected}")
        if self._comm is not None:
            if self._comm.rank != self.rank or self._comm.nranks != world_size:
                raise RuntimeError("cannot reuse an NCCL M2N communicator with a different topology")
            return

        self._master_metadata = master_metadata
        self._comm = Communicator.init(world_size, self.rank, UniqueId.from_bytes(master_metadata.unique_id))
        self._handle = Handle.create()
        self._window_tensor = shared_tensor
        if self._window_tensor is not None:
            self._window = self._comm.register_window(self._window_tensor)
            if self._window is None:
                raise RuntimeError("NCCL M2N window registration failed")
        self._transfer_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        if role == "destination":
            self._connect_metadata_client(master_metadata)

        if self.rank == 0:
            # Avoid a host NCCL collective here: every PP channel would retain
            # bootstrap P2P buffers. Rank-specific XPUB acknowledgements prove
            # that each destination is ready to receive metadata instead.
            self._wait_for_metadata_subscribers()
        elif self.role == "destination":
            self._wait_for_metadata_publisher()

    def _coerce_weight(self, exported: Any) -> NCCLM2NLocalWeight:
        if isinstance(exported, NCCLM2NLocalWeight):
            return exported

        if isinstance(exported, LocalWeightDesc):
            return NCCLM2NLocalWeight(**vars(exported))

        weight = local_weight_desc_from_shard_api(
            exported, destination_shard_dim=infer_dense_decoder_tp_shard_dim(exported[0])
        )
        source_shard_rank = 0
        if self.rank is not None and weight.source_shard_dim is not None:
            spec = exported[2]
            mesh_dim = next(axis for axis, placement in enumerate(spec.placements) if placement.is_shard())
            device_mesh_rank = int(spec.mesh.get_local_rank(mesh_dim=mesh_dim))
            m2n_mesh_rank = self.rank % self.source_shard_size
            if device_mesh_rank != m2n_mesh_rank:
                raise ValueError(
                    f"source DeviceMesh rank {device_mesh_rank} does not match M2N mesh rank {m2n_mesh_rank}"
                )
            source_shard_rank = device_mesh_rank
        return NCCLM2NLocalWeight(**vars(weight), source_shard_rank=source_shard_rank)

    def _layouts(self, weight: LocalWeightDesc) -> tuple[ReshardLayout, ReshardLayout]:
        if self._packed(weight):
            _, stride, source_rows, destination_rows = self._wire_geometry(weight)
            metadata = self._master_metadata
            source_dims = (self.source_world_size, 1)
            destination_dims = (self.reshard_world_size - self.source_world_size, 1)
            if weight.expert_ids is not None:
                source_dims = (1, metadata.source_ep * metadata.source_etp)
                destination_dims = (
                    destination_dims[0] // metadata.destination_ep,
                    metadata.destination_ep,
                )
            return (
                ReshardLayout(source_dims, 0, (None, 0), torch.Size((source_rows * stride,))),
                ReshardLayout(
                    destination_dims,
                    self.source_world_size,
                    (None, 0),
                    torch.Size((destination_rows * stride,)),
                ),
            )
        return build_reshard_layouts(
            weight,
            source_replica_size=self.source_dp,
            source_shard_size=self.source_shard_size,
            destination_replica_size=self.destination_dp,
            destination_shard_size=self.destination_shard_size,
        )

    def _packed(self, weight: NCCLM2NLocalWeight) -> bool:
        """Whether this channel uses byte-packed rows in its registered window."""

        return self._window_tensor is not None and (
            weight.expert_ids is not None or (weight.source_shard_dim is None and weight.destination_shard_dim is None)
        )

    def _wire_geometry(self, weight: NCCLM2NLocalWeight) -> tuple[int, int, int, int]:
        """Return payload bytes, padded row stride, source rows, and destination rows."""

        shape = weight.tensor.shape
        source_rows = destination_rows = 1
        if weight.expert_ids is not None:
            metadata = self._master_metadata
            if metadata is None:
                raise RuntimeError("NCCL M2N metadata is unavailable")
            expert_count = weight.global_shape[0]
            if expert_count <= 0 or expert_count % metadata.source_ep or expert_count % metadata.destination_ep:
                raise ValueError(
                    f"expert count {expert_count} must divide source EP {metadata.source_ep} "
                    f"and destination EP {metadata.destination_ep}"
                )
            source_rows = expert_count // metadata.source_ep
            destination_rows = expert_count // metadata.destination_ep * metadata.source_etp
            if (
                shape[0] != source_rows
                or len(weight.expert_ids) != source_rows
                or weight.source_shard_size != metadata.source_etp
            ):
                raise ValueError(f"incorrect local expert stack: {weight.name}")
            full_shape = list(shape)
            full_shape[0] = expert_count
            if weight.source_shard_dim is not None:
                if weight.source_shard_dim not in range(1, len(shape)):
                    raise ValueError(f"invalid expert tensor-parallel axis: {weight.name}")
                full_shape[weight.source_shard_dim] *= metadata.source_etp
            elif metadata.source_etp != 1:
                raise ValueError("ETP-sharded experts require a source tensor shard axis")
            if tuple(full_shape) != tuple(weight.global_shape):
                raise ValueError(f"inconsistent global expert shape: {weight.name}")
            shape = shape[1:]

        payload = math.prod(shape) * weight.tensor.element_size()
        ctas = int(os.environ.get("NCCLM2N_NUM_CTAS", os.environ.get("NCCL_RESHARD_NUM_CTAS", "8")))
        if payload <= 0 or ctas <= 0:
            raise ValueError("NCCL M2N payload and CTA count must be positive")
        # Every independently routed row has a stripe for every RING CTA.
        stride = (payload + ctas - 1) // ctas * ctas
        if max(source_rows, destination_rows) * stride > self.bucket_size:
            raise ValueError(f"tensor family exceeds the NCCL M2N window: {weight.name}")
        return payload, stride, source_rows, destination_rows

    def _validate_owner(self, weight: NCCLM2NLocalWeight) -> None:
        if weight.expert_ids is not None:
            metadata = self._master_metadata
            if metadata is None:
                raise RuntimeError("NCCL M2N metadata is unavailable")
            count = len(weight.expert_ids)
            first = self.rank // metadata.source_etp * count
            if weight.expert_ids != tuple(range(first, first + count)):
                raise ValueError(f"expert ownership does not match communicator rank: {weight.name}")
            if weight.source_shard_rank != self.rank % metadata.source_etp:
                raise ValueError(f"expert tensor-parallel lane does not match communicator rank: {weight.name}")
        elif weight.source_shard_dim is not None and weight.source_shard_rank != self.rank % self.source_shard_size:
            raise ValueError(f"tensor-parallel lane does not match communicator rank: {weight.name}")

    def _descriptors(
        self,
        weight: LocalWeightDesc,
        source: torch.Tensor | None,
        destination: torch.Tensor | None,
    ) -> tuple[Any, Any]:
        source_layout, destination_layout = self._layouts(weight)
        dtype = torch.uint8 if self._packed(weight) else weight.tensor.dtype
        return (
            DistTensor(
                source,
                local_shape=source_layout.local_shape,
                dtype=dtype,
                mesh=Mesh(source_layout.mesh_dims, start_rank=source_layout.start_rank),
                placements=_placement_objects(source_layout),
            ),
            DistTensor(
                destination,
                local_shape=destination_layout.local_shape,
                dtype=dtype,
                mesh=Mesh(destination_layout.mesh_dims, start_rank=destination_layout.start_rank),
                placements=_placement_objects(destination_layout),
            ),
        )

    def _publish(self, payload: dict[str, Any]) -> None:
        self._socket.send_string(self.topic, flags=zmq.SNDMORE)
        self._socket.send_pyobj(payload)

    def _receive(self) -> dict[str, Any]:
        if self._socket.recv_string() != self.topic:
            raise RuntimeError("received an unexpected NCCL M2N metadata topic")
        return self._socket.recv_pyobj()

    @torch.no_grad()
    async def send_weights(self, weights: Generator, global_steps: int | None = None) -> dict:
        """Reshard local source tensors into destination rank-local tensors.

        Args:
            weights: Generator yielding ``LocalWeightDesc`` objects or
                ``(name, local_tensor, ShardSpec)`` tuples.
            global_steps: Optional trainer step associated with the update. Reserved
                for checkpoint-engine interface compatibility.

        Returns:
            An empty metrics dictionary after all transfers have been enqueued.
            ``finalize`` waits for their completion.
        """

        del global_steps
        if self.rank is None:
            raise RuntimeError("NCCL M2N process group is not initialized")
        if self.role != "source" or any(resource is None for resource in (self._comm, self._handle)):
            raise RuntimeError(f"invalid NCCL M2N sender state for role={self.role!r}")

        caller_stream = torch.cuda.current_stream()
        stream = self._transfer_stream
        if stream is None:
            raise RuntimeError("NCCL M2N transfer stream is not initialized")
        for exported in weights:
            weight = self._coerce_weight(exported)
            self._validate_owner(weight)
            stream.wait_stream(caller_stream)
            packed = self._packed(weight)
            source = weight.tensor
            if self._window_tensor is not None:
                source_layout, _ = self._layouts(weight)
                source = _window_view(
                    self._window_tensor,
                    source_layout.local_shape,
                    torch.uint8 if packed else weight.tensor.dtype,
                )
                with torch.cuda.stream(stream):
                    if packed:
                        payload, stride, source_rows, _ = self._wire_geometry(weight)
                        rows = source.view(source_rows, stride)
                        rows[:, :payload].copy_(
                            weight.tensor.contiguous().view(torch.uint8).reshape(source_rows, payload)
                        )
                        rows[:, payload:].zero_()
                    else:
                        source.copy_(weight.tensor, non_blocking=True)
                if weight.tensor.is_cuda:
                    weight.tensor.record_stream(stream)
            if self.rank == 0:
                self._publish(
                    {
                        "kind": "weight",
                        "name": weight.name,
                        "global_shape": tuple(weight.global_shape),
                        "dtype": weight.tensor.dtype,
                        "destination_shard_dim": weight.destination_shard_dim,
                        "source_shard_dim": weight.source_shard_dim,
                        "source_shard_rank": weight.source_shard_rank,
                        "source_shard_size": weight.source_shard_size,
                        "local_shape": tuple(weight.tensor.shape),
                        "expert_ids": weight.expert_ids,
                    }
                )
            source_desc, destination_desc = self._descriptors(weight, source, None)
            if self._window is None:
                # The simple PP1/EP1 path retains the direct API and lets M2N own
                # any internal staging needed by Handle.reshard().
                self._handle.reshard(
                    self._comm,
                    source_desc,
                    destination_desc,
                    stream=_nccl_stream(stream),
                )
                caller_stream.wait_stream(stream)
            else:
                self._handle.reshard_with_window(
                    self._comm,
                    self._window,
                    source_desc,
                    destination_desc,
                    stream=_nccl_stream(stream),
                )
                # One byte window is reused for all families in this channel.
                stream.synchronize()
        if self.rank == 0:
            self._publish({"kind": "end"})
        # finalize() fences every matching asynchronous M2N operation. The
        # handle-owned staging pool remains alive until close() destroys the handle.
        return {}

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive destination rank-local tensors in source publication order.

        Args:
            global_steps: Optional trainer step associated with the update. Reserved
                for checkpoint-engine interface compatibility.

        Yields:
            The parameter name and its newly allocated destination rank-local tensor.
        """

        del global_steps
        if self.rank is None or self.rank < 0 or self.role != "destination":
            raise RuntimeError(f"invalid NCCL M2N receiver state: rank={self.rank}, role={self.role!r}")
        if any(resource is None for resource in (self._comm, self._handle)):
            raise RuntimeError("NCCL M2N receiver resources are not initialized")

        caller_stream = torch.cuda.current_stream()
        stream = self._transfer_stream
        if stream is None:
            raise RuntimeError("NCCL M2N transfer stream is not initialized")
        while True:
            metadata = self._receive()
            if metadata.get("kind") == "end":
                break
            if metadata.get("kind") != "weight":
                raise RuntimeError(f"invalid NCCL M2N metadata: {metadata!r}")

            local_shape = metadata.get("local_shape", metadata["global_shape"])
            weight = NCCLM2NLocalWeight(
                name=metadata["name"],
                tensor=torch.empty(local_shape, dtype=metadata["dtype"], device="meta"),
                global_shape=torch.Size(metadata["global_shape"]),
                destination_shard_dim=metadata["destination_shard_dim"],
                source_shard_dim=metadata["source_shard_dim"],
                source_shard_size=int(metadata["source_shard_size"]),
                source_shard_rank=int(metadata.get("source_shard_rank", 0)),
                expert_ids=metadata.get("expert_ids"),
            )
            _, destination_layout = self._layouts(weight)
            packed = self._packed(weight)
            destination = (
                _allocate_destination(destination_layout.local_shape, weight.tensor.dtype)
                if self._window_tensor is None
                else _window_view(
                    self._window_tensor,
                    destination_layout.local_shape,
                    torch.uint8 if packed else weight.tensor.dtype,
                )
            )
            source_desc, destination_desc = self._descriptors(weight, None, destination)
            stream.wait_stream(caller_stream)
            if self._window is None:
                self._handle.reshard(
                    self._comm,
                    source_desc,
                    destination_desc,
                    stream=_nccl_stream(stream),
                )
            else:
                self._handle.reshard_with_window(
                    self._comm,
                    self._window,
                    source_desc,
                    destination_desc,
                    stream=_nccl_stream(stream),
                )
            caller_stream.wait_stream(stream)
            if not packed:
                yield weight.name, destination
                continue

            payload, stride, _, destination_rows = self._wire_geometry(weight)
            rows = destination.view(destination_rows, stride)
            if weight.expert_ids is None:
                yield weight.name, rows[0, :payload].view(weight.tensor.dtype).view(weight.tensor.shape)
                continue

            metadata = self._master_metadata
            local_count = weight.global_shape[0] // metadata.destination_ep
            first = (self.rank - self.source_world_size) % metadata.destination_ep * local_count
            chunks = rows.view(metadata.source_etp, local_count, stride)
            for index in range(local_count):
                pieces = [
                    chunk[index, :payload].view(weight.tensor.dtype).view(weight.tensor.shape[1:]) for chunk in chunks
                ]
                tensor = (
                    pieces[0]
                    if metadata.source_etp == 1
                    else torch.cat(
                        [piece.contiguous().view(torch.uint8) for piece in pieces],
                        dim=weight.source_shard_dim - 1,
                    ).view(weight.tensor.dtype)
                )
                yield weight.name.format(first + index), tensor

    def finalize(self) -> None:
        """Host-wait for outstanding transfer and consumer work after an update."""

        if self.rank is not None and self.rank >= 0:
            if self._transfer_stream is not None:
                self._transfer_stream.synchronize()
            torch.cuda.current_stream().synchronize()

    def close(self) -> None:
        """Wait for M2N work and release the handle, communicator, and ZMQ resources."""

        if self._closed:
            return
        self._closed = True
        try:
            if self._transfer_stream is not None:
                self._transfer_stream.synchronize()
            if self._handle is not None:
                self._handle.destroy()
            if self._comm is not None:
                self._comm.destroy()
            if self._socket is not None:
                self._socket.close(linger=0)
            if self._zmq_context is not None:
                self._zmq_context.term()
        except Exception:
            logger.exception("failed to close NCCL M2N resources")
        finally:
            self._comm = self._handle = self._window = self._window_tensor = self._transfer_stream = None
            self._socket = self._zmq_context = None


def _close_channels(channels: list[_NCCLM2NChannel]) -> None:
    """Release several M2N channels without invalidating a live communicator.

    The final M2N handle clears process-global device-communicator state, so all
    handles must be destroyed while every owning host communicator is alive.
    """

    for channel in channels:
        channel.finalize()
    for channel in channels:
        if channel._handle is not None:
            channel._handle.destroy()
            channel._handle = None
    for channel in channels:
        channel.close()


@CheckpointEngineRegistry.register("nccl_m2n")
class NCCLM2NCheckpointEngine(_NCCLM2NChannel):
    """NCCL M2N backend with one shared transfer window across PP stages."""

    def __init__(
        self,
        *args,
        source_layout: dict[str, Any] | None = None,
        destination_expert_parallel_size: int = 1,
        gpus_per_node: int = 8,
        derive_topology: bool = False,
        **kwargs,
    ) -> None:
        self.source_layout = source_layout
        self.destination_ep = int(destination_expert_parallel_size)
        self.gpus_per_node = int(gpus_per_node)
        self.derive_topology = bool(derive_topology)
        self._channels: list[_NCCLM2NChannel] = []
        self._source_channel: _NCCLM2NChannel | None = None
        self._parallel = False
        if min(self.destination_ep, self.gpus_per_node) <= 0:
            raise ValueError("expert parallel size and GPUs per node must be positive")

        # Parallel topology is derived later from per-rank prepare() metadata.
        # Placeholder sizes let actor and rollout workers construct the backend
        # before CheckpointEngineManager has collected that metadata.
        parallel_topology = source_layout is not None or self.derive_topology
        if parallel_topology:
            for key in ("source_dp", "source_shard_size", "destination_dp", "destination_shard_size"):
                kwargs.setdefault(key, 1)
        super().__init__(*args, **kwargs)

    @classmethod
    def is_master_rank(cls, trainer_rank: int, **engine_kwargs) -> bool:
        layout = engine_kwargs.get("source_layout")
        if layout is not None:
            return layout["edp"] == layout["ep"] == layout["etp"] == 0
        return super().is_master_rank(trainer_rank, **engine_kwargs)

    def prepare(self) -> NCCLM2NMasterMetadata | dict[str, Any] | None:
        layout = self.source_layout
        if layout is not None and (layout["pp_size"] > 1 or layout["ep_size"] > 1):
            return {
                "layout": layout,
                "host": socket.gethostname(),
                "gpus_per_node": self.gpus_per_node,
                "master": super().prepare(),
            }
        if layout is None and not self.is_master and self.derive_topology:
            return {
                "host": socket.gethostname(),
                "tp_size": self.destination_shard_size,
                "ep_size": self.destination_ep,
            }
        return super().prepare()

    @classmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[Any]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        sources = metadata[:trainer_world_size]
        destinations = metadata[trainer_world_size:]
        if not any(isinstance(item, dict) and "layout" in item for item in sources):
            return super().build_topology(trainer_world_size, rollout_world_size, metadata)
        if len(destinations) != rollout_world_size or not all(isinstance(item, dict) for item in metadata):
            raise ValueError("incomplete NCCL M2N rank metadata")

        layout = sources[0]["layout"]
        pp, ep, etp, tp = (int(layout[key]) for key in ("pp_size", "ep_size", "etp_size", "tp_size"))
        destination_ep = int(destinations[0]["ep_size"])
        destination_tp = int(destinations[0]["tp_size"])
        node_size = int(sources[0]["gpus_per_node"])
        source_width = ep * etp
        if (
            min(pp, ep, etp, tp, destination_ep, destination_tp, node_size) <= 0
            or source_width % tp
            or rollout_world_size % destination_ep
            or rollout_world_size % destination_tp
        ):
            raise ValueError("incompatible NCCL M2N source/destination parallel sizes")
        if etp > 1 and ep != destination_ep:
            raise NotImplementedError("changing EP with ETP > 1 requires a different expert packing order")
        if any(
            (int(item["ep_size"]), int(item["tp_size"])) != (destination_ep, destination_tp) for item in destinations
        ):
            raise ValueError("rollout ranks disagree on TP/EP geometry")
        if any(
            tuple(int(item["layout"][key]) for key in ("pp_size", "ep_size", "etp_size", "tp_size"))
            != (pp, ep, etp, tp)
            or int(item["gpus_per_node"]) != node_size
            for item in sources
        ):
            raise ValueError("trainer ranks disagree on parallel geometry")

        def _node_hosts(group: list[dict[str, Any]]) -> set[str]:
            if len(group) % node_size:
                raise ValueError("NCCL M2N participants must occupy complete physical nodes")
            hosts: list[str] = []
            for start in range(0, len(group), node_size):
                chunk = {item["host"] for item in group[start : start + node_size]}
                if len(chunk) != 1:
                    raise ValueError("NCCL M2N ranks must be contiguous within each physical node")
                hosts.append(chunk.pop())
            if len(hosts) != len(set(hosts)):
                raise ValueError("a physical node occurs in multiple NCCL M2N rank blocks")
            return set(hosts)

        destination_hosts = _node_hosts(destinations)
        masters: list[NCCLM2NMasterMetadata] = []
        for stage in range(pp):
            group = sorted(
                (item for item in sources if item["layout"]["pp"] == stage and item["layout"]["edp"] == 0),
                key=lambda item: (item["layout"]["ep"], item["layout"]["etp"]),
            )
            lanes = [(item["layout"]["ep"], item["layout"]["etp"]) for item in group]
            if lanes != [(expert, expert_tp) for expert in range(ep) for expert_tp in range(etp)]:
                raise ValueError(f"incomplete EP/ETP owner group for PP{stage}")
            if any(item["layout"]["tp"] != rank % tp for rank, item in enumerate(group)):
                raise ValueError("selected EP/ETP owners do not form complete dense TP replicas")
            if _node_hosts(group) & destination_hosts:
                raise ValueError("trainer and rollout NCCL M2N groups must occupy separate physical nodes")
            master = group[0]["master"]
            if not isinstance(master, NCCLM2NMasterMetadata):
                raise ValueError(f"missing NCCL M2N metadata master for PP{stage}")
            masters.append(
                replace(
                    master,
                    source_dp=source_width // tp,
                    source_shard_size=tp,
                    destination_dp=rollout_world_size // destination_tp,
                    destination_shard_size=destination_tp,
                    source_ep=ep,
                    source_etp=etp,
                    destination_ep=destination_ep,
                )
            )

        return (
            {
                "role": ["source" if item["layout"]["edp"] == 0 else "inactive" for item in sources],
                "rank": [item["layout"]["ep"] * etp + item["layout"]["etp"] for item in sources],
                "stage": [item["layout"]["pp"] for item in sources],
                "masters": [masters] * trainer_world_size,
            },
            {
                "role": ["destination"] * rollout_world_size,
                "rank": list(range(source_width, source_width + rollout_world_size)),
                "stage": [None] * rollout_world_size,
                "masters": [masters] * rollout_world_size,
            },
        )

    def init_process_group(
        self,
        rank: int,
        world_size: int | None = None,
        master_metadata: NCCLM2NMasterMetadata | None = None,
        role: str | None = None,
        *,
        masters: list[NCCLM2NMasterMetadata] | None = None,
        stage: int | None = None,
    ) -> None:
        if masters is None:
            if world_size is None or master_metadata is None or role is None:
                raise ValueError("simple NCCL M2N topology requires world_size, master_metadata, and role")
            return super().init_process_group(rank, world_size, master_metadata, role)

        self._parallel = True
        self.rank, self.role = int(rank), role
        if role == "inactive" or self._channels:
            return
        if role not in {"source", "destination"}:
            raise ValueError(f"invalid NCCL M2N role {role!r}")
        _require_nccl()
        self._window_tensor = nccl_empty(
            self.bucket_size,
            dtype=torch.uint8,
            device=torch.cuda.current_device(),
        )
        selected = [masters[stage]] if role == "source" else masters
        for master in selected:
            channel = _NCCLM2NChannel(
                self.bucket_size,
                source_dp=master.source_dp,
                source_shard_size=master.source_shard_size,
                destination_dp=master.destination_dp,
                destination_shard_size=master.destination_shard_size,
            )
            self._channels.append(channel)
            # Register before initialization so a partially initialized channel
            # is still closed if communicator creation raises.
            atexit.register(self.close)
            if role == "source":
                self._source_channel = channel
                channel._zmq_context, channel._socket = self._zmq_context, self._socket
                self._zmq_context = self._socket = None
            channel.init_process_group(
                rank,
                channel.reshard_world_size,
                master,
                role,
                shared_tensor=self._window_tensor,
            )

    async def send_weights(self, weights: Generator, global_steps: int | None = None) -> dict:
        if not self._parallel:
            return await super().send_weights(weights, global_steps)
        if self.role == "inactive":
            return {}
        return await self._source_channel.send_weights(weights, global_steps)

    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        if not self._parallel:
            async for item in super().receive_weights(global_steps):
                yield item
            return
        for channel in self._channels:
            async for item in channel.receive_weights(global_steps):
                yield item

    def finalize(self) -> None:
        if self._parallel:
            for channel in self._channels:
                channel.finalize()
        else:
            super().finalize()

    def close(self) -> None:
        _close_channels(self._channels)
        self._channels.clear()
        super().close()

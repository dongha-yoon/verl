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

"""CPU regression for NCCL M2N metadata readiness using real ZMQ sockets."""

import zmq

from verl.checkpoint_engine.nccl_m2n_checkpoint_engine import NCCLM2NCheckpointEngine, NCCLM2NMasterMetadata


def test_readiness_ack_does_not_leak_into_metadata_with_real_zmq():
    # Destination ranks 1 through 10 include a prefix collision between 1 and 10.
    topology = dict(source_dp=1, source_shard_size=1, destination_dp=10, destination_shard_size=1)
    publisher = NCCLM2NCheckpointEngine(bucket_size=256, **topology)
    clients = []
    try:
        publisher._zmq_context = zmq.Context()
        publisher._socket = publisher._zmq_context.socket(zmq.XPUB)
        publisher._socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        publisher._socket.setsockopt(zmq.RCVTIMEO, 5000)
        publisher._socket.setsockopt(zmq.SNDTIMEO, 5000)
        port = publisher._socket.bind_to_random_port("tcp://127.0.0.1")
        metadata = NCCLM2NMasterMetadata(
            unique_id=b"test-id",
            zmq_ip="127.0.0.1",
            zmq_port=port,
            **topology,
        )
        for rank in range(1, 11):
            client = NCCLM2NCheckpointEngine(bucket_size=256, **topology)
            clients.append(client)
            client.rank = rank
            client._connect_metadata_client(metadata)
            client._socket.setsockopt(zmq.RCVTIMEO, 5000)

        publisher._wait_for_metadata_subscribers()
        for client in clients:
            client._wait_for_metadata_publisher()

        # The old protocol passes the handshake but leaves rank 10's ACK queued
        # at rank 1. Verify metadata delivery, including subsequent updates.
        for step in range(3):
            payload = {"kind": "weight", "name": "cpu_protocol_probe", "step": step}
            publisher._publish(payload)
            for client in clients:
                assert client._receive() == payload
    finally:
        for client in clients:
            client.close()
        publisher.close()

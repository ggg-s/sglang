"""Two P/two D local-file namespaces with actual TCP control replication.

Native Mooncake metadata serialization/parsing and replica resolution are loaded
from source. SMG selection, ZMQ delivery and native admission remain boundaries.
"""

import secrets
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dsv41_remote_probe_support import definitions
from sglang_remote_attn import pd_mesh
from sglang_remote_attn.pre_layerwise import read_json, write_json


class Cluster:
    def __init__(self, root, directory):
        self.meshes = {}
        for i, name in enumerate(("P0", "P1", "D0", "D1")):
            host = "127.0.0.1" if name.startswith("P") else "127.0.0.2"
            self.meshes[name] = pd_mesh.Mesh(
                Path(directory) / name,
                name,
                secrets.token_hex(16),
                0,
                host,
                host,
                (host, 32000 + i),
            )
        self.reader = definitions(
            root,
            "python/sglang/srt/disaggregation/mooncake/conn.py",
            owner="TransferInfo",
            members={"from_zmq"},
        ).from_zmq.__func__
        self.reader.__globals__.update(np=np, unpack_int_lists=lambda raw, fmt: [])
        self.builder = definitions(
            root,
            "python/sglang/srt/disaggregation/mooncake/conn.py",
            owner="MooncakeKVManager",
            members={"build_receiver_metadata_message"},
        ).build_receiver_metadata_message
        self.builder.__globals__["np"] = np
        self.resolve = definitions(
            root,
            "python/sglang/srt/disaggregation/common/conn.py",
            owner="CommonKVManager",
            members={"resolve_kv_replica_factor"},
        ).resolve_kv_replica_factor
        self.links = []

    def route(self, p_index, d_index, service, number):
        p, d = self.meshes[f"P{p_index}"], self.meshes[f"D{d_index}"]
        manager = SimpleNamespace(
            local_ip=d.source_endpoint[0],
            rank_port=d.source_endpoint[1],
            get_session_id=lambda: f"session-D{d_index}",
            _remote_attn_mesh=d,
        )
        receiver = SimpleNamespace(kv_mgr=manager, bootstrap_room=number)
        message = self.builder(
            manager,
            bootstrap_room=number,
            required_dst_info_num=1,
            kv_indices=np.array([0], dtype=np.int32),
            aux_index=0,
            state_indices=[],
            decode_prefix_len=0,
            is_dummy=False,
        )
        sent = []
        pd_mesh.bootstrap(
            lambda receiver, **kw: sent.append(kw["message"]) or True,
            receiver,
            bootstrap_info=dict(
                rank_ip=p.source_endpoint[0],
                rank_port=p.source_endpoint[1],
                is_dummy=False,
            ),
            message=message,
            operation="send_metadata",
        )
        assert sent[0][:10] == message
        peer = pd_mesh.parse_transfer(
            lambda msg: self.reader(SimpleNamespace, msg), sent[0]
        )
        pd_mesh.resolve_peers(
            self.resolve,
            SimpleNamespace(_remote_attn_mesh=p, is_mla_backend=True),
            {peer.mooncake_session_id: peer},
        )
        source = pd_mesh.JournalPath(peer._remote_attn_room)
        target = pd_mesh.JournalPath(receiver._remote_attn_room)
        generation = secrets.token_hex(16)
        write_json(
            target / "offer.json",
            dict(generation=generation, session=manager.get_session_id()),
        )
        self.wait(lambda: read_json(source / "offer.json"))
        source, target = source / generation, target / generation
        write_json(
            source / "remote-service.json", dict(generation=generation, route=service)
        )
        self.wait(lambda: read_json(target / "remote-service.json"))
        route = read_json(target / "remote-service.json")["route"]
        assert route == service and Path(source) != Path(target)
        self.links.append((source, target))
        return target, generation, route

    @staticmethod
    def wait(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError("mesh control propagation")
            time.sleep(0.005)

    def close(self):
        for mesh in self.meshes.values():
            mesh.close()

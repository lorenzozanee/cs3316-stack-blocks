"""TCP environment used by the online fine-tunes.

The Isaac process is the server. The learner, running in the ``dp`` or ``act``
conda environment, is the client. Messages use a binary-framed header (JSON
metadata plus raw contiguous uint8 RGB payloads) so ACT receives the exact same
lossless RGB pixels as official eval without PNG/base64 overhead on localhost.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any

_FRAME_HEADER = struct.Struct("!QQ")


def _enable_nodelay(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


def serve(
    env, host: str, port: int, once: bool = False, *,
    should_recycle: Callable[[], bool] | None = None,
    coordinated_recycle: bool = False,
    free_vram_mib: Callable[[], int] | None = None,
) -> bool:
    """Serve batched ``reset`` / ``step`` and return true for a planned recycle.

    One message steps every active scene. ``once`` exits after the first wave
    of ``num_envs`` episodes, so Diffusion Policy can start a fresh Isaac
    process instead of resetting this one a second time. ``should_recycle``
    can request a fresh server when the next reset follows a complete wave.
    """

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(1)
    width = int(getattr(env, "num_envs", 1))
    print(f"[env] listening on {host}:{port} num_envs {width}", flush=True)
    connection, _address = listener.accept()
    _enable_nodelay(connection)
    print("[env] learner connected", flush=True)
    finished = finished_at_reset = 0
    try:
        while True:
            frame = _read(connection)
            if frame is None:
                break
            message, _blob = frame
            op = message.get("op")
            if op == "ping":
                _write(connection, {"ok": True, "num_envs": width})
            elif op == "headroom":
                free = None if free_vram_mib is None else int(free_vram_mib())
                _write(connection, {"ok": True, "free_vram_mib": free})
            elif op == "reset":
                seeds = message.get("seeds")
                if seeds is None:
                    seeds = [message.get("seed")]
                if (should_recycle is not None and finished - finished_at_reset == width
                        and len(seeds) == width and should_recycle()):
                    # EOF lets the client retry these layouts on the replacement server.
                    return True
                observations, infos = env.reset_batch(seeds)
                finished_at_reset = finished
                privileged = env.privileged_batch(list(range(width)))
                blobs = _BlobCollector()
                _write(
                    connection,
                    {
                        "observations": [_pack_observation(item, blobs) for item in observations],
                        "privileged": [_pack(item) for item in privileged],
                        "infos": [_pack(item) for item in infos],
                        "num_envs": width,
                    },
                    blobs.chunks,
                )
            elif op == "step":
                active = [int(index) for index in message["active"]]
                results = env.step_batch(message["actions"], active)
                privileged = env.privileged_batch(active)
                terminated = [bool(item[2]) for item in results]
                truncated = [bool(item[3]) for item in results]
                blobs = _BlobCollector()
                _write(
                    connection,
                    {
                        "observations": [_pack_observation(item[0], blobs) for item in results],
                        "privileged": [_pack(item) for item in privileged],
                        "scores": [float(item[1]) for item in results],
                        "terminated": terminated,
                        "truncated": truncated,
                        "infos": [_pack(item[4]) for item in results],
                    },
                    blobs.chunks,
                )
                finished += sum(1 for done_flag, cut in zip(terminated, truncated) if done_flag or cut)
                if once and finished >= width:
                    break
            elif op == "recycle" and coordinated_recycle:
                if finished - finished_at_reset != width:
                    raise RuntimeError("Cannot recycle Isaac during an incomplete wave.")
                _write(connection, {"ok": True})
                return True
            elif op == "close":
                try:
                    _write(connection, {"ok": True})
                except OSError:
                    pass
                break
            else:
                _write(connection, {"error": f"unknown op {op}"})
    except Exception as error:
        # Deliver deterministic simulator/reset failures to the learner. An EOF
        # alone would be treated as a crashed server and retried for hours.
        try:
            _write(connection, {"error": f"{type(error).__name__}: {error}"})
        except OSError:
            pass
        raise
    finally:
        connection.close()
        listener.close()
        print(f"[env] closed after {finished} episode(s)", flush=True)
    return False


class RemoteEnv:
    """Client side of :func:`serve`. ``reset`` waits until Isaac is listening."""

    def __init__(
        self,
        host: str,
        port: int,
        restart_each_episode: bool = False,
        timeout: float = 2400.0,
        reset_timeout: float = 10800.0,
    ):
        self.host = host
        self.port = int(port)
        self.restart_each_episode = restart_each_episode
        self.timeout = float(timeout)
        # The first reset builds every cluttered scene. That exceeds the
        # per-step socket timeout when the policy and Isaac share one GPU.
        self.reset_timeout = float(reset_timeout)
        self._socket: socket.socket | None = None
        self._sock_lock = threading.Lock()
        self._aborted = threading.Event()
        self._blob: memoryview = memoryview(b"")
        self._privileged: list[dict] = []

    def connect(self) -> None:
        self._ensure(timeout=self.reset_timeout)

    def worker_width(self) -> int:
        return int(self._request({"op": "ping"})["num_envs"])

    def headroom_mib(self) -> int | None:
        reply = self._request({"op": "headroom"})
        free = reply.get("free_vram_mib")
        return None if free is None else int(free)

    def recycle(self) -> None:
        try:
            self._request({"op": "recycle"})
            with self._sock_lock:
                sock = self._socket
            if sock is not None and sock.recv(1):
                raise RuntimeError("Isaac did not close its connection after recycling.")
        finally:
            self.close()

    def abort(self) -> None:
        self._aborted.set()
        with self._sock_lock:
            sock = self._socket
            self._socket = None
            self._blob = memoryview(b"")
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def reset_batch(self, seeds):
        # The simulator process is restarted until it stays up. A reset that
        # dies with the process should wait for the next one. A live reset
        # keeps the socket open for reset_timeout; closing it early kills
        # the scene build.
        deadline = time.monotonic() + self.reset_timeout
        last_error: Exception | None = None
        while True:
            if self._aborted.is_set():
                raise RuntimeError("Remote env aborted.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConnectionError(
                    f"Isaac env server did not complete reset within {self.reset_timeout:.0f}s"
                ) from last_error
            self._ensure(timeout=remaining)
            try:
                reply = self._request(
                    {"op": "reset", "seeds": [None if seed is None else int(seed) for seed in seeds]},
                    timeout=remaining,
                )
            except TimeoutError as exc:
                if self._aborted.is_set():
                    raise RuntimeError("Remote env aborted.") from exc
                self.close()
                raise ConnectionError(
                    f"Isaac env server did not complete reset within {self.reset_timeout:.0f}s"
                ) from exc
            except (ConnectionError, OSError) as exc:
                if self._aborted.is_set():
                    raise RuntimeError("Remote env aborted.") from exc
                if not isinstance(exc, ConnectionError):
                    raise
                last_error = exc
                self.close()
                if self._aborted.wait(min(2.0, max(deadline - time.monotonic(), 0.0))):
                    raise RuntimeError("Remote env aborted.") from exc
                continue
            observations = [_unpack_observation(item, self._blob) for item in reply["observations"]]
            return observations, reply.get("infos") or [{} for _ in observations]

    def step_batch(self, actions, active):
        reply = self._request(
            {
                "op": "step",
                "actions": [[float(item) for item in action] for action in actions],
                "active": [int(index) for index in active],
            }
        )
        results = []
        infos = reply.get("infos") or [{} for _ in reply["scores"]]
        for index, observation in enumerate(reply["observations"]):
            results.append(
                (
                    _unpack_observation(observation, self._blob),
                    float(reply["scores"][index]),
                    bool(reply["terminated"][index]),
                    bool(reply["truncated"][index]),
                    infos[index],
                )
            )
        return results

    def privileged(self, index: int) -> dict:
        """Privileged state aligned with the last reset or step reply."""

        return dict(self._privileged[index])

    def close(self) -> None:
        with self._sock_lock:
            sock = self._socket
            self._socket = None
            self._blob = memoryview(b"")
        if sock is None:
            return
        try:
            _write(sock, {"op": "close"})
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _ensure(self, timeout: float | None = None) -> None:
        with self._sock_lock:
            if self._aborted.is_set():
                raise RuntimeError("Remote env aborted.")
            if self._socket is not None:
                return
        limit = self.timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._aborted.is_set():
                raise RuntimeError("Remote env aborted.")
            connect_timeout = min(1.0, max(deadline - time.monotonic(), 0.1))
            try:
                sock = socket.create_connection((self.host, self.port), timeout=connect_timeout)
            except OSError as exc:
                last_error = exc
                if self._aborted.wait(min(2.0, max(deadline - time.monotonic(), 0.0))):
                    raise RuntimeError("Remote env aborted.") from exc
                continue
            with self._sock_lock:
                if self._aborted.is_set():
                    try:
                        sock.close()
                    except OSError:
                        pass
                    raise RuntimeError("Remote env aborted.")
                _enable_nodelay(sock)
                sock.settimeout(self.timeout)
                self._socket = sock
                self._blob = memoryview(b"")
                self._privileged = []
            return
        raise TimeoutError(
            f"Isaac env server did not accept {self.host}:{self.port} within {limit:.0f}s"
        ) from last_error

    def _request(self, message: dict, timeout: float | None = None) -> dict:
        with self._sock_lock:
            if self._aborted.is_set():
                raise RuntimeError("Remote env aborted.")
            sock = self._socket
        if sock is None:
            raise RuntimeError("Remote env is not connected.")
        sock.settimeout(self.timeout if timeout is None else float(timeout))
        _write(sock, message)
        frame = _read(sock)
        if frame is None:
            self.close()
            if self._aborted.is_set():
                raise RuntimeError("Remote env aborted.")
            raise ConnectionError("Isaac env server closed the connection.")
        reply, blob = frame
        if "error" in reply:
            raise RuntimeError(reply["error"])
        self._blob = blob
        if "privileged" in reply:
            self._privileged = list(reply["privileged"])
        return reply


class ShardedRemoteEnv:
    """Synchronous sharded client over equal-width ``RemoteEnv`` workers."""

    def __init__(
        self,
        host: str,
        ports: Sequence[int],
        *,
        num_envs: int,
        min_reset_free_vram_mib: int = 0,
        timeout: float = 2400.0,
        reset_timeout: float = 10800.0,
    ):
        port_list = [int(port) for port in ports]
        if not port_list or len(set(port_list)) != len(port_list):
            raise ValueError("ShardedRemoteEnv requires nonempty unique ports.")
        if any(port < 1 or port > 65535 for port in port_list):
            raise ValueError("ShardedRemoteEnv ports must lie in 1..65535.")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < len(port_list):
            raise ValueError("num_envs must be an integer at least the worker count.")
        if num_envs % len(port_list) != 0:
            raise ValueError(f"num_envs ({num_envs}) must be divisible by worker count ({len(port_list)}).")
        if isinstance(min_reset_free_vram_mib, bool) or int(min_reset_free_vram_mib) < 0:
            raise ValueError("min_reset_free_vram_mib must be nonnegative.")
        self.host = str(host)
        self.ports = tuple(port_list)
        self.num_envs = int(num_envs)
        self.worker_envs = self.num_envs // len(self.ports)
        self.min_reset_free_vram_mib = int(min_reset_free_vram_mib)
        self._workers = tuple(
            RemoteEnv(self.host, port, timeout=timeout, reset_timeout=reset_timeout)
            for port in self.ports
        )
        self._pool = ThreadPoolExecutor(max_workers=len(self._workers))
        self._privileged: list[dict] = []
        self._completed_waves = 0
        self._in_wave = False
        self._wave_finished = [False] * self.num_envs
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ShardedRemoteEnv is closed.")

    def _run_workers(self, jobs: Sequence[tuple[int, Callable[[], Any]]]) -> dict[int, Any]:
        futures = [(worker_index, self._pool.submit(fn)) for worker_index, fn in jobs]
        pending = {future for _, future in futures}
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            trigger = next(
                (future.exception() for _, future in futures if future in done and future.exception() is not None),
                None,
            )
            if trigger is not None:
                self._abort_workers()
                if pending:
                    wait(pending)
                self.close()
                raise trigger
        return {worker_index: future.result() for worker_index, future in futures}

    def _abort_workers(self) -> None:
        for worker in self._workers:
            worker.abort()

    def connect(self) -> None:
        self._require_open()

        def _connect_worker(worker: RemoteEnv) -> None:
            worker.connect()
            width = worker.worker_width()
            if width != self.worker_envs:
                raise RuntimeError(
                    f"Isaac worker on {worker.host}:{worker.port} reported num_envs={width}, "
                    f"expected {self.worker_envs}."
                )

        self._run_workers([(i, lambda w=worker: _connect_worker(w)) for i, worker in enumerate(self._workers)])

    def reset_batch(self, seeds):
        self._require_open()
        seed_list = [None if seed is None else int(seed) for seed in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seed_list)}.")
        if self._completed_waves > 0 and self.min_reset_free_vram_mib > 0:
            try:
                free_mib = self._workers[0].headroom_mib()
            except Exception:
                self._abort_workers()
                self.close()
                raise
            if free_mib is not None and free_mib < self.min_reset_free_vram_mib:
                print(
                    f"[env] coordinated recycle before reset: free VRAM {free_mib} MiB "
                    f"below {self.min_reset_free_vram_mib} MiB across {len(self._workers)} workers",
                    flush=True,
                )
                self._run_workers([(i, worker.recycle) for i, worker in enumerate(self._workers)])
        width = self.worker_envs
        outputs = self._run_workers(
            [
                (i, lambda w=worker, s=seed_list[i * width : (i + 1) * width]: w.reset_batch(s))
                for i, worker in enumerate(self._workers)
            ]
        )
        observations: list[dict] = []
        infos: list[dict] = []
        privileged: list[dict] = []
        for i, worker in enumerate(self._workers):
            worker_obs, worker_infos = outputs[i]
            if len(worker_obs) != width or len(worker_infos) != width or len(worker._privileged) != width:
                self.close()
                raise RuntimeError(
                    f"Isaac worker {i} returned inconsistent reset width "
                    f"({len(worker_obs)}, {len(worker_infos)}, {len(worker._privileged)}) != {width}."
                )
            base = i * width
            for local_index in range(width):
                observations.append(worker_obs[local_index])
                info = dict(worker_infos[local_index])
                if "env_index" in info:
                    info["env_index"] = base + local_index
                infos.append(info)
                privileged.append(worker._privileged[local_index])
        self._privileged = privileged
        self._wave_finished = [False] * self.num_envs
        self._in_wave = True
        return observations, infos

    def step_batch(self, actions, active):
        self._require_open()
        active_list = [int(index) for index in active]
        if len(actions) != len(active_list):
            raise ValueError(f"Got {len(actions)} actions for {len(active_list)} environments.")
        if not active_list:
            self._privileged = []
            return []
        width = self.worker_envs
        grouped: dict[int, list[tuple[int, int, Any]]] = {}
        for pos, (global_index, action) in enumerate(zip(active_list, actions)):
            if global_index < 0 or global_index >= self.num_envs:
                raise ValueError(f"Active environment index {global_index} out of range 0..{self.num_envs - 1}.")
            worker_index, local_index = divmod(global_index, width)
            grouped.setdefault(worker_index, []).append((pos, local_index, action))
        jobs = []
        for worker_index in sorted(grouped):
            items = grouped[worker_index]
            w_actions = [item[2] for item in items]
            w_active = [item[1] for item in items]
            worker = self._workers[worker_index]
            jobs.append((worker_index, lambda w=worker, a=w_actions, idx=w_active: w.step_batch(a, idx)))
        outputs = self._run_workers(jobs)
        results: list[Any] = [None] * len(active_list)
        privileged: list[Any] = [None] * len(active_list)
        for worker_index in sorted(grouped):
            items = grouped[worker_index]
            worker = self._workers[worker_index]
            worker_results = outputs[worker_index]
            if len(worker_results) != len(items) or len(worker._privileged) != len(items):
                self.close()
                raise RuntimeError(
                    f"Isaac worker {worker_index} returned {len(worker_results)} step results "
                    f"for {len(items)} active environments."
                )
            for item_pos, (global_pos, _local_index, _action) in enumerate(items):
                obs, score, terminated, truncated, info = worker_results[item_pos]
                mapped_info = dict(info)
                if "env_index" in mapped_info:
                    mapped_info["env_index"] = active_list[global_pos]
                results[global_pos] = (obs, score, terminated, truncated, mapped_info)
                privileged[global_pos] = worker._privileged[item_pos]
                if terminated or truncated:
                    self._wave_finished[active_list[global_pos]] = True
        if self._in_wave and all(self._wave_finished):
            self._completed_waves += 1
            self._in_wave = False
        self._privileged = privileged
        return results

    def privileged(self, index: int) -> dict:
        return dict(self._privileged[index])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass
        self._pool.shutdown(wait=True, cancel_futures=True)


class _BlobCollector:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self._offset = 0

    def append(self, raw: bytes) -> tuple[int, int]:
        offset = self._offset
        size = len(raw)
        self.chunks.append(raw)
        self._offset += size
        return offset, size


def _recv_exact(connection: socket.socket, size: int) -> bytearray | None:
    if size == 0:
        return bytearray()
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        received = connection.recv_into(view[got:])
        if received == 0:
            return None
        got += received
    return buf


def _read(connection: socket.socket) -> tuple[dict, memoryview] | None:
    prefix = _recv_exact(connection, _FRAME_HEADER.size)
    if prefix is None:
        return None
    header_len, blob_len = _FRAME_HEADER.unpack(prefix)
    header_bytes = _recv_exact(connection, header_len)
    if header_bytes is None:
        return None
    blob_bytes = _recv_exact(connection, blob_len)
    if blob_bytes is None:
        return None
    return json.loads(bytes(header_bytes).decode("utf-8")), memoryview(blob_bytes)


def _write(connection: socket.socket, message: dict, blobs: list[bytes] | tuple[bytes, ...] = ()) -> None:
    header = json.dumps(message, separators=(",", ":")).encode("utf-8")
    blob_len = sum(len(chunk) for chunk in blobs)
    connection.sendall(_FRAME_HEADER.pack(len(header), blob_len))
    connection.sendall(header)
    for chunk in blobs:
        if chunk:
            connection.sendall(chunk)


def _pack_observation(observation: dict, blobs: _BlobCollector) -> dict:
    packed = {}
    for key, value in observation.items():
        if key == "image":
            packed[key] = _pack_image(value, blobs)
        else:
            packed[key] = _pack(value)
    return packed


def _unpack_observation(payload: dict, blob: memoryview) -> dict:
    observation = {}
    for key, value in payload.items():
        if key == "image":
            observation[key] = _unpack_image(value, blob)
        else:
            observation[key] = value
    return observation


def _pack_image(value, blobs: _BlobCollector) -> dict:
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    raw = np.ascontiguousarray(array).tobytes()
    offset, nbytes = blobs.append(raw)
    return {
        "offset": offset,
        "nbytes": nbytes,
        "shape": list(array.shape),
    }


def _unpack_image(payload: dict, blob: memoryview):
    import numpy as np

    offset = int(payload["offset"])
    nbytes = int(payload["nbytes"])
    shape = tuple(int(dim) for dim in payload["shape"])
    expected = int(np.prod(shape))
    if nbytes != expected or offset < 0 or offset + nbytes > len(blob):
        raise RuntimeError(f"Corrupt raw RGB frame header: shape={shape} offset={offset} nbytes={nbytes}.")
    return np.frombuffer(blob[offset : offset + nbytes], dtype=np.uint8).reshape(shape).copy()


def _pack(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)

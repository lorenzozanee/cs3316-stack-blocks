"""A Linux process deadline for Isaac reset, independent of Kit's GIL."""

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

# Unified Linux 5.3+ syscall numbers (shared across x86_64/arm64).
_SYS_PIDFD_OPEN = 434
_SYS_PIDFD_SEND_SIGNAL = 424


def _libc():
    if sys.platform != "linux":
        raise RuntimeError("reset_deadline requires Linux pidfd syscalls.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _pidfd_open(pid: int) -> int:
    fd = _libc().syscall(ctypes.c_long(_SYS_PIDFD_OPEN), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def _pidfd_send_signal(pid_fd: int, sig: int) -> None:
    rc = _libc().syscall(
        ctypes.c_long(_SYS_PIDFD_SEND_SIGNAL),
        ctypes.c_int(pid_fd),
        ctypes.c_int(sig),
        ctypes.c_void_p(0),
        ctypes.c_uint(0),
    )
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


@contextmanager
def reset_deadline(seconds: float | None, port: int):
    if seconds is None:
        yield
        return

    deadline = time.monotonic() + seconds
    pid_fd = _pidfd_open(os.getpid())
    try:
        guard = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), str(pid_fd), str(deadline), str(port)],
            pass_fds=(pid_fd,), stdin=subprocess.PIPE,
        )
    finally:
        os.close(pid_fd)
    try:
        yield
    finally:
        # Reap the guard before sending a reset reply: no timer can reach a step.
        guard.communicate(b"\x01")
        if guard.returncode:
            raise RuntimeError(f"Reset deadline process failed on port {port}: {guard.returncode}.")


def _watch(pid_fd: int, deadline: float, port: int) -> None:
    ready, _, _ = select.select([sys.stdin], [], [], max(0.0, deadline - time.monotonic()))
    if ready:
        return
    print(f"[env] reset deadline exceeded on port {port}; stopping simulator", file=sys.stderr, flush=True)
    _pidfd_send_signal(pid_fd, signal.SIGKILL)
    raise SystemExit(1)


if __name__ == "__main__":
    _watch(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))

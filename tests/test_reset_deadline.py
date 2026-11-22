import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest

from starter.envs.remote import ShardedRemoteEnv
from starter.envs.reset_deadline import reset_deadline

ROOT = Path(__file__).resolve().parents[1]
SERVER = r"""
import sys
import time
sys.path.insert(0, sys.argv[1])
from starter.envs.remote import serve

class Task:
    num_envs = 1

    def reset_batch(self, seeds):
        print(f"reset {seeds}", flush=True)
        if sys.argv[3] == "hang":
            time.sleep(60)
        self.seed = seeds[0]
        return [{"proprio": [self.seed]}], [{"env_index": 0, "seed": self.seed}]

    def privileged_batch(self, indices):
        return [{"seed": self.seed} for i in indices]

    def step_batch(self, actions, active):
        print("step", flush=True)
        return [({"proprio": [self.seed]}, 0.0, False, True, {}) for i in active]

serve(Task(), "127.0.0.1", int(sys.argv[2]), reset_deadline_seconds=0.5)
"""


@unittest.skipUnless(sys.platform == "linux", "Linux pidfd required")
class ResetDeadlineTest(unittest.TestCase):
    @staticmethod
    def _port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _run_generations(self, modes, drive):
        p0, p1 = self._port(), self._port()
        while p0 == p1:
            p1 = self._port()
        results = [[], []]
        errors = []
        stop = threading.Event()

        def supervisor(index, port, generations):
            try:
                for mode in generations:
                    if stop.is_set():
                        return
                    with subprocess.Popen(
                        [sys.executable, "-I", "-c", SERVER, str(ROOT / "src"), str(port), mode],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    ) as process:
                        try:
                            output, _ = process.communicate(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate()
                            raise
                        results[index].append((process.returncode, output))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=supervisor, args=spec) for spec in (
            (0, p0, ["ok"]), (1, p1, modes),
        )]
        for thread in threads:
            thread.start()
        env = ShardedRemoteEnv("127.0.0.1", [p0, p1], num_envs=2,
                               timeout=10, reset_timeout=15, reset_retries=2)
        try:
            env.connect()
            drive(env)
        finally:
            stop.set()
            env.close()
            for thread in threads:
                thread.join(timeout=22)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(errors)
        return results

    def test_hung_worker_recovers_same_slot_and_sibling_keeps_running(self):
        def drive(env):
            obs, info = env.reset_batch([14, 23])
            self.assertEqual([o["proprio"][0] for o in obs], [14, 23])
            self.assertEqual([i["env_index"] for i in info], [0, 1])
            self.assertEqual([env.privileged(i)["seed"] for i in range(2)], [14, 23])
            time.sleep(0.6)  # Completed reset guards cannot terminate a later step.
            self.assertTrue(all(row[3] for row in env.step_batch([[0.0], [0.0]], [0, 1])))

        healthy, replacement = self._run_generations(["hang", "hang", "ok"], drive)
        self.assertEqual([code for code, _ in healthy], [0])
        self.assertEqual([code for code, _ in replacement], [-signal.SIGKILL, -signal.SIGKILL, 0])
        self.assertEqual(healthy[0][1].count("reset [14]"), 1)
        self.assertTrue(all("reset [23]" in log for _, log in replacement))
        self.assertTrue(all("reset reply" not in log for _, log in replacement[:-1]))
        self.assertIn("step\n", healthy[0][1])
        self.assertIn("step\n", replacement[-1][1])

    def test_three_hangs_fail_without_fourth_reset(self):
        def drive(env):
            with self.assertRaisesRegex(ConnectionError, "Reset retry limit reached"):
                env.reset_batch([14, 23])

        healthy, stalled = self._run_generations(["hang", "hang", "hang"], drive)
        self.assertEqual([code for code, _ in stalled], [-signal.SIGKILL] * 3)
        self.assertTrue(all("reset [23]" in log for _, log in stalled))
        self.assertEqual(healthy[0][1].count("reset [14]"), 1)

    def test_none_deadline_is_noop(self):
        ran = False
        with reset_deadline(None, 5555):
            ran = True
        self.assertTrue(ran)

    def test_exception_disarms_guard_without_kill(self):
        with self.assertRaisesRegex(ValueError, "boom"):
            with reset_deadline(0.4, 5555):
                raise ValueError("boom")
        time.sleep(0.5)


if __name__ == "__main__":
    unittest.main()

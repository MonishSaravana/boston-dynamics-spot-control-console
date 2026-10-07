import threading
import time
import unittest

from scope.runtime import Health, LatestWorker, ModuleConfig, Runtime


class RuntimeTests(unittest.TestCase):
    def test_rates_toggles_stale_failure_and_recovery(self):
        rt = Runtime({"pose": ModuleConfig(max_hz=5), "map": ModuleConfig()})
        self.assertEqual(rt.run("pose", 0., lambda: 7), 7)
        self.assertIsNone(rt.run("pose", .1, lambda: 8))
        self.assertEqual(rt.snapshot(1.)["pose"]["status"], "STALE")
        with self.assertLogs("scope.runtime", level="ERROR"):
            rt.run("pose", 1., lambda: 1/0)
        self.assertEqual(rt.state("pose").last_output, 7)
        self.assertEqual(rt.snapshot(1.)["pose"]["status"], "FAILED")
        self.assertEqual(rt.run("map", 1., lambda: 9), 9)
        self.assertEqual(rt.run("pose", 1.2, lambda: 10), 10)
        rt.configure("pose", ModuleConfig(enabled=False))
        self.assertIsNone(rt.run("pose", 2., lambda: 11))
        self.assertEqual(rt.snapshot(2.)["pose"]["status"], "DISABLED")
        rt.configure("pose", ModuleConfig())
        self.assertEqual(rt.run("pose", 2., lambda: 12), 12)
        self.assertEqual(rt.state("pose").restarts, 1)
        self.assertIsNone(rt.snapshot(2.)["pose"]["gpu_inference_ms"])

    def test_slow_worker_does_not_block_mapping_and_queue_is_bounded(self):
        gate = threading.Event()
        entered = threading.Event()
        def infer(item):
            entered.set()
            gate.wait(2)
            return item * 2
        worker = LatestWorker(infer)
        try:
            worker.submit(1)
            self.assertTrue(entered.wait(1))
            rt = Runtime()
            for i in range(20):
                worker.submit(i)
                rt.run("map", i/30, lambda: i)
            self.assertEqual(rt.state("map").successes, 20)
            self.assertEqual(worker.queue_depth, 1)
            self.assertGreater(worker.dropped, 10)
            gate.set()
            deadline = time.monotonic()+2
            result = None
            while result is None and time.monotonic()<deadline:
                result = worker.poll()
                time.sleep(.001)
            self.assertIsNotNone(result)
        finally:
            gate.set()
            worker.close()

    def test_invalid_configuration(self):
        for kwargs in ({"max_hz": 0}, {"max_hz": float("nan")}, {"every_n": 0}, {"stale_after_s": -1}):
            with self.assertRaises(ValueError):
                ModuleConfig(**kwargs)

import asyncio
import threading
import time
import unittest

from better_gemini.grok_core import BetterGrokError
from better_gemini.grok_rate_limit import (
    GrokRateLimitCoordinator,
    GrokRateLimitError,
    GrokRequestCancelled,
    grok_rate_limit_coordinator,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class GrokRateLimitCoordinatorTests(unittest.TestCase):
    def test_module_exposes_process_singleton(self):
        self.assertIsInstance(grok_rate_limit_coordinator, GrokRateLimitCoordinator)

    def test_parallel_attempts_are_evenly_spaced(self):
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=20,
            max_in_flight=3,
            max_retries=0,
        )
        barrier = threading.Barrier(4)
        attempt_times = []
        attempt_lock = threading.Lock()

        def worker():
            barrier.wait()

            def operation():
                with attempt_lock:
                    attempt_times.append(time.monotonic())

            coordinator.execute(model="image-model", operation=operation)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        ordered = sorted(attempt_times)
        self.assertEqual(len(ordered), 3)
        self.assertGreaterEqual(ordered[1] - ordered[0], 0.04)
        self.assertGreaterEqual(ordered[2] - ordered[1], 0.04)

    def test_max_in_flight_blocks_extra_parallel_request(self):
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=2,
            max_retries=0,
        )
        release = threading.Event()
        two_started = threading.Event()
        state_lock = threading.Lock()
        started = 0

        def operation():
            nonlocal started
            with state_lock:
                started += 1
                if started == 2:
                    two_started.set()
            release.wait(timeout=2)

        threads = [
            threading.Thread(
                target=coordinator.execute,
                kwargs={"model": "image-model", "operation": operation},
            )
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()

        self.assertTrue(two_started.wait(timeout=1))
        time.sleep(0.03)
        with state_lock:
            self.assertEqual(started, 2)
        self.assertEqual(coordinator.snapshot("image-model").active_requests, 2)

        release.set()
        for thread in threads:
            thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(started, 3)
        self.assertEqual(coordinator.snapshot("image-model").active_requests, 0)

    def test_rate_limit_retries_with_exponential_backoff(self):
        clock = FakeClock()
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=3,
            backoff_base_seconds=1,
            backoff_max_seconds=10,
            jitter_ratio=0,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise GrokRateLimitError("limited")
            return "ok"

        result = coordinator.execute(model="image-model", operation=operation)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(clock.sleeps, [1.0, 2.0])
        snapshot = coordinator.snapshot("image-model")
        self.assertEqual(snapshot.active_requests, 0)
        self.assertEqual(snapshot.attempts_last_minute, 3)

    def test_retry_after_header_delay_wins_over_exponential_delay(self):
        clock = FakeClock()
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=1,
            backoff_base_seconds=1,
            backoff_max_seconds=10,
            jitter_ratio=0,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise GrokRateLimitError("limited", retry_after_seconds=7.0)
            return "ok"

        self.assertEqual(coordinator.execute(model="image-model", operation=operation), "ok")
        self.assertEqual(clock.sleeps, [7.0])

    def test_one_429_pauses_other_parallel_callers_for_same_model(self):
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=2,
            max_retries=1,
            backoff_base_seconds=0.1,
            backoff_max_seconds=0.1,
            jitter_ratio=0,
        )
        limited = threading.Event()
        limited_at = []
        second_attempt_at = []
        first_attempt = True

        def first_operation():
            nonlocal first_attempt
            if first_attempt:
                first_attempt = False
                limited_at.append(time.monotonic())
                limited.set()
                raise GrokRateLimitError("limited")
            return "first-ok"

        first_thread = threading.Thread(
            target=coordinator.execute,
            kwargs={"model": "image-model", "operation": first_operation},
        )
        first_thread.start()
        self.assertTrue(limited.wait(timeout=1))
        for _ in range(100):
            if coordinator.snapshot("image-model").blocked_for_seconds > 0:
                break
            time.sleep(0.001)
        self.assertGreater(coordinator.snapshot("image-model").blocked_for_seconds, 0)

        second_thread = threading.Thread(
            target=coordinator.execute,
            kwargs={
                "model": "image-model",
                "operation": lambda: second_attempt_at.append(time.monotonic()),
            },
        )
        second_thread.start()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(len(second_attempt_at), 1)
        self.assertGreaterEqual(second_attempt_at[0] - limited_at[0], 0.08)

    def test_retry_exhaustion_releases_active_request(self):
        clock = FakeClock()
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=2,
            backoff_base_seconds=1,
            backoff_max_seconds=10,
            jitter_ratio=0,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise GrokRateLimitError("still limited")

        with self.assertRaisesRegex(BetterGrokError, "after 3 attempt"):
            coordinator.execute(model="image-model", operation=operation)

        self.assertEqual(attempts, 3)
        self.assertEqual(coordinator.snapshot("image-model").active_requests, 0)

    def test_non_rate_limit_failure_is_not_retried_and_releases_request(self):
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=5,
        )
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise RuntimeError("failed")

        with self.assertRaisesRegex(RuntimeError, "failed"):
            coordinator.execute(model="image-model", operation=operation)

        self.assertEqual(attempts, 1)
        self.assertEqual(coordinator.snapshot("image-model").active_requests, 0)

    def test_cancelled_coroutine_releases_entry_when_worker_finishes(self):
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=0,
        )
        started = threading.Event()
        release = threading.Event()

        def operation():
            started.set()
            release.wait(timeout=2)

        async def run_test():
            task = asyncio.create_task(
                asyncio.to_thread(
                    coordinator.execute,
                    model="image-model",
                    operation=operation,
                )
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            self.assertEqual(coordinator.snapshot("image-model").active_requests, 1)
            release.set()
            for _ in range(100):
                if coordinator.snapshot("image-model").active_requests == 0:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(coordinator.snapshot("image-model").active_requests, 0)

        asyncio.run(run_test())

    def test_cancellation_during_backoff_stops_retry_and_releases_request(self):
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=5,
            backoff_base_seconds=5,
            backoff_max_seconds=5,
            jitter_ratio=0,
        )
        cancel_event = threading.Event()
        failures = []
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise GrokRateLimitError("limited")

        def worker():
            try:
                coordinator.execute(
                    model="image-model",
                    operation=operation,
                    cancel_event=cancel_event,
                )
            except GrokRequestCancelled as error:
                failures.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        for _ in range(100):
            if coordinator.snapshot("image-model").blocked_for_seconds > 0:
                break
            time.sleep(0.001)
        cancel_event.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(attempts, 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], GrokRequestCancelled)
        self.assertEqual(coordinator.snapshot("image-model").active_requests, 0)

    def test_recent_attempt_entries_expire_after_one_minute(self):
        clock = FakeClock()
        coordinator = GrokRateLimitCoordinator(
            max_requests_per_second=1000,
            max_in_flight=1,
            max_retries=0,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        coordinator.execute(model="image-model", operation=lambda: "ok")
        self.assertEqual(coordinator.snapshot("image-model").attempts_last_minute, 1)

        clock.now += 61
        self.assertEqual(coordinator.snapshot("image-model").attempts_last_minute, 0)


if __name__ == "__main__":
    unittest.main()

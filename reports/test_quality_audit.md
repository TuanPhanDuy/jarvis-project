# Test Quality Audit — Q1

Reviewed: `tests/test_events.py` and `tests/test_queue.py`

---

## test_events.py

### Issues Found

1. **[FIXED] `test_subscribe_and_publish` called `asyncio.get_event_loop().run_until_complete()` inside an async test.**
   This raises `RuntimeError: This event loop is already running` because pytest-asyncio already manages the event loop. Fixed by removing the erroneous line — the `asyncio.ensure_future` + `await asyncio.sleep(0)` pattern is correct.

2. **[LOW] `test_fires_cpu_event_when_above_threshold` inlines the poll logic rather than calling `SystemMonitor.run()`.**
   The test simulates a poll cycle manually instead of calling the monitor's actual method. This tests the logic correctly but is slightly brittle if the polling code moves.
   **Recommendation:** Acceptable for now — the monitor's `run()` loop uses `asyncio.sleep(INTERVAL)` (60s), making it impractical to test end-to-end without heavy time mocking. The direct logic test is a reasonable trade-off.

3. **[PASS] Async tests use `@pytest.mark.asyncio` correctly via `asyncio_mode=strict` in pyproject.toml.**
   All async test methods are properly decorated and run in isolated event loops.

4. **[PASS] No temp file state leaks.** `TestFileWatcher` uses pytest's `tmp_path` fixture, which is automatically cleaned up after each test.

5. **[PASS] Mock patches are function-scoped.** No global monkeypatching found.

6. **[PASS] Missing edge case covered:** `test_handler_exception_does_not_stop_bus` verifies that a failing handler doesn't prevent other handlers from running — this is the most important resilience property of the event bus.

### Verdict: PASS WITH NOTES

---

## test_queue.py

### Issues Found

1. **[FIXED] `test_publish_sets_persistent_delivery_mode` compared `BasicProperties.delivery_mode` (int `2`) with `pika.DeliveryMode.Persistent` (enum).**
   Python's `==` returns False for `2 == <DeliveryMode.Persistent: 2>` in this pika version. Fixed by comparing `props.delivery_mode in (2, "2")` which is value-based.

2. **[LOW] `TestWorkerDispatch` tests `_on_message` (the worker's callback) directly rather than through the full `channel.start_consuming()` loop.**
   This is correct — testing `_on_message` in isolation is the right approach since `start_consuming()` is a blocking loop. The tests accurately verify: JSON parse errors → nack, valid tasks → ack, reply_to routing.

3. **[PASS] No RabbitMQ connection is ever opened.** All tests use `unittest.mock.patch` to intercept `pika.BlockingConnection` before it dials the network.

4. **[PASS] `test_publish_connection_always_closed` verifies the `finally: connection.close()` branch.** This is an important guarantee — failed publishes must still release the connection.

5. **[PASS] Error path tested:** `test_processing_error_still_acks` verifies that a task error result (not an exception) still ACKs the message, which is the correct behavior — errors are communicated in the result payload, not via nack.

6. **[LOW] No test for the exponential backoff in `_connect_with_retry`.** The function retries up to 10 times with `min(2^attempt, 60)` second delays. Adding a test for retry exhaustion would be useful.
   **Recommendation:** Add `test_connect_with_retry_raises_after_max_attempts` in a future iteration.

### Verdict: PASS WITH NOTES

---

## Summary

| File | Verdict | Issues Fixed | Remaining |
|------|---------|-------------|-----------|
| `test_events.py` | PASS WITH NOTES | 1 (event loop error) | 1 minor (inline logic) |
| `test_queue.py` | PASS WITH NOTES | 1 (enum comparison) | 1 minor (no retry exhaustion test) |

Both files are production-quality: no state leaks, proper isolation, meaningful edge cases covered. The two remaining low-priority items are acceptable for this iteration.

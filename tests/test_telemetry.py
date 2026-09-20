import asyncio
import pytest
from vaxguard.telemetry.metrics import track_latency


@pytest.mark.asyncio
async def test_track_latency_preserves_return_value():
    @track_latency("test_operation")
    async def sample_coro(x, y):
        await asyncio.sleep(0.01)
        return x + y

    result = await sample_coro(10, 20)
    assert result == 30


@pytest.mark.asyncio
async def test_track_latency_propagates_exception():
    @track_latency("failing_operation")
    async def failing_coro():
        await asyncio.sleep(0.01)
        raise ValueError("Operation failed intentionally.")

    with pytest.raises(ValueError, match="Operation failed intentionally."):
        await failing_coro()


@pytest.mark.asyncio
async def test_track_latency_with_kwargs():
    @track_latency("kwargs_operation")
    async def kwargs_coro(name: str, prefix: str = "Hello"):
        return f"{prefix}, {name}!"

    res = await kwargs_coro("VaxGuard", prefix="Greetings")
    assert res == "Greetings, VaxGuard!"

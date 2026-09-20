import time
from functools import wraps
from vaxguard.core.logger import get_logger

logger = get_logger("Telemetry")

def track_latency(event_name: str):
    """
    Decorator to track the latency of asynchronous operations.
    In a full production environment, this pushes to Datadog/Prometheus.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                logger.debug(f"[METRIC] {event_name} took {latency_ms:.2f}ms")
        return wrapper
    return decorator

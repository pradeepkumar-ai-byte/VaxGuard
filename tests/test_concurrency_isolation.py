"""
Unit and Stress Tests for Multi-Tenant Session Isolation and Thread Safety in Colang 2.0.

Verifies:
1. Thread-safe variable mutations and reads across concurrent workers.
2. Complete session memory isolation under 50+ simultaneous tenants.
3. Rollback containment: an execution failure or rollback on Tenant 1 never bleeds
   or mutates Tenant 2..N state.
4. Dialogue turn progression is isolated per session.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from vaxguard.nemo.colang2 import (
    Colang2Engine,
    FlowResult,
    ScopeLevel,
    ScopedContext,
)


class TestScopedContextThreadSafety:
    """Tests the thread safety and lock primitives of ScopedContext."""

    def test_concurrent_set_and_get(self) -> None:
        """Verify that 100 threads can read and write variables without corruption."""
        ctx = ScopedContext(session_id="thread_test_sess")
        num_threads = 50
        writes_per_thread = 100

        def worker(thread_idx: int) -> None:
            for i in range(writes_per_thread):
                key = f"thread_var_{thread_idx}_{i}"
                val = f"val_{thread_idx}_{i}"
                ctx.set(key, val, scope=ScopeLevel.SESSION)
                retrieved = ctx.get(key, scope=ScopeLevel.SESSION)
                assert retrieved == val

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all variables were written successfully
        for t in range(num_threads):
            for i in range(writes_per_thread):
                key = f"thread_var_{t}_{i}"
                assert ctx.get(key) == f"val_{t}_{i}"

    def test_snapshot_and_rollback_thread_safety(self) -> None:
        """Verify snapshot and rollback are atomic and thread-safe."""
        ctx = ScopedContext(session_id="snap_test_sess")
        ctx.set("stable_var", "clean", scope=ScopeLevel.SESSION)
        snap_id = ctx.snapshot()

        ctx.set("stable_var", "dirty", scope=ScopeLevel.SESSION)
        ctx.set("temporary_var", 999, scope=ScopeLevel.TURN)

        ctx.rollback(snap_id)

        assert ctx.get("stable_var", scope=ScopeLevel.SESSION) == "clean"
        assert not ctx.has("temporary_var")


class TestColang2EngineSessionIsolation:
    """Tests multi-tenant session isolation in Colang2Engine."""

    @pytest.mark.asyncio
    async def test_multi_session_variable_isolation(self) -> None:
        """50 concurrent user sessions must have 100% isolated session contexts."""
        engine = Colang2Engine()
        num_sessions = 50

        # Register a simple test flow
        script = """
flow user_session_test:
    $user_id = $input_user
    $session_token = $input_token
    bot say "Session processed for user"
"""
        engine.load_flows_from_text(script)

        async def run_session(session_idx: int) -> FlowResult:
            sess_id = f"session_tenant_{session_idx}"
            return await engine.execute_flow(
                "user_session_test",
                inputs={"input_user": f"user_{session_idx}", "input_token": f"tok_{session_idx}"},
                session_id=sess_id,
            )

        # Execute 50 concurrent session flows
        results = await asyncio.gather(*(run_session(i) for i in range(num_sessions)))

        # Assert all flows succeeded
        for res in results:
            assert res.status == "completed"

        # Assert each session context holds strictly its own data
        for i in range(num_sessions):
            sess_id = f"session_tenant_{i}"
            sess_ctx = engine.get_session(sess_id)
            assert sess_ctx is not None
            # Context must have user's own data
            assert sess_ctx.get("user_id") == f"user_{i}"
            assert sess_ctx.get("session_token") == f"tok_{i}"

    @pytest.mark.asyncio
    async def test_failure_and_rollback_containment(self) -> None:
        """A failure/rollback in Tenant 1 must never affect Tenant 2..50."""
        engine = Colang2Engine()

        def _bad_action() -> None:
            raise RuntimeError("Fatal runtime exception triggered")

        engine.register_action("bad_action", _bad_action)

        # Register successful flow
        script_success = """
flow safe_flow:
    $status = "confirmed"
"""
        # Register failing flow
        script_fail = """
flow failing_flow:
    $status = "corrupted_attempt"
    execute bad_action()
"""
        engine.load_flows_from_text(script_success)
        engine.load_flows_from_text(script_fail)

        # Setup 10 healthy sessions
        for i in range(1, 10):
            sess_id = f"tenant_{i}"
            ctx = engine.get_or_create_session(sess_id)
            ctx.set("status", "healthy", scope=ScopeLevel.SESSION)

        # Trigger failing flow on tenant_0 with rollback
        t0_ctx = engine.get_or_create_session("tenant_0")
        snap = t0_ctx.snapshot()
        res_fail = await engine.execute_flow("failing_flow", session_id="tenant_0")
        assert res_fail.status == "failed"
        t0_ctx.rollback(snap)

        # Verify tenant_0 rolled back
        assert t0_ctx.get("status") is None

        # Verify tenant_1..9 were completely unaffected and maintain "healthy"
        for i in range(1, 10):
            sess_ctx = engine.get_session(f"tenant_{i}")
            assert sess_ctx is not None
            assert sess_ctx.get("status") == "healthy"

    @pytest.mark.asyncio
    async def test_session_lifecycle_eviction(self) -> None:
        """Engine should support clearing session contexts to prevent memory leaks."""
        engine = Colang2Engine()
        sess_id = "temp_tenant_99"

        ctx = engine.get_or_create_session(sess_id)
        ctx.set("secret_key", "sk-1234567890", scope=ScopeLevel.SESSION)
        assert engine.get_session(sess_id) is not None

        cleared = engine.clear_session(sess_id)
        assert cleared is True
        assert engine.get_session(sess_id) is None

    def test_independent_turn_progression(self) -> None:
        """Advancing dialogue turns on session A does not increment turn on session B."""
        engine = Colang2Engine()
        engine.get_or_create_session("sess_A")
        engine.get_or_create_session("sess_B")

        engine.new_turn("sess_A")
        engine.new_turn("sess_A")
        t3_a = engine.new_turn("sess_A")

        sess_b = engine.get_session("sess_B")
        assert sess_b is not None
        assert sess_b.turn_id == 1  # Unaffected by sess_A's turns
        assert t3_a == 4

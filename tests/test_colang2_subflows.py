"""
Tests for Colang 2.0 Nested Sub-Dialogues, Subflows, and Context Isolation.
"""

from __future__ import annotations

import asyncio
import pytest

from vaxguard.nemo.colang2 import (
    Colang2Engine,
    Colang2Parser,
    Event,
    EventBus,
    Flow2,
    ScopedContext,
    ScopeLevel,
    SendStep,
    SubflowStep,
    AssignStep,
    StopStep,
)


@pytest.mark.asyncio
async def test_subflow_step_execution_and_argument_passing():
    """Verify that SubflowStep executes a child flow with arguments in isolated context."""
    bus = EventBus()
    context = ScopedContext(event_bus=bus)
    engine = Colang2Engine(event_bus=bus, context=context)

    # 1. Define child subflow that calculates discounted price
    child_script = """
flow calculate_discount:
  $discount = "20"
  $final_price = "80"
  send bot say "Calculated price is 80"
"""
    engine.load_flows_from_text(child_script)

    # 2. Define parent flow calling child subflow
    parent_script = """
flow checkout_flow:
  call flow calculate_discount
  send bot say "Checkout complete"
"""
    engine.load_flows_from_text(parent_script)

    # Execute parent flow
    result = await engine.execute_flow("checkout_flow")
    assert result.success is True
    assert result.status == "completed"

    # Events on bus should contain both child and parent emissions
    history = bus.history
    bot_messages = [e.data.get("text") for e in history if e.name == "BotUtterance"]
    assert "Calculated price is 80" in bot_messages
    assert "Checkout complete" in bot_messages


@pytest.mark.asyncio
async def test_subflow_return_value_capture():
    """Verify that `call <flow> as $variable` captures subflow output."""
    bus = EventBus()
    context = ScopedContext(event_bus=bus)
    engine = Colang2Engine(event_bus=bus, context=context)

    # Child flow returns a computed value via AssignStep
    child_flow = Flow2(name="security_clearance_check")
    child_flow.add_step(AssignStep(variable="clearance_level", value="TOP_SECRET", scope=ScopeLevel.TURN))
    engine.register_flow(child_flow)

    # Parent flow invokes child flow and binds result
    parent_flow = Flow2(name="access_request")
    parent_flow.add_step(SubflowStep(flow_name="security_clearance_check", variable="user_clearance"))
    parent_flow.add_step(SendStep(event_name="BotUtterance", source="bot", data={"text": "Clearance: $user_clearance"}))
    engine.register_flow(parent_flow)

    res = await engine.execute_flow("access_request")
    assert res.success is True

    bot_events = [e.data.get("text") for e in bus.history if e.name == "BotUtterance"]
    assert "Clearance: TOP_SECRET" in bot_events


@pytest.mark.asyncio
async def test_subflow_parser_syntax_variants():
    """Test parser recognition of various subflow invocation syntaxes."""
    script = """
flow master_orchestrator:
  call flow auth_verification
  call compute_metrics(threshold="0.95") as $metrics
  subflow log_audit_trail
  subdialogue customer_escalation
"""
    flows = Colang2Parser.parse(script)
    assert "master_orchestrator" in flows
    master = flows["master_orchestrator"]
    assert len(master.steps) == 4

    step0 = master.steps[0]
    assert isinstance(step0, SubflowStep)
    assert step0.flow_name == "auth_verification"

    step1 = master.steps[1]
    assert isinstance(step1, SubflowStep)
    assert step1.flow_name == "compute_metrics"
    assert step1.arguments == {"threshold": "0.95"}
    assert step1.variable == "metrics"

    step2 = master.steps[2]
    assert isinstance(step2, SubflowStep)
    assert step2.flow_name == "log_audit_trail"

    step3 = master.steps[3]
    assert isinstance(step3, SubflowStep)
    assert step3.flow_name == "customer_escalation"


@pytest.mark.asyncio
async def test_subflow_failure_rollback_isolation():
    """Verify that a failing nested subflow triggers rollback and leaves parent context uncorrupted."""
    bus = EventBus()
    context = ScopedContext(event_bus=bus)
    context.set("parent_status", "clean", scope=ScopeLevel.SESSION)
    engine = Colang2Engine(event_bus=bus, context=context)

    # Missing subflow should cause StepResult(success=False)
    parent_flow = Flow2(name="fragile_flow")
    parent_flow.add_step(AssignStep(variable="parent_status", value="mutated", scope=ScopeLevel.SESSION))
    parent_flow.add_step(SubflowStep(flow_name="non_existent_subflow", enable_rollback_on_failure=True))
    engine.register_flow(parent_flow)

    res = await engine.execute_flow("fragile_flow")
    assert res.success is False
    assert "not found in registered flows" in res.error

    # Session variable should be rolled back to snapshot value "clean"
    assert context.get("parent_status") == "clean"

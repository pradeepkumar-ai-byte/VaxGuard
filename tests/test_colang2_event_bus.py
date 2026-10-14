"""Unit and integration test suite for Colang 2.0 Async Event-Driven State Bus & Engine.

Validates:
1. Event primitives: Event, EventType, EventPattern, EventBus.
2. Async event flow execution: Flow2, ActionStep, WhenStep, MatchStep, ForkStep, JoinStep, SendStep, AwaitStep, AssignStep.
3. Multi-turn dialogue context with scoped variable lifetimes (turn, session, global), snapshots, rollbacks, compensations.
4. Concurrency & Branching: Fork and join semantics (mode="all", mode="any"), multi-actor coordination.
5. Colang 2.0 Parser:
   - when user said "..."
   - send bot say "..."
   - match ...
   - fork / join ...
6. End-to-end Colang2Engine multi-turn execution.
"""

import asyncio
import re
import pytest

from vaxguard.nemo.colang2 import (
    EventType,
    Event,
    EventPattern,
    EventBus,
    Flow2,
    FlowResult,
    ActionStep,
    WhenStep,
    MatchStep,
    ForkStep,
    JoinStep,
    SendStep,
    AwaitStep,
    AssignStep,
    StopStep,
    ScopeLevel,
    ScopedContext,
    ContextSnapshot,
    Colang2Parser,
    Colang2Engine,
    FlowStep,
    StepResult,
    UserUtterance,
    BotUtterance,
    FlowStarted,
    FlowFinished,
    ContextUpdate,
    CustomActionExecution,
)


# ─── 1. EVENT PRIMITIVES TESTS ────────────────────────────────────────────────

def test_event_creation_and_properties():
    ev = Event.user_said("Hello doctor", session_id="s123")
    assert ev.name == "UserUtterance"
    assert ev.event_type == EventType.USER_UTTERANCE
    assert ev.source == "user"
    assert ev.text == "Hello doctor"
    assert ev.get("session_id") == "s123"
    assert "text" in ev
    assert ev["session_id"] == "s123"

    bot_ev = Event.bot_said("I am here to help.")
    assert bot_ev.name == "BotUtterance"
    assert bot_ev.event_type == EventType.BOT_UTTERANCE
    assert bot_ev.source == "bot"
    assert bot_ev.text == "I am here to help."

    started_ev = Event.flow_started("check_vitals")
    assert started_ev.event_type == EventType.FLOW_STARTED
    assert started_ev.flow_name == "check_vitals"

    finished_ev = Event.flow_finished("check_vitals", status="completed")
    assert finished_ev.event_type == EventType.FLOW_FINISHED
    assert finished_ev.flow_name == "check_vitals"
    assert finished_ev.get("status") == "completed"

    action_ev = Event.custom_action("verify_dosage", arguments={"drug": "Paxlovid"})
    assert action_ev.event_type == EventType.CUSTOM_ACTION_EXECUTION
    assert action_ev.action_name == "verify_dosage"
    assert action_ev.get("arguments") == {"drug": "Paxlovid"}


def test_functional_event_constructors():
    u = UserUtterance("Vaccine schedule")
    assert u.text == "Vaccine schedule"
    assert u.event_type == EventType.USER_UTTERANCE

    b = BotUtterance("Here is the schedule")
    assert b.text == "Here is the schedule"
    assert b.event_type == EventType.BOT_UTTERANCE

    fs = FlowStarted("safety_rail")
    assert fs.flow_name == "safety_rail"

    ff = FlowFinished("safety_rail", status="stopped")
    assert ff.flow_name == "safety_rail"
    assert ff.get("status") == "stopped"

    cu = ContextUpdate("patient_id", 42, scope="session")
    assert cu.get("key") == "patient_id"
    assert cu.get("value") == 42
    assert cu.get("scope") == "session"

    ca = CustomActionExecution(action="lookup_icd10", arguments={"code": "Z23"})
    assert ca.action_name == "lookup_icd10"
    assert ca.get("arguments") == {"code": "Z23"}


def test_event_serialization():
    ev = UserUtterance("Test input", note="urgent")
    data = ev.to_dict()
    assert data["name"] == "UserUtterance"
    assert data["event_type"] == EventType.USER_UTTERANCE.value
    assert data["data"]["text"] == "Test input"

    reconstructed = Event.from_dict(data)
    assert reconstructed.event_id == ev.event_id
    assert reconstructed.name == ev.name
    assert reconstructed.text == "Test input"
    assert reconstructed.get("note") == "urgent"


# ─── 2. EVENT PATTERN MATCHING TESTS ──────────────────────────────────────────

def test_event_pattern_matching():
    pat_exact = EventPattern.user_said("hello")
    assert pat_exact.matches(UserUtterance("hello"))
    assert not pat_exact.matches(UserUtterance("goodbye"))
    assert not pat_exact.matches(BotUtterance("hello"))

    # Wildcard glob matching
    pat_wildcard = EventPattern.user_said("*fever*")
    assert pat_wildcard.matches(UserUtterance("I have a high fever today"))
    assert pat_wildcard.matches(UserUtterance("FEVER"))
    assert not pat_wildcard.matches(UserUtterance("I have a cough"))

    # Source matching
    pat_bot = EventPattern(source="bot")
    assert pat_bot.matches(BotUtterance("Response"))
    assert not pat_bot.matches(UserUtterance("Response"))

    # Regex matching
    pat_regex = EventPattern(
        event_type=EventType.USER_UTTERANCE,
        attributes={"text": re.compile(r"COVID-\d{2}", re.IGNORECASE)},
    )
    assert pat_regex.matches(UserUtterance("Information on covid-19"))
    assert not pat_regex.matches(UserUtterance("Information on Influenza"))

    # Callable predicate matching
    pat_pred = EventPattern(
        predicate=lambda e: e.get("severity", 0) > 5
    )
    assert pat_pred.matches(Event(name="Triage", data={"severity": 8}))
    assert not pat_pred.matches(Event(name="Triage", data={"severity": 3}))


def test_event_pattern_bindings():
    pat = EventPattern(attributes={"code": "$code_var", "unit": "mg"})
    ev = Event(name="DosageEvent", data={"code": "MED101", "unit": "mg"})
    bindings = pat.extract_bindings(ev)
    assert bindings == {"code_var": "MED101"}


# ─── 3. ASYNC EVENT BUS TESTS ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_bus_publish_and_subscribe():
    bus = EventBus()
    received = []

    async def _listener(event: Event) -> None:
        received.append(event)

    sub_id = bus.subscribe(EventPattern.user_said("*"), _listener)
    assert sub_id.startswith("sub_")

    await bus.send(UserUtterance("Hi"))
    await bus.send(BotUtterance("Hello"))
    await bus.send(UserUtterance("How are you?"))

    assert len(received) == 2
    assert received[0].text == "Hi"
    assert received[1].text == "How are you?"

    # Unsubscribe
    bus.unsubscribe(sub_id)
    await bus.send(UserUtterance("After unsub"))
    assert len(received) == 2


@pytest.mark.asyncio
async def test_event_bus_wait_for():
    bus = EventBus()

    async def _delayed_publish():
        await asyncio.sleep(0.02)
        await bus.send(BotUtterance("Async response delivered"))

    asyncio.create_task(_delayed_publish())
    event = await bus.wait_for(EventPattern.bot_said("*"), timeout=1.0)
    assert event.text == "Async response delivered"


@pytest.mark.asyncio
async def test_event_bus_wait_for_timeout():
    bus = EventBus()
    with pytest.raises(TimeoutError):
        await bus.wait_for(EventPattern.user_said("never_arriving"), timeout=0.05)


@pytest.mark.asyncio
async def test_event_bus_history_and_filtering():
    bus = EventBus()
    await bus.send(UserUtterance("First"))
    await bus.send(BotUtterance("Second"))
    await bus.send(UserUtterance("Third"))

    assert len(bus.history) == 3
    user_events = bus.filter_history(EventPattern.user_said("*"))
    assert len(user_events) == 2
    assert [e.text for e in user_events] == ["First", "Third"]

    bus.clear_history()
    assert len(bus.history) == 0


# ─── 4. SCOPED VARIABLE LIFETIMES & CONTEXT TESTS ─────────────────────────────

def test_scoped_context_hierarchical_resolution():
    ctx = ScopedContext()
    ctx.set("global_var", "G", scope=ScopeLevel.GLOBAL)
    ctx.set("session_var", "S", scope=ScopeLevel.SESSION)
    ctx.set("turn_var", "T", scope=ScopeLevel.TURN)

    # Hierarchical reads
    assert ctx.get("global_var") == "G"
    assert ctx.get("session_var") == "S"
    assert ctx.get("turn_var") == "T"
    assert ctx.get("nonexistent", "default") == "default"

    # Explicit scope reads
    assert ctx.get("global_var", scope=ScopeLevel.GLOBAL) == "G"
    assert ctx.get("global_var", scope=ScopeLevel.TURN) is None

    # Dotted prefix reads
    assert ctx.get("global.global_var") == "G"
    assert ctx.get("session.session_var") == "S"
    assert ctx.get("turn.turn_var") == "T"

    # Scope precedence: turn shadows session shadows global
    ctx.set("shared", "global_val", scope=ScopeLevel.GLOBAL)
    assert ctx.get("shared") == "global_val"
    ctx.set("shared", "session_val", scope=ScopeLevel.SESSION)
    assert ctx.get("shared") == "session_val"
    ctx.set("shared", "turn_val", scope=ScopeLevel.TURN)
    assert ctx.get("shared") == "turn_val"


def test_scoped_context_new_turn():
    ctx = ScopedContext()
    ctx.set("turn.query", "What are side effects?")
    ctx.set("session.patient_age", 45)
    ctx.set("global.system_mode", "clinical")

    assert ctx.turn_id == 1
    new_id = ctx.new_turn()
    assert new_id == 2
    assert ctx.turn_id == 2

    # Turn variables should be cleared
    assert ctx.get("query") is None
    # Session and Global should persist
    assert ctx.get("patient_age") == 45
    assert ctx.get("system_mode") == "clinical"


@pytest.mark.asyncio
async def test_scoped_context_rollback_and_compensations():
    ctx = ScopedContext()
    ctx.set("session.balance", 100)
    ctx.set("turn.item", "vaccine_dose")

    snapshot = ctx.snapshot()

    # Mutate state
    ctx.set("session.balance", 50)
    ctx.set("turn.item", "modified")

    compensated = []

    def undo_action():
        compensated.append("refund_processed")

    ctx.register_compensation(undo_action)

    # Perform rollback
    ctx.rollback(snapshot)
    await ctx.execute_compensations()

    # Balance restored to 100, item restored to vaccine_dose
    assert ctx.get("session.balance") == 100
    assert ctx.get("turn.item") == "vaccine_dose"
    assert compensated == ["refund_processed"]


# ─── 5. ASYNC FLOW STEP EXECUTION TESTS ───────────────────────────────────────

@pytest.mark.asyncio
async def test_send_and_match_step_execution():
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    flow = Flow2(name="test_send_flow")
    flow.add_step(SendStep(event_name="BotUtterance", source="bot", data={"text": "Greetings"}))

    res = await flow.execute(ctx, bus)
    assert res.success is True
    assert res.status == "completed"

    # Verify event was published
    assert len(bus.history) >= 3  # FlowStarted, BotUtterance, FlowFinished
    bot_events = bus.filter_history(EventPattern.bot_said("*"))
    assert len(bot_events) == 1
    assert bot_events[0].text == "Greetings"


@pytest.mark.asyncio
async def test_action_step_execution():
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    async def mock_calculator(a: int, b: int) -> int:
        return int(a) + int(b)

    custom_actions = {"add_numbers": mock_calculator}

    flow = Flow2(name="math_flow")
    flow.add_step(ActionStep(action_name="add_numbers", arguments={"a": 10, "b": 25}, variable="sum_result"))

    res = await flow.execute(ctx, bus, custom_actions=custom_actions)
    assert res.success is True
    assert ctx.get("sum_result") == 35

    # Check that CustomActionExecution events were sent
    action_events = bus.filter_history(EventPattern.action("add_numbers"))
    assert len(action_events) == 2  # started, finished
    assert action_events[0].get("status") == "started"
    assert action_events[1].get("status") == "finished"
    assert action_events[1].get("result") == 35


@pytest.mark.asyncio
async def test_when_step_event_listener():
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    triggered = []

    class RecordStep(FlowStep):
        async def execute(self, f_ctx):
            triggered.append(f_ctx.context.get("user_input"))
            await f_ctx.event_bus.send(BotUtterance(f"Acknowledged: {f_ctx.context.get('user_input')}"))
            return StepResult(success=True)

    when_step = WhenStep(
        pattern=EventPattern.user_said("*emergency*"),
        steps=[RecordStep()],
        persistent=True,
    )

    flow = Flow2(name="emergency_guard")
    flow.add_step(when_step)

    # Initialize flow listener
    await flow.execute(ctx, bus)

    # Send matching user utterance
    await bus.send(UserUtterance("This is a medical emergency!"))
    await asyncio.sleep(0.01)

    assert len(triggered) == 1
    assert "emergency" in triggered[0].lower()

    # Verify bot acknowledged
    bot_replies = bus.filter_history(EventPattern.bot_said("*"))
    assert any("Acknowledged" in b.text for b in bot_replies)


# ─── 6. CONCURRENCY & BRANCHING (FORK / JOIN) TESTS ───────────────────────────

@pytest.mark.asyncio
async def test_fork_and_join_all_branches():
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    records = []

    async def branch_task(branch_id: str, delay: float):
        await asyncio.sleep(delay)
        records.append(branch_id)
        return branch_id

    class BranchStep(FlowStep):
        def __init__(self, bid: str, d: float):
            self.bid = bid
            self.d = d

        async def execute(self, f_ctx):
            res = await branch_task(self.bid, self.d)
            return StepResult(success=True, output=res)

    flow = Flow2(name="concurrency_flow")
    flow.add_step(ForkStep(branch_name="branch_fast", steps=[BranchStep("fast", 0.01)]))
    flow.add_step(ForkStep(branch_name="branch_slow", steps=[BranchStep("slow", 0.03)]))
    flow.add_step(JoinStep(mode="all"))

    res = await flow.execute(ctx, bus)
    assert res.success is True
    assert "fast" in records
    assert "slow" in records
    assert len(records) == 2


@pytest.mark.asyncio
async def test_fork_and_join_any_mode():
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    records = []

    class BranchStep(FlowStep):
        def __init__(self, bid: str, d: float):
            self.bid = bid
            self.d = d

        async def execute(self, f_ctx):
            await asyncio.sleep(self.d)
            records.append(self.bid)
            return StepResult(success=True, output=self.bid)

    flow = Flow2(name="race_flow")
    flow.add_step(ForkStep(branch_name="racer1", steps=[BranchStep("racer1", 0.01)]))
    flow.add_step(ForkStep(branch_name="racer2", steps=[BranchStep("racer2", 0.1)]))
    flow.add_step(JoinStep(mode="any"))

    res = await flow.execute(ctx, bus)
    assert res.success is True
    assert "racer1" in records


# ─── 7. COLANG 2.0 FULL PARSER TESTS ──────────────────────────────────────────

def test_colang2_parser_full_script():
    colang_code = """
    # Colang 2.0 Policy Definition
    flow greeting_flow
      when user said "hello"
        send bot say "Hello! How can I help you today?"

    flow allergy_check
      match user said "*allergic to penicillin*"
      $turn.allergy_flag = true
      send bot say "Noted penicillin allergy."
      stop

    flow parallel_investigation
      fork flow check_ehr
      fork flow check_contraindications
      join all

    flow check_dosage
      action calculate_dosage(weight="70", drug="amoxicillin")
    """

    flows = Colang2Parser.parse(colang_code)

    assert "greeting_flow" in flows
    assert "allergy_check" in flows
    assert "parallel_investigation" in flows
    assert "check_dosage" in flows

    # Validate greeting_flow structure
    g_flow = flows["greeting_flow"]
    assert len(g_flow.steps) == 1
    assert isinstance(g_flow.steps[0], WhenStep)
    assert g_flow.steps[0].pattern.event_type == EventType.USER_UTTERANCE
    assert len(g_flow.steps[0].steps) == 1
    assert isinstance(g_flow.steps[0].steps[0], SendStep)

    # Validate allergy_check structure
    a_flow = flows["allergy_check"]
    assert isinstance(a_flow.steps[0], MatchStep)
    assert isinstance(a_flow.steps[1], AssignStep)
    assert a_flow.steps[1].variable == "allergy_flag"
    assert isinstance(a_flow.steps[2], SendStep)
    assert isinstance(a_flow.steps[3], StopStep)

    # Validate parallel_investigation
    p_flow = flows["parallel_investigation"]
    assert isinstance(p_flow.steps[0], ForkStep)
    assert p_flow.steps[0].branch_name == "check_ehr"
    assert isinstance(p_flow.steps[1], ForkStep)
    assert p_flow.steps[1].branch_name == "check_contraindications"
    assert isinstance(p_flow.steps[2], JoinStep)
    assert p_flow.steps[2].mode == "all"


# ─── 8. END-TO-END COLANG2 ENGINE INTERACTION TESTS ───────────────────────────

@pytest.mark.asyncio
async def test_colang2_engine_end_to_end_dialogue():
    engine = Colang2Engine()

    script = """
    flow triage_greeting
      when user said "hello*"
        send bot say "Welcome to VaxGuard Clinical Portal."

    flow fever_alert
      when user said "*fever*"
        send bot say "Please measure your temperature and contact a clinician if above 101F."
    """

    engine.load_flows_from_text(script)
    await engine.initialize_listeners()

    # First turn
    replies = await engine.send_user_utterance("Hello doctor")
    assert any("Welcome to VaxGuard" in r for r in replies)

    # Advance turn
    engine.new_turn()

    # Second turn with fever
    fever_replies = await engine.send_user_utterance("I feel a high fever")
    assert any("temperature" in r for r in fever_replies)


@pytest.mark.asyncio
async def test_colang2_engine_with_custom_action():
    engine = Colang2Engine()

    async def get_patient_record(patient_id: str):
        return {"id": patient_id, "status": "vaccinated"}

    engine.register_action("get_patient_record", get_patient_record)

    flow = Flow2(name="fetch_patient")
    flow.add_step(ActionStep("get_patient_record", arguments={"patient_id": "P-99"}, variable="record"))
    flow.add_step(SendStep(event_name="BotUtterance", source="bot", data={"text": "Patient record fetched"}))

    engine.register_flow(flow)
    result = await engine.execute_flow("fetch_patient")

    assert result.success is True
    assert engine.context.get("record") == {"id": "P-99", "status": "vaccinated"}


# ─── 9. ADVANCED MULTI-ACTOR & COMPENSATION TESTS ─────────────────────────────

@pytest.mark.asyncio
async def test_multi_actor_dynamic_coordination():
    """Validates coordination between user, bot, clinician, and system actors."""
    bus = EventBus()
    actor_logs = []

    async def log_actor(event: Event):
        actor_logs.append((event.source, event.name, event.data))

    bus.subscribe(EventPattern(), log_actor)

    # 1. User asks a question
    await bus.send(Event(name="UserUtterance", source="user", data={"text": "Requesting doctor review"}))
    # 2. System flags triage
    await bus.send(Event(name="TriageFlagged", source="system", data={"priority": "high"}))
    # 3. Clinician overrides/approves
    await bus.send(Event(name="ClinicianApproval", source="clinician", data={"approved": True, "notes": "Approved"}))
    # 4. Bot notifies patient
    await bus.send(Event(name="BotUtterance", source="bot", data={"text": "Your clinician has approved the request."}))

    assert len(actor_logs) == 4
    sources = [entry[0] for entry in actor_logs]
    assert sources == ["user", "system", "clinician", "bot"]


@pytest.mark.asyncio
async def test_flow_automatic_rollback_on_step_failure():
    """Validates that a step failure triggers state rollback and compensation transitions."""
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    ctx.set("session.account_balance", 200)
    ctx.set("turn.step_progress", 0)

    compensation_calls = []

    def rollback_hook():
        compensation_calls.append("undo_payment")

    class FailingStep(FlowStep):
        async def execute(self, f_ctx):
            f_ctx.context.set("session.account_balance", 50)  # Mutate
            f_ctx.context.register_compensation(rollback_hook)
            return StepResult(success=False, error="Simulated payment failure")

    flow = Flow2(name="transaction_flow")
    flow.add_step(AssignStep(variable="step_progress", value=1))
    flow.add_step(FailingStep())
    flow.add_step(AssignStep(variable="step_progress", value=2))

    res = await flow.execute(ctx, bus, enable_rollback_on_failure=True)
    assert res.success is False
    assert res.status == "failed"
    assert "Simulated payment failure" in (res.error or "")

    # Context should be rolled back to original state
    assert ctx.get("session.account_balance") == 200
    assert ctx.get("turn.step_progress") == 0
    # Compensation should have fired
    assert compensation_calls == ["undo_payment"]


@pytest.mark.asyncio
async def test_template_variable_interpolation():
    """Validates variable interpolation $var and {var} in SendStep and ActionStep."""
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)
    ctx.set("patient_name", "Alice")
    ctx.set("session.clinic", "HealthFirst")

    flow = Flow2(name="template_flow")
    flow.add_step(
        SendStep(
            event_name="BotUtterance",
            source="bot",
            data={"text": "Hello $patient_name, welcome to {clinic}!"}
        )
    )

    res = await flow.execute(ctx, bus)
    assert res.success is True

    bot_events = bus.filter_history(EventPattern.bot_said("*"))
    assert len(bot_events) == 1
    assert bot_events[0].text == "Hello Alice, welcome to HealthFirst!"


@pytest.mark.asyncio
async def test_match_step_with_variable_capture():
    """Validates MatchStep captures matched event payload into designated variable."""
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    flow = Flow2(name="match_capture_flow")
    flow.add_step(MatchStep(pattern=EventPattern.user_said("*"), variable="user_captured"))

    async def _send_input():
        await asyncio.sleep(0.01)
        await bus.send(UserUtterance("Captured message payload"))

    asyncio.create_task(_send_input())
    res = await flow.execute(ctx, bus)

    assert res.success is True
    assert ctx.get("user_captured") == "Captured message payload"


def test_scoped_context_delete_and_has():
    """Validates scoped context variable deletion and existence checking."""
    ctx = ScopedContext()
    ctx.set("turn.temp", 98.6)
    ctx.set("session.patient_name", "Bob")

    assert ctx.has("temp") is True
    assert ctx.has("patient_name") is True
    assert ctx.has("nonexistent") is False

    all_v = ctx.all_variables()
    assert "temp" in all_v
    assert "patient_name" in all_v

    assert ctx.delete("temp") is True
    assert ctx.has("temp") is False
    assert ctx.get("temp") is None
    assert ctx.has("patient_name") is True


@pytest.mark.asyncio
async def test_stop_step_halts_subsequent_steps():
    """Validates that StopStep immediately halts flow execution."""
    bus = EventBus()
    ctx = ScopedContext(event_bus=bus)

    executed_steps = []

    class TrackStep(FlowStep):
        def __init__(self, step_id: str):
            self.step_id = step_id

        async def execute(self, f_ctx):
            executed_steps.append(self.step_id)
            return StepResult(success=True)

    flow = Flow2(name="halting_flow")
    flow.add_step(TrackStep("step1"))
    flow.add_step(StopStep())
    flow.add_step(TrackStep("step2"))

    res = await flow.execute(ctx, bus)
    assert res.success is True
    assert res.status == "stopped"
    assert executed_steps == ["step1"]


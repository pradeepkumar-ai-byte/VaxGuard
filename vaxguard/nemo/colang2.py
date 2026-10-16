"""VaxGuard NeMo Guardrails: Colang 2.0 Async Event-Driven State Bus & Flow Engine.

100% pure Python implementation of NVIDIA Colang 2.0 asynchronous event-driven model:
- Event primitives: Event, EventType, EventPattern, EventBus
- Async event flow execution: Flow2, ActionStep, WhenStep, MatchStep, ForkStep, JoinStep, SendStep, AwaitStep, AssignStep
- Multi-turn dialogue context with scoped variable lifetimes (turn, session, global)
- Concurrency & Branching: Flow fork, join, dynamic multi-actor coordination
- State Rollback & Compensation transitions
- Full parser for Colang 2.0 event syntax:
  `when user said "..."`
  `send bot say "..."`
  `match ...`
  `fork ...`
  `join ...`
  `await ...`

Zero heavy external binaries. Fully asynchronous with native asyncio.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import inspect
import logging
import re
import time
import uuid
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field

logger = logging.getLogger("vaxguard.nemo.colang2")


# ─── 1. EVENT PRIMITIVES & ENUMS ──────────────────────────────────────────────

class EventType(str, Enum):
    """Categorization of Colang 2.0 dialogue and engine events."""
    USER_UTTERANCE = "UserUtterance"
    BOT_UTTERANCE = "BotUtterance"
    FLOW_STARTED = "FlowStarted"
    FLOW_FINISHED = "FlowFinished"
    CONTEXT_UPDATE = "ContextUpdate"
    CUSTOM_ACTION_EXECUTION = "CustomActionExecution"
    SYSTEM = "SystemEvent"
    CUSTOM = "CustomEvent"

    @classmethod
    def from_str(cls, val: str) -> EventType:
        """Resolve various syntax variations to canonical EventType."""
        if isinstance(val, EventType):
            return val
        s = str(val).strip()
        for member in cls:
            if s.lower() == member.value.lower():
                return member
        v = s.lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if v == member.name.lower():
                return member
        if v in ("user_said", "user_utterance", "userutterance", "usersaid", "user"):
            return cls.USER_UTTERANCE
        if v in ("bot_say", "bot_said", "bot_utterance", "botutterance", "botsay", "botsaid", "bot"):
            return cls.BOT_UTTERANCE
        if v in ("flow_started", "flowstarted", "flow_start"):
            return cls.FLOW_STARTED
        if v in ("flow_finished", "flowfinished", "flow_completed", "flowcompleted", "flow_finish", "flow_end"):
            return cls.FLOW_FINISHED
        if v in ("context_update", "contextupdate", "context_updated"):
            return cls.CONTEXT_UPDATE
        if v in ("custom_action_execution", "customactionexecution", "action_execution", "action_started", "action_finished", "action"):
            return cls.CUSTOM_ACTION_EXECUTION
        if v in ("system", "system_event", "systemevent"):
            return cls.SYSTEM
        return cls.CUSTOM


class Event(BaseModel):
    """Discrete event in the Colang 2.0 asynchronous event bus."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = "UserUtterance"
    event_type: EventType = EventType.USER_UTTERANCE
    source: str = "user"  # "user", "bot", "system", "action", "flow:<flow_name>"
    target: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, **data: Any) -> None:
        if "name" in data and "event_type" not in data:
            data["event_type"] = EventType.from_str(str(data["name"]))
        elif "event_type" in data and "name" not in data:
            et = data["event_type"]
            data["name"] = et.value if isinstance(et, EventType) else str(et)
        super().__init__(**data)

    @property
    def text(self) -> str:
        """Convenience property for text payload."""
        return str(self.data.get("text") or self.data.get("utterance") or "")

    @property
    def action_name(self) -> str:
        """Convenience property for action payload."""
        return str(self.data.get("action") or self.data.get("action_name") or "")

    @property
    def flow_name(self) -> str:
        """Convenience property for flow payload."""
        return str(self.data.get("flow_name") or self.data.get("flow") or "")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "event_type": self.event_type.value,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Event:
        return cls(
            event_id=d.get("event_id", str(uuid.uuid4())[:12]),
            name=d.get("name", "UserUtterance"),
            event_type=EventType(d.get("event_type", EventType.USER_UTTERANCE.value)),
            source=d.get("source", "user"),
            target=d.get("target"),
            data=d.get("data", {}),
            timestamp=d.get("timestamp", time.time()),
        )

    # ── Convenient Event Factory Helpers ──
    @classmethod
    def user_said(cls, text: str, **kwargs: Any) -> Event:
        data = dict(kwargs)
        data["text"] = text
        return cls(name="UserUtterance", event_type=EventType.USER_UTTERANCE, source="user", data=data)

    @classmethod
    def bot_said(cls, text: str, **kwargs: Any) -> Event:
        data = dict(kwargs)
        data["text"] = text
        return cls(name="BotUtterance", event_type=EventType.BOT_UTTERANCE, source="bot", data=data)

    @classmethod
    def flow_started(cls, flow_name: str, **kwargs: Any) -> Event:
        data = dict(kwargs)
        data["flow_name"] = flow_name
        return cls(name="FlowStarted", event_type=EventType.FLOW_STARTED, source=f"flow:{flow_name}", data=data)

    @classmethod
    def flow_finished(cls, flow_name: str, status: str = "completed", **kwargs: Any) -> Event:
        data = dict(kwargs)
        data["flow_name"] = flow_name
        data["status"] = status
        return cls(name="FlowFinished", event_type=EventType.FLOW_FINISHED, source=f"flow:{flow_name}", data=data)

    @classmethod
    def context_update(cls, key: str, value: Any, scope: str = "turn", **kwargs: Any) -> Event:
        data = dict(kwargs)
        data.update({"key": key, "value": value, "scope": scope})
        return cls(name="ContextUpdate", event_type=EventType.CONTEXT_UPDATE, source="system", data=data)

    @classmethod
    def custom_action(cls, action: str, arguments: Optional[Dict[str, Any]] = None, status: str = "started", result: Any = None, **kwargs: Any) -> Event:
        data = dict(kwargs)
        data.update({"action": action, "arguments": arguments or {}, "status": status, "result": result})
        return cls(name="CustomActionExecution", event_type=EventType.CUSTOM_ACTION_EXECUTION, source="action", data=data)


# Functional alias constructors
def UserUtterance(text: str, **kwargs: Any) -> Event:
    return Event.user_said(text=text, **kwargs)

def BotUtterance(text: str, **kwargs: Any) -> Event:
    return Event.bot_said(text=text, **kwargs)

def FlowStarted(flow_name: str, **kwargs: Any) -> Event:
    return Event.flow_started(flow_name=flow_name, **kwargs)

def FlowFinished(flow_name: str, status: str = "completed", **kwargs: Any) -> Event:
    return Event.flow_finished(flow_name=flow_name, status=status, **kwargs)

def ContextUpdate(key: str, value: Any, scope: str = "turn", **kwargs: Any) -> Event:
    return Event.context_update(key=key, value=value, scope=scope, **kwargs)

def CustomActionExecution(action: str, arguments: Optional[Dict[str, Any]] = None, status: str = "started", result: Any = None, **kwargs: Any) -> Event:
    return Event.custom_action(action=action, arguments=arguments, status=status, result=result, **kwargs)


# ─── 2. EVENT PATTERN & PATTERN MATCHING ──────────────────────────────────────

class EventPattern:
    """Colang 2.0 Event Pattern matching criteria.

    Supports:
    - Event name/type matching (exact, wildcard `*`, or EventType)
    - Source/Actor matching
    - Attribute / argument matching (exact, glob wildcard `*`, regex, callable predicate)
    - Custom predicate function
    """

    def __init__(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        predicate: Optional[Callable[[Event], bool]] = None,
    ) -> None:
        self.event_type = event_type
        self.source = source
        self.target = target
        self.attributes = attributes or {}
        self.predicate = predicate

    def matches(self, event: Event) -> bool:
        """Check whether the given event matches this pattern."""
        # 1. Check Event Type / Name
        if self.event_type is not None:
            if isinstance(self.event_type, EventType):
                if event.event_type != self.event_type:
                    return False
            else:
                target_str = str(self.event_type).strip()
                if target_str != "*":
                    # Check canonical match
                    target_et = EventType.from_str(target_str)
                    if target_et != EventType.CUSTOM:
                        if event.event_type != target_et:
                            return False
                    else:
                        # Check direct string / wildcard match against name
                        if not fnmatch.fnmatch(event.name.lower(), target_str.lower()):
                            return False

        # 2. Check Source
        if self.source is not None and self.source != "*":
            if not fnmatch.fnmatch(event.source.lower(), self.source.lower()):
                return False

        # 3. Check Target
        if self.target is not None and self.target != "*":
            if not event.target or not fnmatch.fnmatch(event.target.lower(), self.target.lower()):
                return False

        # 4. Check Data Attributes
        for key, expected_val in self.attributes.items():
            actual_val = event.data.get(key)
            if callable(expected_val):
                try:
                    if not expected_val(actual_val):
                        return False
                except Exception:
                    return False
            elif isinstance(expected_val, Pattern):
                if actual_val is None or not expected_val.search(str(actual_val)):
                    return False
            elif isinstance(expected_val, str) and ("*" in expected_val or "?" in expected_val):
                if actual_val is None or not fnmatch.fnmatch(str(actual_val).lower(), expected_val.lower()):
                    return False
            else:
                # Comparison: if both are strings, case-insensitive equality or substring if specified
                if isinstance(expected_val, str) and isinstance(actual_val, str):
                    if expected_val.lower() != actual_val.lower():
                        return False
                elif actual_val != expected_val:
                    return False

        # 5. Check Custom Predicate
        if self.predicate is not None:
            try:
                if not self.predicate(event):
                    return False
            except Exception:
                return False

        return True

    def extract_bindings(self, event: Event) -> Dict[str, Any]:
        """Extract matching bindings from event data into variables."""
        bindings: Dict[str, Any] = {}
        for key, val in self.attributes.items():
            if isinstance(val, str) and val.startswith("$"):
                var_name = val[1:]
                bindings[var_name] = event.data.get(key)
        return bindings

    @classmethod
    def user_said(cls, pattern: str = "*") -> EventPattern:
        """Create pattern matching user utterances with text pattern."""
        attrs = {} if pattern == "*" else {"text": pattern}
        return cls(event_type=EventType.USER_UTTERANCE, source="user", attributes=attrs)

    @classmethod
    def bot_said(cls, pattern: str = "*") -> EventPattern:
        """Create pattern matching bot utterances with text pattern."""
        attrs = {} if pattern == "*" else {"text": pattern}
        return cls(event_type=EventType.BOT_UTTERANCE, source="bot", attributes=attrs)

    @classmethod
    def flow_started(cls, flow_name: str = "*") -> EventPattern:
        attrs = {} if flow_name == "*" else {"flow_name": flow_name}
        return cls(event_type=EventType.FLOW_STARTED, attributes=attrs)

    @classmethod
    def flow_finished(cls, flow_name: str = "*", status: str = "*") -> EventPattern:
        attrs = {}
        if flow_name != "*":
            attrs["flow_name"] = flow_name
        if status != "*":
            attrs["status"] = status
        return cls(event_type=EventType.FLOW_FINISHED, attributes=attrs)

    @classmethod
    def action(cls, action_name: str = "*") -> EventPattern:
        attrs = {} if action_name == "*" else {"action": action_name}
        return cls(event_type=EventType.CUSTOM_ACTION_EXECUTION, attributes=attrs)

    def __repr__(self) -> str:
        return f"EventPattern(type={self.event_type}, source={self.source}, attrs={self.attributes})"


# ─── 3. ASYNC EVENT BUS ───────────────────────────────────────────────────────

class EventBus:
    """High-performance, asynchronous Event Bus for Colang 2.0 state flows."""

    def __init__(self) -> None:
        self._subscribers: List[Tuple[str, EventPattern, Callable[[Event], Awaitable[None]]]] = []
        self._waiters: List[Tuple[EventPattern, asyncio.Future[Event]]] = []
        self._history: List[Event] = []
        self._lock = asyncio.Lock()

    @property
    def history(self) -> List[Event]:
        """Chronological audit log of all dispatched events."""
        return list(self._history)

    async def send(self, event: Event) -> None:
        """Publish an event to the state bus.

        Dispatches to all matching registered listeners and completes pending waiters.
        """
        async with self._lock:
            self._history.append(event)
            # Find and resolve all waiters matching pattern
            remaining_waiters: List[Tuple[EventPattern, asyncio.Future[Event]]] = []
            for pattern, fut in self._waiters:
                if not fut.done() and pattern.matches(event):
                    fut.set_result(event)
                else:
                    if not fut.done():
                        remaining_waiters.append((pattern, fut))
            self._waiters = remaining_waiters

            # Copy subscribers to iterate safely
            subscribers_to_invoke = [
                (sub_id, handler)
                for (sub_id, pattern, handler) in self._subscribers
                if pattern.matches(event)
            ]

        # Dispatch subscriber handlers concurrently
        if subscribers_to_invoke:
            tasks = [
                asyncio.create_task(self._safe_invoke_handler(handler, event))
                for (_, handler) in subscribers_to_invoke
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_invoke_handler(
        self,
        handler: Callable[[Event], Awaitable[None]],
        event: Event
    ) -> None:
        try:
            res = handler(event)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            logger.error("Error in Colang 2.0 subscriber callback: %s", e, exc_info=True)

    async def wait_for(
        self,
        pattern: Union[EventPattern, EventType, str],
        timeout: Optional[float] = None
    ) -> Event:
        """Asynchronously await the next event on the bus matching pattern.

        Supports `await <event>` and `match <pattern>`.
        """
        if isinstance(pattern, str):
            p = EventPattern(event_type=pattern)
        elif isinstance(pattern, EventType):
            p = EventPattern(event_type=pattern)
        else:
            p = pattern

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Event] = loop.create_future()

        async with self._lock:
            self._waiters.append((p, fut))

        try:
            if timeout is not None:
                return await asyncio.wait_for(fut, timeout=timeout)
            return await fut
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiters = [(pat, f) for (pat, f) in self._waiters if f is not fut]
            raise TimeoutError(f"Timed out after {timeout}s awaiting event matching {p}")

    def subscribe(
        self,
        pattern: EventPattern,
        handler: Callable[[Event], Awaitable[None]],
        sub_id: Optional[str] = None
    ) -> str:
        """Register an active event listener handler (`when <pattern>`)."""
        sid = sub_id or f"sub_{uuid.uuid4().hex[:8]}"
        self._subscribers.append((sid, pattern, handler))
        return sid

    def unsubscribe(self, sub_id: str) -> bool:
        """Remove a subscriber by its ID."""
        initial_count = len(self._subscribers)
        self._subscribers = [s for s in self._subscribers if s[0] != sub_id]
        return len(self._subscribers) < initial_count

    def clear_history(self) -> None:
        """Clear event log."""
        self._history.clear()

    def filter_history(self, pattern: EventPattern) -> List[Event]:
        """Filter event log using an EventPattern."""
        return [e for e in self._history if pattern.matches(e)]


# ─── 4. SCOPED VARIABLE LIFETIMES & DIALOGUE CONTEXT ─────────────────────────

class ScopeLevel(str, Enum):
    """Scoped variable lifetime tiers."""
    TURN = "turn"          # Current interaction turn only; reset on new_turn()
    SESSION = "session"    # Conversation session; persists across multiple turns
    GLOBAL = "global"      # Cross-session / application-wide lifetime


class ContextSnapshot(BaseModel):
    """Deep snapshot for state rollback and compensation transitions."""
    turn_vars: Dict[str, Any]
    session_vars: Dict[str, Any]
    global_vars: Dict[str, Any]
    turn_id: int
    timestamp: float = Field(default_factory=time.time)


class ScopedContext:
    """Multi-turn dialogue state container with variable lifetime scoping and rollback.

    Hierarchical lookup resolution: `turn` -> `session` -> `global`.
    Explicit scope routing: `turn.var_name`, `session.var_name`, `global.var_name`.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        global_vars: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id: str = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        self.turn_id: int = 1
        self.event_bus = event_bus

        self._turn_vars: Dict[str, Any] = {}
        self._session_vars: Dict[str, Any] = {}
        self._global_vars: Dict[str, Any] = dict(global_vars or {})
        self._compensations: List[Callable[[], Union[None, Awaitable[None]]]] = []

    def _parse_key_scope(self, key: str, default_scope: ScopeLevel = ScopeLevel.TURN) -> Tuple[str, ScopeLevel]:
        clean = key.strip()
        if clean.startswith("$"):
            clean = clean[1:]
        if clean.startswith("turn."):
            return clean[len("turn."):], ScopeLevel.TURN
        if clean.startswith("session."):
            return clean[len("session."):], ScopeLevel.SESSION
        if clean.startswith("global."):
            return clean[len("global."):], ScopeLevel.GLOBAL
        return clean, default_scope

    def get(self, key: str, default: Any = None, scope: Optional[ScopeLevel] = None) -> Any:
        """Retrieve variable value hierarchically or from explicit scope."""
        parsed_name, resolved_scope = self._parse_key_scope(key, default_scope=scope or ScopeLevel.TURN)
        
        # If explicit scope was specified or parsed via prefix:
        if scope is not None or (key.startswith(("turn.", "session.", "global.", "$turn.", "$session.", "$global."))):
            target_scope = resolved_scope
            if target_scope == ScopeLevel.TURN:
                return self._turn_vars.get(parsed_name, default)
            elif target_scope == ScopeLevel.SESSION:
                return self._session_vars.get(parsed_name, default)
            elif target_scope == ScopeLevel.GLOBAL:
                return self._global_vars.get(parsed_name, default)

        # Hierarchical lookup: turn -> session -> global
        if parsed_name in self._turn_vars:
            return self._turn_vars[parsed_name]
        if parsed_name in self._session_vars:
            return self._session_vars[parsed_name]
        if parsed_name in self._global_vars:
            return self._global_vars[parsed_name]
        return default

    def set(self, key: str, value: Any, scope: ScopeLevel = ScopeLevel.TURN) -> None:
        """Store variable into designated scope."""
        parsed_name, resolved_scope = self._parse_key_scope(key, default_scope=scope)
        if resolved_scope == ScopeLevel.TURN:
            self._turn_vars[parsed_name] = value
        elif resolved_scope == ScopeLevel.SESSION:
            self._session_vars[parsed_name] = value
        elif resolved_scope == ScopeLevel.GLOBAL:
            self._global_vars[parsed_name] = value

    def has(self, key: str, scope: Optional[ScopeLevel] = None) -> bool:
        """Check if variable exists."""
        return self.get(key, default=None, scope=scope) is not None

    def delete(self, key: str, scope: Optional[ScopeLevel] = None) -> bool:
        """Delete variable from designated or all scopes."""
        parsed_name, resolved_scope = self._parse_key_scope(key, default_scope=scope or ScopeLevel.TURN)
        deleted = False
        if scope is not None or key.startswith(("turn.", "session.", "global.", "$")):
            d = {
                ScopeLevel.TURN: self._turn_vars,
                ScopeLevel.SESSION: self._session_vars,
                ScopeLevel.GLOBAL: self._global_vars
            }[resolved_scope]
            if parsed_name in d:
                del d[parsed_name]
                return True
            return False

        for d in (self._turn_vars, self._session_vars, self._global_vars):
            if parsed_name in d:
                del d[parsed_name]
                deleted = True
        return deleted

    def new_turn(self) -> int:
        """Advance dialogue turn: resets turn-scoped variables while preserving session & global."""
        self._turn_vars.clear()
        self.turn_id += 1
        return self.turn_id

    def snapshot(self) -> ContextSnapshot:
        """Create a deep snapshot for potential rollback."""
        return ContextSnapshot(
            turn_vars=copy.deepcopy(self._turn_vars),
            session_vars=copy.deepcopy(self._session_vars),
            global_vars=copy.deepcopy(self._global_vars),
            turn_id=self.turn_id,
        )

    def rollback(self, snapshot: ContextSnapshot) -> None:
        """Rollback state to specified snapshot."""
        self._turn_vars = copy.deepcopy(snapshot.turn_vars)
        self._session_vars = copy.deepcopy(snapshot.session_vars)
        self._global_vars = copy.deepcopy(snapshot.global_vars)
        self.turn_id = snapshot.turn_id

    def create_child_context(self) -> ScopedContext:
        """Create a child ScopedContext sharing session and global scopes with isolated local turn scope."""
        child = ScopedContext(
            session_id=self.session_id,
            event_bus=self.event_bus,
            global_vars=self._global_vars,
        )
        child._session_vars = self._session_vars
        child._turn_vars = copy.deepcopy(self._turn_vars)
        child.turn_id = self.turn_id
        return child

    def register_compensation(self, fn: Callable[[], Union[None, Awaitable[None]]]) -> None:
        """Register an undo/compensation handler for rollback transitions."""
        self._compensations.append(fn)

    async def execute_compensations(self) -> None:
        """Execute registered compensation handlers in LIFO order."""
        while self._compensations:
            comp = self._compensations.pop()
            try:
                res = comp()
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.error("Error executing compensation handler: %s", e, exc_info=True)

    def all_variables(self) -> Dict[str, Any]:
        """Combined dictionary of active variables with turn overriding session overriding global."""
        combined: Dict[str, Any] = {}
        combined.update(self._global_vars)
        combined.update(self._session_vars)
        combined.update(self._turn_vars)
        return combined


# ─── 5. ASYNC FLOW EXECUTION STEPS & CONCURRENCY ─────────────────────────────

class StepResult(BaseModel):
    """Result returned by execution of a flow step."""
    success: bool = True
    stop_flow: bool = False
    output: Any = None
    error: Optional[str] = None


class FlowExecutionContext:
    """Execution context passed to each step in a running flow."""

    def __init__(
        self,
        flow_name: str,
        flow_id: str,
        context: ScopedContext,
        event_bus: EventBus,
        action_registry: Optional[Any] = None,
        custom_actions: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> None:
        self.flow_name = flow_name
        self.flow_id = flow_id
        self.context = context
        self.event_bus = event_bus
        self.action_registry = action_registry
        self.custom_actions: Dict[str, Callable[..., Any]] = custom_actions or {}
        self.forked_tasks: Dict[str, asyncio.Task[Any]] = {}

    def resolve_template(self, text: str) -> str:
        """Interpolate $var or {var} in string using current context variables."""
        if not text:
            return ""
        all_vars = self.context.all_variables()

        def _sub_dollar(m: re.Match[str]) -> str:
            var_name = m.group(1)
            return str(self.context.get(var_name, m.group(0)))

        # Substitute $var_name or $scope.var_name
        res = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_\.]*)", _sub_dollar, text)

        # Substitute {var_name}
        def _sub_curly(m: re.Match[str]) -> str:
            var_name = m.group(1)
            val = self.context.get(var_name, None)
            return str(val) if val is not None else m.group(0)

        res = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_\.]*)\}", _sub_curly, res)
        return res


class FlowStep:
    """Abstract base class for Colang 2.0 flow steps."""

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        raise NotImplementedError


class SendStep(FlowStep):
    """Dispatches an event onto the EventBus (`send bot say "..."`, `send <event>`)."""

    def __init__(
        self,
        event_name: str = "BotUtterance",
        event_type: Optional[EventType] = None,
        source: str = "bot",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.event_name = event_name
        self.event_type = event_type or EventType.from_str(event_name)
        self.source = source
        self.data = data or {}

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        # Interpolate variables in data strings
        interpolated_data: Dict[str, Any] = {}
        for k, v in self.data.items():
            if isinstance(v, str):
                interpolated_data[k] = ctx.resolve_template(v)
            else:
                interpolated_data[k] = v

        event = Event(
            name=self.event_name,
            event_type=self.event_type,
            source=self.source,
            data=interpolated_data,
        )
        await ctx.event_bus.send(event)
        return StepResult(success=True, output=event)


class MatchStep(FlowStep):
    """Waits for an incoming event matching the pattern (`match user said "..."`)."""

    def __init__(
        self,
        pattern: EventPattern,
        variable: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.pattern = pattern
        self.variable = variable
        self.timeout = timeout

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        try:
            event = await ctx.event_bus.wait_for(self.pattern, timeout=self.timeout)
            if self.variable:
                ctx.context.set(self.variable, event.data.get("text", event.data))
            # Extract pattern bindings if any
            bindings = self.pattern.extract_bindings(event)
            for k, v in bindings.items():
                ctx.context.set(k, v)
            return StepResult(success=True, output=event)
        except Exception as e:
            return StepResult(success=False, error=str(e))


class WhenStep(FlowStep):
    """Active event listener primitive (`when user said "..."`).

    Can execute immediately if an event matches, or register an asynchronous
    listener on the bus that triggers the enclosed sub-steps.
    """

    def __init__(
        self,
        pattern: EventPattern,
        steps: List[FlowStep],
        persistent: bool = False,
    ) -> None:
        self.pattern = pattern
        self.steps = steps
        self.persistent = persistent
        self.subscription_id: Optional[str] = None

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        """Register listener on the event bus to execute enclosed steps when triggered."""
        async def _on_event(event: Event) -> None:
            # Bind event data if needed
            ctx.context.set("last_event", event.data)
            if "text" in event.data:
                ctx.context.set("user_input", event.data["text"])
            for step in self.steps:
                res = await step.execute(ctx)
                if res.stop_flow or not res.success:
                    break
            if not self.persistent and self.subscription_id:
                ctx.event_bus.unsubscribe(self.subscription_id)

        self.subscription_id = ctx.event_bus.subscribe(self.pattern, _on_event)
        return StepResult(success=True, output=self.subscription_id)


class ActionStep(FlowStep):
    """Executes a custom or registered action asynchronously."""

    def __init__(
        self,
        action_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        variable: Optional[str] = None,
    ) -> None:
        self.action_name = action_name
        self.arguments = arguments or {}
        self.variable = variable

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        # Resolve argument templates
        resolved_args: Dict[str, Any] = {}
        for k, v in self.arguments.items():
            if isinstance(v, str):
                resolved_args[k] = ctx.resolve_template(v)
            else:
                resolved_args[k] = v

        # Emit action execution started event
        await ctx.event_bus.send(
            CustomActionExecution(action=self.action_name, arguments=resolved_args, status="started")
        )

        result_val: Any = None
        try:
            # 1. Check custom_actions in execution context
            if self.action_name in ctx.custom_actions:
                fn = ctx.custom_actions[self.action_name]
                if inspect.iscoroutinefunction(fn):
                    result_val = await fn(**resolved_args)
                else:
                    result_val = fn(**resolved_args)

            # 2. Check ActionRegistry if present
            elif ctx.action_registry is not None and hasattr(ctx.action_registry, "execute"):
                act_res = await ctx.action_registry.execute(self.action_name, resolved_args)
                result_val = act_res.return_value if hasattr(act_res, "return_value") else act_res

            # 3. Default fallback mock / pass-through
            else:
                result_val = f"Executed {self.action_name}({resolved_args})"

            # Store result in context variable if specified
            if self.variable:
                ctx.context.set(self.variable, result_val)

            # Emit action finished event
            await ctx.event_bus.send(
                CustomActionExecution(
                    action=self.action_name,
                    arguments=resolved_args,
                    status="finished",
                    result=result_val
                )
            )
            return StepResult(success=True, output=result_val)

        except Exception as e:
            await ctx.event_bus.send(
                CustomActionExecution(
                    action=self.action_name,
                    arguments=resolved_args,
                    status="failed",
                    result=str(e)
                )
            )
            return StepResult(success=False, error=str(e))


class ForkStep(FlowStep):
    """Concurrency primitive: forks a flow or steps into a concurrent asynchronous branch."""

    def __init__(
        self,
        branch_name: Optional[str] = None,
        flow_target: Optional[Union[str, Flow2]] = None,
        steps: Optional[List[FlowStep]] = None,
    ) -> None:
        self.branch_name = branch_name or f"branch_{uuid.uuid4().hex[:6]}"
        self.flow_target = flow_target
        self.steps = steps or []

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        async def _run_branch() -> Any:
            if self.flow_target:
                # If target is flow name or Flow2 instance
                flow_inst: Optional[Flow2] = None
                if isinstance(self.flow_target, Flow2):
                    flow_inst = self.flow_target
                else:
                    # Lookup flow from global registry or context
                    flow_inst = ctx.context.get(f"_flow_{self.flow_target}")

                if flow_inst:
                    return await flow_inst.execute(
                        context=ctx.context,
                        event_bus=ctx.event_bus,
                        action_registry=ctx.action_registry,
                        custom_actions=ctx.custom_actions,
                    )
            
            # Execute inline steps
            for step in self.steps:
                res = await step.execute(ctx)
                if res.stop_flow or not res.success:
                    return res
            return StepResult(success=True)

        task = asyncio.create_task(_run_branch(), name=f"fork_{self.branch_name}")
        ctx.forked_tasks[self.branch_name] = task
        return StepResult(success=True, output=self.branch_name)


class JoinStep(FlowStep):
    """Synchronization primitive: waits for concurrent forked branches to finish.

    Modes:
    - "all": Wait for all specified (or all active) branches to complete.
    - "any": Wait for the first branch to complete.
    """

    def __init__(
        self,
        branches: Optional[List[str]] = None,
        mode: str = "all",
    ) -> None:
        self.branches = branches
        self.mode = mode.lower()

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        target_tasks: List[asyncio.Task[Any]] = []
        if self.branches:
            for b in self.branches:
                if b in ctx.forked_tasks:
                    target_tasks.append(ctx.forked_tasks[b])
        else:
            target_tasks = list(ctx.forked_tasks.values())

        if not target_tasks:
            return StepResult(success=True, output=[])

        try:
            if self.mode == "any":
                done, pending = await asyncio.wait(target_tasks, return_when=asyncio.FIRST_COMPLETED)
                results = [t.result() for t in done if not t.cancelled() and t.exception() is None]
                return StepResult(success=True, output=results)
            else:  # "all"
                results = await asyncio.gather(*target_tasks, return_exceptions=True)
                return StepResult(success=True, output=results)
        except Exception as e:
            return StepResult(success=False, error=str(e))


class AwaitStep(FlowStep):
    """Awaits an event, flow execution, or action (`await <target>`)."""

    def __init__(
        self,
        target_step: FlowStep,
    ) -> None:
        self.target_step = target_step

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        return await self.target_step.execute(ctx)


class AssignStep(FlowStep):
    """Assigns an expression or literal to a scoped context variable (`$var = ...`)."""

    def __init__(
        self,
        variable: str,
        value: Any,
        scope: ScopeLevel = ScopeLevel.TURN,
    ) -> None:
        self.variable = variable
        self.value = value
        self.scope = scope

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        val = self.value
        if isinstance(val, str):
            val = ctx.resolve_template(val)
        ctx.context.set(self.variable, val, scope=self.scope)
        return StepResult(success=True, output=val)


class StopStep(FlowStep):
    """Stops flow execution."""

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        return StepResult(success=True, stop_flow=True)


class SubflowStep(FlowStep):
    """Executes a nested sub-dialogue or subflow asynchronously (`call flow <name>`, `subflow <name>`).

    Provides:
    - Scoped child context execution.
    - Argument mapping and dynamic variable binding.
    - Transactional rollback: if child subflow fails, parent context is protected.
    - Return value capturing into a designated variable (`call <flow_name>(...) as $result`).
    """

    def __init__(
        self,
        flow_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        variable: Optional[str] = None,
        enable_rollback_on_failure: bool = True,
    ) -> None:
        self.flow_name = flow_name
        self.arguments = arguments or {}
        self.variable = variable
        self.enable_rollback_on_failure = enable_rollback_on_failure

    async def execute(self, ctx: FlowExecutionContext) -> StepResult:
        # Resolve target flow from context or registry
        target_flow: Optional[Flow2] = ctx.context.get(f"_flow_{self.flow_name}")
        if target_flow is None and ctx.action_registry and hasattr(ctx.action_registry, "flows"):
            target_flow = ctx.action_registry.flows.get(self.flow_name)

        if target_flow is None:
            return StepResult(
                success=False,
                error=f"Subflow '{self.flow_name}' not found in registered flows.",
            )

        # Resolve argument templates
        resolved_args: Dict[str, Any] = {}
        for k, v in self.arguments.items():
            if isinstance(v, str):
                resolved_args[k] = ctx.resolve_template(v)
            else:
                resolved_args[k] = v

        # Create isolated child context for subflow execution
        child_ctx = ctx.context.create_child_context()
        for k, v in resolved_args.items():
            child_ctx.set(k, v)

        flow_res = await target_flow.execute(
            context=child_ctx,
            event_bus=ctx.event_bus,
            action_registry=ctx.action_registry,
            custom_actions=ctx.custom_actions,
            inputs=resolved_args,
            enable_rollback_on_failure=self.enable_rollback_on_failure,
        )

        if not flow_res.success:
            return StepResult(
                success=False,
                error=f"Nested subflow '{self.flow_name}' failed: {flow_res.error}",
            )

        if self.variable:
            ctx.context.set(self.variable, flow_res.output)

        return StepResult(
            success=True,
            output=flow_res.output,
            stop_flow=(flow_res.status == "stopped"),
        )


# ─── 6. FLOW2 DEFINITION & RUNTIME ───────────────────────────────────────────

class FlowResult(BaseModel):
    """Final outcome of executing a Flow2 instance."""
    flow_name: str
    flow_id: str
    success: bool
    status: str = "completed"  # "completed", "stopped", "failed"
    output: Any = None
    error: Optional[str] = None
    turn_id: int = 1


class Flow2:
    """Colang 2.0 Flow Definition and Async Execution Engine."""

    def __init__(
        self,
        name: str,
        parameters: Optional[List[str]] = None,
        steps: Optional[List[FlowStep]] = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.parameters = parameters or []
        self.steps = steps or []
        self.description = description

    def add_step(self, step: FlowStep) -> Flow2:
        self.steps.append(step)
        return self

    async def execute(
        self,
        context: ScopedContext,
        event_bus: EventBus,
        action_registry: Optional[Any] = None,
        custom_actions: Optional[Dict[str, Callable[..., Any]]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        enable_rollback_on_failure: bool = True,
    ) -> FlowResult:
        """Asynchronously execute this Flow2 instance with event publishing and scoped tracking."""
        flow_id = f"flow_{self.name}_{uuid.uuid4().hex[:6]}"
        ctx = FlowExecutionContext(
            flow_name=self.name,
            flow_id=flow_id,
            context=context,
            event_bus=event_bus,
            action_registry=action_registry,
            custom_actions=custom_actions,
        )

        # Bind inputs
        if inputs:
            for k, v in inputs.items():
                context.set(k, v)

        # Emit FlowStarted event
        await event_bus.send(FlowStarted(flow_name=self.name, flow_id=flow_id))

        # Take state snapshot in case rollback is needed
        snapshot = context.snapshot() if enable_rollback_on_failure else None
        last_output: Any = None

        try:
            for step in self.steps:
                res = await step.execute(ctx)
                if not res.success:
                    if enable_rollback_on_failure and snapshot:
                        context.rollback(snapshot)
                        await context.execute_compensations()
                    await event_bus.send(FlowFinished(flow_name=self.name, flow_id=flow_id, status="failed", error=res.error))
                    return FlowResult(
                        flow_name=self.name,
                        flow_id=flow_id,
                        success=False,
                        status="failed",
                        error=res.error,
                        turn_id=context.turn_id,
                    )
                last_output = res.output
                if res.stop_flow:
                    await event_bus.send(FlowFinished(flow_name=self.name, flow_id=flow_id, status="stopped"))
                    return FlowResult(
                        flow_name=self.name,
                        flow_id=flow_id,
                        success=True,
                        status="stopped",
                        output=last_output,
                        turn_id=context.turn_id,
                    )

            # Await any remaining forked tasks if not joined
            if ctx.forked_tasks:
                pending = [t for t in ctx.forked_tasks.values() if not t.done()]
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

            await event_bus.send(FlowFinished(flow_name=self.name, flow_id=flow_id, status="completed"))
            return FlowResult(
                flow_name=self.name,
                flow_id=flow_id,
                success=True,
                status="completed",
                output=last_output,
                turn_id=context.turn_id,
            )

        except Exception as e:
            if enable_rollback_on_failure and snapshot:
                context.rollback(snapshot)
                await context.execute_compensations()
            await event_bus.send(FlowFinished(flow_name=self.name, flow_id=flow_id, status="failed", error=str(e)))
            return FlowResult(
                flow_name=self.name,
                flow_id=flow_id,
                success=False,
                status="failed",
                error=str(e),
                turn_id=context.turn_id,
            )


# ─── 7. COLANG 2.0 FULL SYNTAX PARSER ─────────────────────────────────────────

class Colang2Parser:
    """Parser for Colang 2.0 event-driven language syntax.

    Supports:
    - Flow headers: `flow <name>(<params>):` or `flow <name>`
    - Event triggers: `when user said "<pattern>"` / `when <event>(...)`
    - Event matching: `match user said "<pattern>"` / `match <pattern>`
    - Event emission: `send bot say "<text>"` / `send <event>(...)`
    - Concurrency: `fork flow <name>`, `fork <name>`, `join`, `join all`, `join any`
    - Action execution: `action <name>(...)`, `await action <name>(...)`
    - Context variable assignments: `$var = <expr>`, `$scope.var = <expr>`
    - Control flow: `stop`
    """

    @classmethod
    def parse(cls, script: str) -> Dict[str, Flow2]:
        """Parse Colang 2.0 source code into a dictionary of Flow2 instances."""
        lines = script.splitlines()
        flows: Dict[str, Flow2] = {}
        current_flow: Optional[Flow2] = None
        current_when: Optional[WhenStep] = None
        when_indent = 0

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()

            # Ignore empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue

            # Calculate indentation level (spaces)
            indent = len(line) - len(line.lstrip(" "))

            # Flow declaration: `flow <name>[(<params>)]:`
            if stripped.startswith("flow ") or stripped.startswith("flow\t"):
                flow_decl = stripped[4:].strip().rstrip(":")
                params: List[str] = []
                if "(" in flow_decl and flow_decl.endswith(")"):
                    name_part, param_part = flow_decl[:-1].split("(", 1)
                    flow_name = name_part.strip()
                    params = [p.strip() for p in param_part.split(",") if p.strip()]
                else:
                    flow_name = flow_decl.strip()

                current_flow = Flow2(name=flow_name, parameters=params)
                flows[flow_name] = current_flow
                current_when = None
                continue

            if current_flow is None:
                # Default flow for top-level statements
                current_flow = Flow2(name="default_flow")
                flows["default_flow"] = current_flow

            # Check if we exited a `when` block
            if current_when is not None and indent <= when_indent:
                current_when = None

            # Parse step
            parsed_step = cls._parse_line_step(stripped)
            if parsed_step is None:
                continue

            # If it's a WhenStep, we track it for nested step accumulation
            if isinstance(parsed_step, WhenStep):
                current_when = parsed_step
                when_indent = indent
                current_flow.add_step(parsed_step)
            elif current_when is not None and indent > when_indent:
                current_when.steps.append(parsed_step)
            else:
                current_flow.add_step(parsed_step)

        return flows

    @classmethod
    def _parse_line_step(cls, line: str) -> Optional[FlowStep]:
        # 1. `when user said "..."` or `when bot said "..."` or `when <pattern>`
        if line.startswith("when "):
            stmt = line[5:].strip().rstrip(":")
            pat = cls._parse_pattern(stmt)
            return WhenStep(pattern=pat, steps=[], persistent=True)

        # 2. `match user said "..."` or `match <pattern>`
        if line.startswith("match "):
            stmt = line[6:].strip()
            var_name = None
            if " as " in stmt:
                stmt_part, var_part = stmt.split(" as ", 1)
                stmt = stmt_part.strip()
                var_name = var_part.strip().lstrip("$")
            pat = cls._parse_pattern(stmt)
            return MatchStep(pattern=pat, variable=var_name)

        # 3. `send bot say "..."` or `send user said "..."` or `send <EventName>(...)`
        if line.startswith("send "):
            stmt = line[5:].strip()
            return cls._parse_send(stmt)

        # 4. `fork flow <name>` or `fork <name>`
        if line.startswith("fork"):
            stmt = line[4:].strip()
            if stmt.startswith("flow "):
                target_flow = stmt[5:].strip()
                return ForkStep(branch_name=target_flow, flow_target=target_flow)
            elif stmt:
                return ForkStep(branch_name=stmt, flow_target=stmt)
            return ForkStep()

        # 5. `join` or `join all` or `join any` or `join <b1>, <b2>`
        if line.startswith("join"):
            stmt = line[4:].strip()
            if not stmt or stmt == "all":
                return JoinStep(mode="all")
            elif stmt == "any":
                return JoinStep(mode="any")
            else:
                branches = [b.strip() for b in stmt.split(",") if b.strip()]
                return JoinStep(branches=branches, mode="all")

        # 6. `await ...`
        if line.startswith("await "):
            inner = line[6:].strip()
            sub_step = cls._parse_line_step(inner)
            if sub_step:
                return AwaitStep(target_step=sub_step)

        # 7. `action <name>(...)` or `execute <name>(...)`
        if line.startswith("action ") or line.startswith("execute "):
            stmt = line.split(" ", 1)[1].strip()
            return cls._parse_action(stmt)

        # 8. Variable assignment: `$var = <expr>` or `set $var = <expr>`
        if "=" in line and (line.startswith("$") or line.startswith("set $") or line.startswith("turn.") or line.startswith("session.") or line.startswith("global.")):
            clean_line = line[4:].strip() if line.startswith("set ") else line
            lhs, rhs = clean_line.split("=", 1)
            raw_var = lhs.strip().lstrip("$")
            rhs_clean = rhs.strip().strip('"').strip("'")

            scope = ScopeLevel.TURN
            var_name = raw_var
            if raw_var.startswith("turn."):
                scope = ScopeLevel.TURN
                var_name = raw_var[5:]
            elif raw_var.startswith("session."):
                scope = ScopeLevel.SESSION
                var_name = raw_var[8:]
            elif raw_var.startswith("global."):
                scope = ScopeLevel.GLOBAL
                var_name = raw_var[7:]
            return AssignStep(variable=var_name, value=rhs_clean, scope=scope)

        # 9. `stop`
        if line == "stop":
            return StopStep()

        # 10. Nested Sub-Dialogue / Subflow: `call flow <name>`, `call <name>(...)`, `subflow <name>`, `subdialogue <name>`
        if line.startswith("call ") or line.startswith("subflow ") or line.startswith("subdialogue "):
            clean = line
            var_name = None
            if " as " in clean:
                clean, var_part = clean.split(" as ", 1)
                var_name = var_part.strip().lstrip("$")

            if clean.startswith("call flow "):
                flow_part = clean[10:].strip()
            elif clean.startswith("call "):
                flow_part = clean[5:].strip()
            elif clean.startswith("subflow "):
                flow_part = clean[8:].strip()
            else:
                flow_part = clean[12:].strip()

            args: Dict[str, Any] = {}
            if "(" in flow_part and flow_part.endswith(")"):
                f_name, arg_str = flow_part[:-1].split("(", 1)
                f_name = f_name.strip()
                args = cls._parse_kwargs(arg_str)
            else:
                f_name = flow_part.strip()

            return SubflowStep(flow_name=f_name, arguments=args, variable=var_name)

        return None

    @classmethod
    def _parse_pattern(cls, stmt: str) -> EventPattern:
        """Parse pattern string into EventPattern."""
        # e.g. `user said "hello*"`
        m_user = re.match(r'user\s+(?:said|utterance)\s+["\'](.*)["\']', stmt, re.IGNORECASE)
        if m_user:
            return EventPattern.user_said(m_user.group(1))

        # e.g. `bot said "..."` or `bot say "..."`
        m_bot = re.match(r'bot\s+(?:said|say|utterance)\s+["\'](.*)["\']', stmt, re.IGNORECASE)
        if m_bot:
            return EventPattern.bot_said(m_bot.group(1))

        # e.g. `UserUtterance(text="*fever*")`
        m_cls = re.match(r'([A-Za-z0-9_]+)\((.*)\)', stmt)
        if m_cls:
            ev_name = m_cls.group(1)
            args_str = m_cls.group(2)
            attrs = cls._parse_kwargs(args_str)
            return EventPattern(event_type=ev_name, attributes=attrs)

        # e.g. Simple event name
        return EventPattern(event_type=stmt)

    @classmethod
    def _parse_send(cls, stmt: str) -> SendStep:
        """Parse `send ...` expression into SendStep."""
        # `bot say "Hello"` or `bot said "Hello"`
        m_bot = re.match(r'bot\s+(?:say|said)\s+["\'](.*)["\']', stmt, re.IGNORECASE)
        if m_bot:
            return SendStep(event_name="BotUtterance", source="bot", data={"text": m_bot.group(1)})

        # `user said "Hello"`
        m_user = re.match(r'user\s+said\s+["\'](.*)["\']', stmt, re.IGNORECASE)
        if m_user:
            return SendStep(event_name="UserUtterance", source="user", data={"text": m_user.group(1)})

        # `BotUtterance(text="...")` or `<EventName>(...)`
        m_cls = re.match(r'([A-Za-z0-9_]+)\((.*)\)', stmt)
        if m_cls:
            ev_name = m_cls.group(1)
            args = cls._parse_kwargs(m_cls.group(2))
            source = "bot" if "bot" in ev_name.lower() else "system"
            return SendStep(event_name=ev_name, source=source, data=args)

        return SendStep(event_name=stmt, source="system")

    @classmethod
    def _parse_action(cls, stmt: str) -> ActionStep:
        """Parse action invocation."""
        var_name = None
        if "=" in stmt:
            lhs, stmt = stmt.split("=", 1)
            var_name = lhs.strip().lstrip("$")

        m_call = re.match(r'([A-Za-z0-9_]+)\((.*)\)', stmt.strip())
        if m_call:
            act_name = m_call.group(1)
            args = cls._parse_kwargs(m_call.group(2))
            return ActionStep(action_name=act_name, arguments=args, variable=var_name)
        else:
            return ActionStep(action_name=stmt.strip(), variable=var_name)

    @staticmethod
    def _parse_kwargs(args_str: str) -> Dict[str, Any]:
        """Simple kwargs parser e.g. `text="hi", count=2`."""
        args: Dict[str, Any] = {}
        if not args_str.strip():
            return args

        # Match key = "value" or key = value
        pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,]+))')
        for m in pattern.finditer(args_str):
            k = m.group(1)
            v = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4).strip())
            args[k] = v
        return args


# ─── 8. HIGH-LEVEL COLANG 2.0 ENGINE / RUNTIME ───────────────────────────────

class Colang2Engine:
    """Central Colang 2.0 Engine coordinating flows, event bus, scoped context, and actions."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        context: Optional[ScopedContext] = None,
        action_registry: Optional[Any] = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.context = context or ScopedContext(event_bus=self.event_bus)
        self.action_registry = action_registry
        self.custom_actions: Dict[str, Callable[..., Any]] = {}
        self.flows: Dict[str, Flow2] = {}

    def register_action(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a custom Python action callable."""
        self.custom_actions[name] = fn

    def register_flow(self, flow: Flow2) -> None:
        """Register a Flow2 definition."""
        self.flows[flow.name] = flow
        # Store in context for dynamic lookup
        self.context.set(f"_flow_{flow.name}", flow, scope=ScopeLevel.GLOBAL)

    def load_flows_from_text(self, script: str) -> List[Flow2]:
        """Parse Colang 2.0 source code and register all declared flows."""
        parsed = Colang2Parser.parse(script)
        for flow in parsed.values():
            self.register_flow(flow)
        return list(parsed.values())

    async def initialize_listeners(self) -> None:
        """Initialize all top-level `when` listeners defined in registered flows."""
        for flow in self.flows.values():
            ctx = FlowExecutionContext(
                flow_name=flow.name,
                flow_id=f"flow_{flow.name}_init",
                context=self.context,
                event_bus=self.event_bus,
                action_registry=self.action_registry,
                custom_actions=self.custom_actions,
            )
            for step in flow.steps:
                if isinstance(step, WhenStep):
                    await step.execute(ctx)

    async def send_user_utterance(self, text: str) -> List[str]:
        """High-level interaction helper.

        Dispatches UserUtterance onto the event bus and collects resulting BotUtterance texts.
        """
        bot_replies: List[str] = []

        async def _bot_listener(event: Event) -> None:
            if "text" in event.data:
                bot_replies.append(event.data["text"])

        sub_id = self.event_bus.subscribe(
            EventPattern(event_type=EventType.BOT_UTTERANCE),
            _bot_listener,
        )

        try:
            # Emit user utterance
            user_event = UserUtterance(text=text)
            await self.event_bus.send(user_event)
            # Short yield to allow async tasks and bus events to resolve
            await asyncio.sleep(0.01)
        finally:
            self.event_bus.unsubscribe(sub_id)

        return bot_replies

    async def execute_flow(self, flow_name: str, inputs: Optional[Dict[str, Any]] = None) -> FlowResult:
        """Execute a specific flow by name."""
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' not found in registry.")
        flow = self.flows[flow_name]
        return await flow.execute(
            context=self.context,
            event_bus=self.event_bus,
            action_registry=self.action_registry,
            custom_actions=self.custom_actions,
            inputs=inputs,
        )

    def new_turn(self) -> int:
        """Advance dialogue turn."""
        return self.context.new_turn()

"""VaxGuard NeMo Guardrails: Custom Action Runtime & Tool Calling Interface.

Provides 100% pure Python action execution runtime, action registry,
asynchronous dispatch, parameter schema extraction, lifecycle middleware,
and OpenAPI/JSON-Schema tool definitions. Zero external heavy binaries.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    get_type_hints,
)

from pydantic import BaseModel, Field


# ─── Action Enums & Result Contracts ─────────────────────────────────────────

class ActionStatus(str, Enum):
    """Execution status of an invoked action."""
    SUCCESS = "success"
    FAILURE = "failure"
    EXCEPTION = "exception"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class ActionType(str, Enum):
    """Categorization of registered actions."""
    BUILTIN = "builtin"
    CUSTOM = "custom"
    GUARDRAIL = "guardrail"
    TOOL = "tool"


class ActionParameter(BaseModel):
    """Schema descriptor for an action parameter."""
    name: str
    type_name: str = "string"
    description: str = ""
    default: Optional[Any] = None
    required: bool = True


class ToolDefinition(BaseModel):
    """OpenAPI / JSON-Schema tool specification for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Format as standard OpenAI/Claude tool calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ActionResult(BaseModel):
    """Audit telemetry and payload returned from an executed action."""
    action_name: str
    status: ActionStatus
    return_value: Any = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    events_emitted: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ActionStatus.SUCCESS


class ActionContext(BaseModel):
    """Execution context injected into actions requiring session metadata."""
    session_id: str = "default_session"
    user_id: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    credentials: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransformTarget(str, Enum):
    USER_MESSAGE = "user_message"
    BOT_MESSAGE = "bot_message"
    RELEVANT_CHUNKS = "relevant_chunks"


class RailOutcome(BaseModel):
    """Normalized response contract for NeMo guardrail actions."""
    action_name: str = ""
    status: ActionStatus = ActionStatus.SUCCESS
    reason: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    transformed_targets: List[Tuple[TransformTarget, str]] = Field(default_factory=list)

    @classmethod
    def allow(cls, action_name: str, data: Optional[Dict[str, Any]] = None) -> RailOutcome:
        return cls(action_name=action_name, status=ActionStatus.SUCCESS, data=data or {})

    @classmethod
    def block(cls, action_name: str, reason: str, data: Optional[Dict[str, Any]] = None) -> RailOutcome:
        return cls(action_name=action_name, status=ActionStatus.BLOCKED, reason=reason, data=data or {})

    @classmethod
    def transform(
        cls,
        targets: List[Tuple[TransformTarget, str]],
        action_name: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> RailOutcome:
        return cls(
            action_name=action_name,
            status=ActionStatus.SUCCESS,
            transformed_targets=targets,
            data=data or {},
        )

    @property
    def is_allowed(self) -> bool:
        return self.status != ActionStatus.BLOCKED


# ─── Action Descriptor ───────────────────────────────────────────────────────

class ActionDescriptor:
    """Introspects and wraps a Python callable for dynamic execution."""

    def __init__(
        self,
        name: str,
        handler: Callable[..., Any],
        description: Optional[str] = None,
        action_type: ActionType = ActionType.CUSTOM,
        parameters: Optional[List[ActionParameter]] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.name = name
        self.handler = handler
        self.action_type = action_type
        self.timeout_sec = timeout_sec
        self.is_coroutine = asyncio.iscoroutinefunction(handler)

        # Inspect docstring if description not provided
        doc = inspect.getdoc(handler) or ""
        self.description = description or (doc.split("\n\n")[0] if doc else f"Action {name}")

        # Introspect parameters if not explicitly provided
        self.parameters = parameters if parameters is not None else self._introspect_parameters()

    def _introspect_parameters(self) -> List[ActionParameter]:
        """Extract typed parameters from handler callable signature."""
        params: List[ActionParameter] = []
        try:
            sig = inspect.signature(self.handler)
            type_hints = {}
            try:
                type_hints = get_type_hints(self.handler)
            except Exception:
                pass

            for p_name, param in sig.parameters.items():
                if p_name in ("self", "cls", "context"):
                    continue

                p_type = "string"
                hint = type_hints.get(p_name, param.annotation)
                if hint is int:
                    p_type = "integer"
                elif hint is float:
                    p_type = "number"
                elif hint is bool:
                    p_type = "boolean"
                elif hint in (list, List) or getattr(hint, "__origin__", None) in (list, List):
                    p_type = "array"
                elif hint in (dict, Dict) or getattr(hint, "__origin__", None) in (dict, Dict):
                    p_type = "object"

                has_default = param.default is not inspect.Parameter.empty
                params.append(
                    ActionParameter(
                        name=p_name,
                        type_name=p_type,
                        description="",
                        default=param.default if has_default else None,
                        required=not has_default,
                    )
                )
        except Exception:
            pass
        return params

    def to_tool_definition(self) -> ToolDefinition:
        """Generate OpenAPI tool specification from parameter descriptors."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in self.parameters:
            properties[p.name] = {
                "type": p.type_name,
                "description": p.description or f"Parameter {p.name}",
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)

        parameters_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters_schema,
        )


# ─── Built-in Default Guardrail Actions ──────────────────────────────────────

def _builtin_check_facts(claim: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Default built-in fact verification action."""
    if not context:
        return {"grounded": False, "confidence": 0.0, "reason": "No context provided"}
    claim_words = set(re.findall(r"\w+", claim.lower()))
    ctx_words = set(re.findall(r"\w+", context.lower()))
    if not claim_words:
        return {"grounded": True, "confidence": 1.0, "overlap": 1.0}
    overlap = len(claim_words & ctx_words) / len(claim_words)
    return {
        "grounded": overlap >= 0.40,
        "confidence": round(overlap, 4),
        "overlap": round(overlap, 4),
    }


def _builtin_retrieve_context(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Default mock retrieval action for RAG pipelines."""
    return [
        {
            "id": f"doc_{i}",
            "text": f"Retrieved reference context passage {i} for query: {query}",
            "relevance": round(1.0 - (i * 0.15), 2),
        }
        for i in range(1, top_k + 1)
    ]


def _builtin_mask_sensitive_data(text: str) -> Dict[str, Any]:
    """Default PII sanitization action."""
    masked = text
    masked = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", masked)
    masked = re.sub(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[CREDIT_CARD_REDACTED]", masked)
    masked = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", masked)
    return {
        "original": text,
        "sanitized": masked,
        "modified": masked != text,
    }


def _builtin_execute_security_audit(event_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Security telemetry logger action."""
    return {
        "event_type": event_type,
        "status": "recorded",
        "timestamp": time.time(),
        "audit_id": f"audit_{int(time.time() * 1000)}",
        "details": details,
    }


def _builtin_escalate_to_human(reason: str, priority: str = "high") -> Dict[str, Any]:
    """Human-in-the-loop escalation trigger."""
    return {
        "escalated": True,
        "reason": reason,
        "priority": priority,
        "ticket_id": f"ESC-{int(time.time())}",
    }


# ─── Action Registry & Execution Runtime ─────────────────────────────────────

class ActionRegistry:
    """Central repository and dispatcher for Colang actions and LLM tools."""

    def __init__(self) -> None:
        self._actions: Dict[str, ActionDescriptor] = {}
        self._before_hooks: List[Callable[[str, Dict[str, Any], ActionContext], None]] = []
        self._after_hooks: List[Callable[[ActionResult, ActionContext], None]] = []
        self._error_hooks: List[Callable[[str, Exception, ActionContext], None]] = []

        # Register standard built-in actions
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register default system actions."""
        self.register(
            "check_facts",
            _builtin_check_facts,
            description="Verify if a claim is grounded in retrieved context",
            action_type=ActionType.BUILTIN,
        )
        self.register(
            "retrieve_context",
            _builtin_retrieve_context,
            description="Retrieve contextual passages for a given query",
            action_type=ActionType.BUILTIN,
        )
        self.register(
            "mask_sensitive_data",
            _builtin_mask_sensitive_data,
            description="Sanitize PII entities from text",
            action_type=ActionType.BUILTIN,
        )
        self.register(
            "execute_security_audit",
            _builtin_execute_security_audit,
            description="Emit security audit log entry",
            action_type=ActionType.BUILTIN,
        )
        self.register(
            "escalate_to_human",
            _builtin_escalate_to_human,
            description="Escalate safety violation to human moderator",
            action_type=ActionType.BUILTIN,
        )
        try:
            from vaxguard.nemo.library import (
                detect_sensitive_data,
                self_check_input,
                self_check_output,
                self_check_facts,
                topic_safety_check_input,
                detect_regex_pattern,
            )
            self.register("detect_sensitive_data", detect_sensitive_data, action_type=ActionType.GUARDRAIL)
            self.register("self_check_input", self_check_input, action_type=ActionType.GUARDRAIL)
            self.register("self_check_output", self_check_output, action_type=ActionType.GUARDRAIL)
            self.register("self_check_facts", self_check_facts, action_type=ActionType.GUARDRAIL)
            self.register("topic_safety_check_input", topic_safety_check_input, action_type=ActionType.GUARDRAIL)
            self.register("detect_regex_pattern", detect_regex_pattern, action_type=ActionType.GUARDRAIL)
        except Exception:
            pass

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: Optional[str] = None,
        action_type: ActionType = ActionType.CUSTOM,
        parameters: Optional[List[ActionParameter]] = None,
        timeout_sec: float = 10.0,
    ) -> ActionDescriptor:
        """Register a callable action in the registry."""
        desc = ActionDescriptor(
            name=name,
            handler=handler,
            description=description,
            action_type=action_type,
            parameters=parameters,
            timeout_sec=timeout_sec,
        )
        self._actions[name] = desc
        return desc

    def action(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        action_type: ActionType = ActionType.CUSTOM,
        timeout_sec: float = 10.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering functions as actions."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            act_name = name or func.__name__
            self.register(
                name=act_name,
                handler=func,
                description=description,
                action_type=action_type,
                timeout_sec=timeout_sec,
            )
            return func
        return decorator

    def unregister(self, name: str) -> bool:
        """Remove an action by name."""
        return self._actions.pop(name, None) is not None

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Get action handler by name."""
        desc = self._actions.get(name)
        return desc.handler if desc else None

    def has_action(self, name: str) -> bool:
        """Check if action exists."""
        return name in self._actions

    def get_action(self, name: str) -> Optional[ActionDescriptor]:
        """Fetch descriptor for action."""
        return self._actions.get(name)

    def list_actions(self) -> List[str]:
        """List all registered action names."""
        return sorted(list(self._actions.keys()))

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Generate tool definitions for all registered actions."""
        return [desc.to_tool_definition() for desc in self._actions.values()]

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Export all actions as OpenAI compatible function tools."""
        return [tool.to_openai_schema() for tool in self.get_tool_definitions()]

    # ─── Middleware Lifecycle Hooks ──────────────────────────────────────────

    def add_before_hook(
        self, hook: Callable[[str, Dict[str, Any], ActionContext], None]
    ) -> None:
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable[[ActionResult, ActionContext], None]) -> None:
        self._after_hooks.append(hook)

    def add_error_hook(self, hook: Callable[[str, Exception, ActionContext], None]) -> None:
        self._error_hooks.append(hook)

    # ─── Execution Logic ─────────────────────────────────────────────────────

    def _resolve_parameters(
        self,
        desc: ActionDescriptor,
        kwargs: Dict[str, Any],
        context: ActionContext,
    ) -> Dict[str, Any]:
        """Resolve argument bindings including variables ($var_name) and context."""
        resolved: Dict[str, Any] = {}
        sig = inspect.signature(desc.handler)

        for p_name, param in sig.parameters.items():
            if (p_name == "action_context" or (p_name == "context" and param.annotation is ActionContext)) and p_name not in kwargs:
                resolved[p_name] = context
                continue

            if p_name in kwargs:
                val = kwargs[p_name]
                # If value is a variable reference (e.g. "$user_query"), resolve from context
                if isinstance(val, str) and val.startswith("$"):
                    var_key = val[1:]
                    resolved[p_name] = context.variables.get(var_key, val)
                else:
                    resolved[p_name] = val
            elif param.default is not inspect.Parameter.empty:
                resolved[p_name] = param.default
            elif p_name in context.variables:
                resolved[p_name] = context.variables[p_name]

        return resolved

    def execute(
        self,
        name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[ActionContext] = None,
        timeout_sec: Optional[float] = None,
    ) -> ActionResult:
        """Synchronously execute a registered action."""
        ctx = context or ActionContext()
        args = dict(kwargs or {})

        if name not in self._actions:
            return ActionResult(
                action_name=name,
                status=ActionStatus.FAILURE,
                error_message=f"Action '{name}' not found in registry.",
            )

        desc = self._actions[name]
        timeout = timeout_sec or desc.timeout_sec
        start_t = time.perf_counter()

        # Run before hooks
        for b_hook in self._before_hooks:
            try:
                b_hook(name, args, ctx)
            except Exception:
                pass

        try:
            resolved_args = self._resolve_parameters(desc, args, ctx)
            if desc.is_coroutine:
                # If async coroutine invoked from sync context, run via event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # Running in existing event loop: run as task or thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            asyncio.run,
                            asyncio.wait_for(desc.handler(**resolved_args), timeout=timeout),
                        )
                        result_val = future.result(timeout=timeout)
                else:
                    result_val = loop.run_until_complete(
                        asyncio.wait_for(desc.handler(**resolved_args), timeout=timeout)
                    )
            else:
                result_val = desc.handler(**resolved_args)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            act_res = ActionResult(
                action_name=name,
                status=ActionStatus.SUCCESS,
                return_value=result_val,
                latency_ms=round(elapsed_ms, 3),
            )

            for a_hook in self._after_hooks:
                try:
                    a_hook(act_res, ctx)
                except Exception:
                    pass

            return act_res

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            act_res = ActionResult(
                action_name=name,
                status=ActionStatus.TIMEOUT,
                error_message=f"Action '{name}' timed out after {timeout}s",
                latency_ms=round(elapsed_ms, 3),
            )
            return act_res

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            for e_hook in self._error_hooks:
                try:
                    e_hook(name, exc, ctx)
                except Exception:
                    pass
            return ActionResult(
                action_name=name,
                status=ActionStatus.EXCEPTION,
                error_message=str(exc),
                latency_ms=round(elapsed_ms, 3),
            )

    async def execute_async(
        self,
        name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[ActionContext] = None,
        timeout_sec: Optional[float] = None,
    ) -> ActionResult:
        """Asynchronously execute a registered action."""
        ctx = context or ActionContext()
        args = dict(kwargs or {})

        if name not in self._actions:
            return ActionResult(
                action_name=name,
                status=ActionStatus.FAILURE,
                error_message=f"Action '{name}' not found in registry.",
            )

        desc = self._actions[name]
        timeout = timeout_sec or desc.timeout_sec
        start_t = time.perf_counter()

        for b_hook in self._before_hooks:
            try:
                b_hook(name, args, ctx)
            except Exception:
                pass

        try:
            resolved_args = self._resolve_parameters(desc, args, ctx)
            if desc.is_coroutine:
                result_val = await asyncio.wait_for(desc.handler(**resolved_args), timeout=timeout)
            else:
                loop = asyncio.get_running_loop()
                result_val = await loop.run_in_executor(None, lambda: desc.handler(**resolved_args))

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            act_res = ActionResult(
                action_name=name,
                status=ActionStatus.SUCCESS,
                return_value=result_val,
                latency_ms=round(elapsed_ms, 3),
            )

            for a_hook in self._after_hooks:
                try:
                    a_hook(act_res, ctx)
                except Exception:
                    pass

            return act_res

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return ActionResult(
                action_name=name,
                status=ActionStatus.TIMEOUT,
                error_message=f"Action '{name}' timed out after {timeout}s",
                latency_ms=round(elapsed_ms, 3),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            for e_hook in self._error_hooks:
                try:
                    e_hook(name, exc, ctx)
                except Exception:
                    pass
            return ActionResult(
                action_name=name,
                status=ActionStatus.EXCEPTION,
                error_message=str(exc),
                latency_ms=round(elapsed_ms, 3),
            )

"""
VaxGuard Enterprise Ecosystem Adapter: Microsoft Semantic Kernel Integration.

Provides production-grade, zero-external-heavy-dependency integrations for Microsoft Semantic Kernel:
1. `VaxGuardFilter`: Semantic Kernel compatible Prompt and Function Invocation filter.
   - Intercepts `on_prompt_render`: Scans rendered prompts for jailbreaks, prompt injections, and policy breaches.
   - Intercepts `on_function_invocation`: Scans tool arguments for shell injection, SQL injection, SSRF,
     path traversal, and validates function outputs against sensitive data leakage.
   - Intercepts `on_auto_function_invocation`: Protects autonomous agent tool loops.
   - Synchronous and asynchronous execution support.

Pure Python 3.10+, deterministic execution SLA < 1.0ms, zero mandatory heavy binaries.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from vaxguard.core.logger import get_logger
from vaxguard.integrations.langchain_guard import VaxGuardSecurityViolation
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    TriHybridMatcher,
)
from vaxguard.nemo.streaming_interceptor import DEFAULT_OUTPUT_LEAK_PATTERNS

logger = get_logger("SemanticKernelGuard")

# Optional Semantic Kernel base classes (graceful fallback when semantic_kernel is not installed)
try:
    from semantic_kernel.filters.filter_types import FilterTypes  # type: ignore[import-untyped]
except Exception:
    class FilterTypes:  # type: ignore[no-redef]
        """Fallback FilterTypes enum when semantic_kernel is not installed."""
        FUNCTION_INVOCATION = "function_invocation"
        PROMPT_RENDER = "prompt_render"
        AUTO_FUNCTION_INVOCATION = "auto_function_invocation"


# ─── Tool Parameter Attack Signatures ─────────────────────────────────────────

SHELL_INJECTION_RE = re.compile(
    r"""(?ix)
    (?:
        [;&|`$><]\s*(?:cat|ls|rm|chmod|chown|wget|curl|nc|bash|sh|cmd|powershell|whoami|id|uname)\b
        |\b(?:bash|sh|cmd(?:\.exe)?|powershell(?:\.exe)?)\s+(?:-c|-e|/c)\b
        |\$\([^\)]+\)
        |`[^`]+`
    )
    """
)

SQL_INJECTION_RE = re.compile(
    r"""(?ix)
    (?:
        (?:--|/\*|\#)
        |\bUNION\s+(?:ALL\s+)?SELECT\b
        |\b(?:OR|AND)\s+['"]?\d+['"]?\s*=\s*['"]?\d+['"]?
        |\b(?:DROP\s+TABLE|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)\b
    )
    """
)

PATH_TRAVERSAL_RE = re.compile(
    r"""(?ix)
    (?:
        (?:\.\.[/\\])+
        |/(?:etc|proc|sys|var/log)/
        |[a-z]:\\(?:windows|boot|system32)
    )
    """
)

SSRF_RE = re.compile(
    r"""(?ix)
    (?:
        (?:https?|ftp|gopher|file)://(?:169\.254\.169\.254|127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|metadata\.google\.internal)
        |169\.254\.169\.254
        |10\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
        |192\.168\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
    )
    """
)


# ─── Semantic Kernel Defense Filter ───────────────────────────────────────────

class VaxGuardFilter:
    """
    Microsoft Semantic Kernel compatible security filter.
    Enforces bidirectional enterprise guardrails across:
      1. Prompt Rendering (`on_prompt_render`):
         Evaluates rendered prompts for prompt injections, jailbreaks, and system prompt leaks.
      2. Function Invocation (`on_function_invocation`):
         Validates function authorization, inspects arguments for command injection, SQLi, SSRF,
         and scans returned results for sensitive credential disclosures or data leakage.
      3. Auto Function Invocation (`on_auto_function_invocation`):
         Guards autonomous tool selection cycles.
    """

    def __init__(
        self,
        matcher: Optional[TriHybridMatcher] = None,
        block_prompts: bool = True,
        block_function_args: bool = True,
        block_function_results: bool = True,
        allowed_functions: Optional[Iterable[str]] = None,
        blocked_functions: Optional[Iterable[str]] = None,
        raise_exception: bool = True,
        redact_on_violation: bool = False,
        on_violation: Optional[Callable[[VaxGuardSecurityViolation], Any]] = None,
    ) -> None:
        self.matcher = matcher or TriHybridMatcher()
        self.block_prompts = block_prompts
        self.block_function_args = block_function_args
        self.block_function_results = block_function_results
        self.allowed_functions: Optional[Set[str]] = (
            set(allowed_functions) if allowed_functions is not None else None
        )
        self.blocked_functions: Set[str] = (
            set(blocked_functions) if blocked_functions is not None else set()
        )
        self.raise_exception = raise_exception
        self.redact_on_violation = redact_on_violation
        self.on_violation = on_violation

        # Telemetry & State
        self.violations: List[VaxGuardSecurityViolation] = []
        self.prompts_scanned: int = 0
        self.function_calls_scanned: int = 0
        self.results_scanned: int = 0
        self._circuit_open: bool = False

        # Compiled leak regexes for result checking
        self._leak_patterns: List[Tuple[re.Pattern, str, str]] = []
        for pat, rule_id, desc in DEFAULT_OUTPUT_LEAK_PATTERNS:
            try:
                self._leak_patterns.append((re.compile(pat), rule_id, desc))
            except re.error:
                pass

    @property
    def is_tripped(self) -> bool:
        """Indicates whether any security violation has occurred."""
        return self._circuit_open

    def reset(self) -> None:
        """Reset filter state and violation log."""
        self._circuit_open = False
        self.violations.clear()
        self.prompts_scanned = 0
        self.function_calls_scanned = 0
        self.results_scanned = 0

    def _handle_violation(self, violation: VaxGuardSecurityViolation) -> None:
        """Record violation and optionally trigger callback or raise."""
        self.violations.append(violation)
        self._circuit_open = True
        logger.warning(str(violation))

        if self.on_violation:
            try:
                self.on_violation(violation)
            except Exception as ex:
                logger.error(f"Error in on_violation callback: {ex}")

        if self.raise_exception:
            raise violation

    # ─── 1. Prompt Render Filter ──────────────────────────────────────────────

    async def on_prompt_render(self, context: Any, next: Callable[[Any], Any]) -> None:
        """
        Semantic Kernel Prompt Render Filter.
        Intercepts prompt rendering. Evaluates the rendered prompt for prompt injection / jailbreaks.
        """
        # Execute prompt template rendering
        if inspect.iscoroutinefunction(next):
            await next(context)
        else:
            res = next(context)
            if inspect.iscoroutine(res):
                await res

        if not self.block_prompts:
            return

        self.prompts_scanned += 1
        rendered_prompt = getattr(context, "rendered_prompt", None)
        if rendered_prompt is None and hasattr(context, "prompt"):
            rendered_prompt = getattr(context, "prompt")

        if not isinstance(rendered_prompt, str) or not rendered_prompt.strip():
            return

        # Scan rendered prompt
        match_res = self.matcher.match(rendered_prompt)
        if match_res.matched and match_res.action == GuardrailAction.BLOCK:
            violation = VaxGuardSecurityViolation(
                message=f"Semantic Kernel prompt blocked: {match_res.category} ({match_res.rule_id})",
                category=match_res.category,
                rule_id=match_res.rule_id,
                confidence=match_res.confidence,
                payload_sample=rendered_prompt[:120],
                metadata={"filter_type": "prompt_render"},
            )
            if self.redact_on_violation:
                # Sanitize prompt in place
                redacted = "[VAXGUARD BLOCKED: Malicious prompt rejected]"
                if hasattr(context, "rendered_prompt"):
                    setattr(context, "rendered_prompt", redacted)
            self._handle_violation(violation)

    # ─── 2. Function Invocation Filter ────────────────────────────────────────

    async def on_function_invocation(self, context: Any, next: Callable[[Any], Any]) -> None:
        """
        Semantic Kernel Function Invocation Filter.
        Intercepts tool execution:
        - Validates function authorization (allowlist/denylist)
        - Scans function arguments for injection attacks (shell, SQL, path, SSRF)
        - Invokes function via `next(context)`
        - Validates returned function output against sensitive credential leakage.
        """
        self.function_calls_scanned += 1

        # Extract function name
        function_name = self._extract_function_name(context)

        # 1. Authorization check
        if self.blocked_functions and function_name in self.blocked_functions:
            violation = VaxGuardSecurityViolation(
                message=f"Function '{function_name}' is blocked by security policy",
                category="function_authorization",
                rule_id="blocked_function_invocation",
                confidence=1.0,
                payload_sample=function_name,
                metadata={"function_name": function_name},
            )
            self._handle_violation(violation)

        if self.allowed_functions is not None and function_name not in self.allowed_functions:
            violation = VaxGuardSecurityViolation(
                message=f"Function '{function_name}' is not in allowed functions list",
                category="function_authorization",
                rule_id="unauthorized_function_invocation",
                confidence=1.0,
                payload_sample=function_name,
                metadata={"function_name": function_name},
            )
            self._handle_violation(violation)

        # 2. Inspect Function Arguments
        if self.block_function_args:
            arguments = self._extract_arguments(context)
            self._inspect_arguments(arguments, function_name)

        # 3. Execute function via next()
        if inspect.iscoroutinefunction(next):
            await next(context)
        else:
            res = next(context)
            if inspect.iscoroutine(res):
                await res

        # 4. Inspect Function Result
        if self.block_function_results:
            self._inspect_result(context, function_name)

    async def on_auto_function_invocation(self, context: Any, next: Callable[[Any], Any]) -> None:
        """Semantic Kernel Auto Function Invocation Filter for agentic tool loops."""
        await self.on_function_invocation(context, next)

    # ─── 3. Helpers: Argument & Result Inspection ─────────────────────────────

    def _extract_function_name(self, context: Any) -> str:
        """Extract function name string from Semantic Kernel context."""
        fn = getattr(context, "function", None)
        if fn is not None:
            if hasattr(fn, "name"):
                return str(fn.name)
            if hasattr(fn, "__name__"):
                return str(fn.__name__)
            return str(fn)
        if hasattr(context, "function_name"):
            return str(context.function_name)
        return "unknown_function"

    def _extract_arguments(self, context: Any) -> Dict[str, Any]:
        """Extract arguments dictionary from Semantic Kernel context."""
        args = getattr(context, "arguments", None)
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if hasattr(args, "items"):
            return dict(args.items())
        return {}

    def _inspect_arguments(self, arguments: Dict[str, Any], function_name: str) -> None:
        """Deep scan of function arguments for injection payloads."""
        for key, val in arguments.items():
            val_str = str(val) if not isinstance(val, (dict, list)) else json.dumps(val)

            # Check Shell Injection
            if SHELL_INJECTION_RE.search(val_str):
                violation = VaxGuardSecurityViolation(
                    message=f"Shell injection detected in argument '{key}' of '{function_name}'",
                    category="shell_injection",
                    rule_id="tool_arg_shell_injection",
                    confidence=0.98,
                    payload_sample=val_str[:120],
                    metadata={"function": function_name, "arg_name": key},
                )
                self._handle_violation(violation)

            # Check SQL Injection
            if SQL_INJECTION_RE.search(val_str):
                violation = VaxGuardSecurityViolation(
                    message=f"SQL injection detected in argument '{key}' of '{function_name}'",
                    category="sql_injection",
                    rule_id="tool_arg_sql_injection",
                    confidence=0.95,
                    payload_sample=val_str[:120],
                    metadata={"function": function_name, "arg_name": key},
                )
                self._handle_violation(violation)

            # Check Path Traversal
            if PATH_TRAVERSAL_RE.search(val_str):
                violation = VaxGuardSecurityViolation(
                    message=f"Path traversal detected in argument '{key}' of '{function_name}'",
                    category="path_traversal",
                    rule_id="tool_arg_path_traversal",
                    confidence=0.95,
                    payload_sample=val_str[:120],
                    metadata={"function": function_name, "arg_name": key},
                )
                self._handle_violation(violation)

            # Check SSRF
            if SSRF_RE.search(val_str):
                violation = VaxGuardSecurityViolation(
                    message=f"SSRF target detected in argument '{key}' of '{function_name}'",
                    category="ssrf_attack",
                    rule_id="tool_arg_ssrf",
                    confidence=0.95,
                    payload_sample=val_str[:120],
                    metadata={"function": function_name, "arg_name": key},
                )
                self._handle_violation(violation)

            # Check TriHybrid Matcher
            if len(val_str) > 10:
                match_res = self.matcher.match(val_str)
                if match_res.matched and match_res.action == GuardrailAction.BLOCK:
                    violation = VaxGuardSecurityViolation(
                        message=f"Malicious payload in argument '{key}': {match_res.category} ({match_res.rule_id})",
                        category=match_res.category,
                        rule_id=match_res.rule_id,
                        confidence=match_res.confidence,
                        payload_sample=val_str[:120],
                        metadata={"function": function_name, "arg_name": key},
                    )
                    self._handle_violation(violation)

    def _inspect_result(self, context: Any, function_name: str) -> None:
        """Scan function execution result for sensitive credential disclosures or prompt leakage."""
        self.results_scanned += 1
        res = getattr(context, "result", None)
        if res is None:
            return

        # Handle FunctionResult object or raw string/dict
        res_text = ""
        if hasattr(res, "value"):
            val = getattr(res, "value")
            res_text = str(val) if not isinstance(val, (dict, list)) else json.dumps(val)
        elif isinstance(res, str):
            res_text = res
        elif isinstance(res, (dict, list)):
            res_text = json.dumps(res)
        else:
            res_text = str(res)

        if not res_text.strip():
            return

        # Check for credential leakage
        for pat, rule_id, desc in self._leak_patterns:
            m = pat.search(res_text)
            if m:
                violation = VaxGuardSecurityViolation(
                    message=f"Output leakage detected in function '{function_name}' result: {desc}",
                    category="credential_leakage",
                    rule_id=rule_id,
                    confidence=1.0,
                    payload_sample=m.group(0)[:60],
                    metadata={"function": function_name, "rule_id": rule_id},
                )
                if self.redact_on_violation:
                    redacted = pat.sub("[REDACTED_SECRET]", res_text)
                    if hasattr(res, "value"):
                        setattr(res, "value", redacted)
                    elif hasattr(context, "result"):
                        setattr(context, "result", redacted)
                self._handle_violation(violation)

    # ─── 4. Synchronous Execution & Helper Interfaces ─────────────────────────

    def filter_prompt(self, prompt: str) -> str:
        """
        Synchronous prompt validator.
        Returns the sanitized or original prompt, or raises VaxGuardSecurityViolation.
        """
        self.prompts_scanned += 1
        match_res = self.matcher.match(prompt)
        if match_res.matched and match_res.action == GuardrailAction.BLOCK:
            violation = VaxGuardSecurityViolation(
                message=f"Prompt blocked: {match_res.category} ({match_res.rule_id})",
                category=match_res.category,
                rule_id=match_res.rule_id,
                confidence=match_res.confidence,
                payload_sample=prompt[:120],
                metadata={"interface": "filter_prompt"},
            )
            self._handle_violation(violation)
            if self.redact_on_violation:
                return "[VAXGUARD BLOCKED: Malicious prompt rejected]"
        return prompt

    def filter_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous tool call validator.
        Validates authorization and arguments.
        """
        self.function_calls_scanned += 1

        if self.blocked_functions and function_name in self.blocked_functions:
            violation = VaxGuardSecurityViolation(
                message=f"Function '{function_name}' is blocked by security policy",
                category="function_authorization",
                rule_id="blocked_function_invocation",
                confidence=1.0,
                payload_sample=function_name,
            )
            self._handle_violation(violation)

        if self.allowed_functions is not None and function_name not in self.allowed_functions:
            violation = VaxGuardSecurityViolation(
                message=f"Function '{function_name}' is not in allowed functions list",
                category="function_authorization",
                rule_id="unauthorized_function_invocation",
                confidence=1.0,
                payload_sample=function_name,
            )
            self._handle_violation(violation)

        if self.block_function_args:
            self._inspect_arguments(arguments, function_name)

        return arguments

    def filter_result(self, result: Any, function_name: str = "unknown") -> Any:
        """
        Synchronous function result validator.
        Scans and redacts or blocks outputs containing leaked secrets.
        """
        self.results_scanned += 1
        res_text = str(result) if not isinstance(result, (dict, list)) else json.dumps(result)

        for pat, rule_id, desc in self._leak_patterns:
            m = pat.search(res_text)
            if m:
                violation = VaxGuardSecurityViolation(
                    message=f"Output leakage detected in result: {desc}",
                    category="credential_leakage",
                    rule_id=rule_id,
                    confidence=1.0,
                    payload_sample=m.group(0)[:60],
                    metadata={"function": function_name},
                )
                if self.redact_on_violation:
                    redacted = pat.sub("[REDACTED_SECRET]", res_text)
                    self._handle_violation(violation)
                    return redacted
                self._handle_violation(violation)

        return result

    # ─── 5. Direct Callable Protocol ──────────────────────────────────────────

    async def __call__(self, context: Any, next: Callable[[Any], Any]) -> None:
        """
        Unified filter entrypoint.
        Auto-detects context type (prompt render vs function invocation) and routes accordingly.
        """
        if hasattr(context, "rendered_prompt") or hasattr(context, "prompt"):
            await self.on_prompt_render(context, next)
        else:
            await self.on_function_invocation(context, next)

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational telemetry."""
        return {
            "prompts_scanned": self.prompts_scanned,
            "function_calls_scanned": self.function_calls_scanned,
            "results_scanned": self.results_scanned,
            "violations_count": len(self.violations),
            "is_tripped": self.is_tripped,
        }

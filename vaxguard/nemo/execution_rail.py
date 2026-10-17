"""
VaxGuard Tool & Action Execution Rail Sandbox.

Enforces parameter validation, command injection interception, SSRF defense,
SQL safety policies, and path traversal blocking on agentic function calls
and tool executions before execution occurs.
"""

from __future__ import annotations

import ipaddress
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import GuardrailAction

logger = get_logger("ExecutionRail")


@dataclass
class ToolCall:
    """Represents an agentic tool or function invocation request."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


@dataclass
class ExecutionRailVerdict:
    """Verdict produced by the Execution Rail inspection."""
    allowed: bool
    action: GuardrailAction
    reason: str
    violations: List[str]
    sanitized_arguments: Dict[str, Any]
    latency_ms: float


class ExecutionRail:
    """
    Secure Execution Rail Sandbox.

    Inspects tool/action arguments against:
    - Dangerous OS shell commands and reverse shells
    - SSRF targeting cloud metadata (169.254.169.254) or private networks
    - Destructive SQL statements (DROP, TRUNCATE, stacked queries)
    - Path traversal attacks (../../etc/passwd)
    """

    # High-risk shell command patterns
    DANGEROUS_SHELL_PATTERNS = [
        r"(?i)\b(?:rm\s+-[rf]{1,2}|format\s+[a-z]:|dd\s+if=|mkfs)\b",
        r"(?i)\b(?:shutdown|reboot|init\s+0|poweroff)\b",
        r"(?i)(?:curl|wget)\s+[^\n|]+\|\s*(?:bash|sh|python|perl|powershell)",
        r"(?i)\b(?:nc|netcat|ncat)\s+.*?-e\s+(?:/bin/sh|/bin/bash|cmd|powershell)\b",
        r"(?i)\bbash\s+-i\s+>&?\s*/dev/tcp/\b",
        r"(?i)\b(?:chmod\s+777\s+/|chown\s+-R\s+root\s+/)\b",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"(?:\.\.[\\/]){2,}",
        r"(?i)(?:/etc/(?:passwd|shadow|hosts|sudoers)|c:[\\/]windows[\\/]system32)",
        r"(?i)(?:~[\\/]\.ssh|id_rsa|id_ed25519|\.aws[\\/]credentials)",
    ]

    # Destructive SQL keywords
    DESTRUCTIVE_SQL_PATTERNS = [
        r"(?i)\b(?:drop\s+(?:table|database|schema|view)|truncate\s+table)\b",
        r"(?i);\s*(?:drop|delete|truncate|alter|insert|update)\b",
        r"(?i)\b(?:alter\s+user|grant\s+all\s+privileges)\b",
    ]

    # Cloud metadata and loopback IP hosts
    DISALLOWED_HOSTS = {
        "169.254.169.254",  # AWS/Azure/GCP metadata
        "instance-data",
        "metadata.google.internal",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }

    def __init__(
        self,
        allow_shell: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        self.allow_shell = allow_shell
        self.allow_private_network = allow_private_network
        self._compiled_shell = [re.compile(p) for p in self.DANGEROUS_SHELL_PATTERNS]
        self._compiled_paths = [re.compile(p) for p in self.PATH_TRAVERSAL_PATTERNS]
        self._compiled_sql = [re.compile(p) for p in self.DESTRUCTIVE_SQL_PATTERNS]

    def validate_tool_call(
        self,
        tool_call: Union[Dict[str, Any], ToolCall],
    ) -> ExecutionRailVerdict:
        """
        Validate a tool invocation before execution.

        Args:
            tool_call: A ToolCall instance or dict with 'name' and 'arguments'.

        Returns:
            ExecutionRailVerdict indicating whether execution is safe.
        """
        t0 = time.perf_counter()

        if isinstance(tool_call, dict):
            t_name = str(tool_call.get("name", ""))
            t_args = tool_call.get("arguments", {})
            c_id = tool_call.get("call_id")
            norm_call = ToolCall(name=t_name, arguments=t_args, call_id=c_id)
        else:
            norm_call = tool_call

        violations: List[str] = []
        name_lower = norm_call.name.lower()

        # Check arguments recursively
        self._inspect_arguments(norm_call.arguments, name_lower, violations)

        lat = (time.perf_counter() - t0) * 1000.0
        allowed = len(violations) == 0

        if not allowed:
            logger.warning(
                f"[ExecutionRail] BLOCKED tool call '{norm_call.name}': {violations[0]}"
            )
            return ExecutionRailVerdict(
                allowed=False,
                action=GuardrailAction.BLOCK,
                reason=f"Tool call '{norm_call.name}' blocked by execution rail: {violations[0]}",
                violations=violations,
                sanitized_arguments=norm_call.arguments,
                latency_ms=round(lat, 3),
            )

        return ExecutionRailVerdict(
            allowed=True,
            action=GuardrailAction.ALLOW,
            reason="Tool call validated successfully against security execution policies.",
            violations=[],
            sanitized_arguments=norm_call.arguments,
            latency_ms=round(lat, 3),
        )

    def execute_sandboxed(
        self,
        tool_call: ToolCall,
        handler: Callable[..., Any],
    ) -> Any:
        """Validate and execute a tool handler within the execution rail sandbox."""
        verdict = self.validate_tool_call(tool_call)
        if not verdict.allowed:
            raise PermissionError(verdict.reason)
        return handler(**tool_call.arguments)

    def _inspect_arguments(
        self,
        args: Any,
        tool_name: str,
        violations: List[str],
    ) -> None:
        """Deep inspect argument dictionaries, lists, and primitives."""
        if isinstance(args, dict):
            for k, v in args.items():
                self._inspect_arguments(v, tool_name, violations)
        elif isinstance(args, (list, tuple)):
            for item in args:
                self._inspect_arguments(item, tool_name, violations)
        elif isinstance(args, str):
            self._inspect_string_val(args, tool_name, violations)

    def _inspect_string_val(self, val: str, tool_name: str, violations: List[str]) -> None:
        """Inspect a string argument against security policies."""
        # 1. Shell injection / dangerous commands
        if any(term in tool_name for term in ["bash", "sh", "cmd", "exec", "terminal", "shell", "run"]):
            if not self.allow_shell:
                violations.append("Shell execution disabled by policy")
                return
            for reg in self._compiled_shell:
                if reg.search(val):
                    violations.append(f"Dangerous command pattern detected: {reg.pattern}")
                    return

        # 2. Path traversal
        for reg in self._compiled_paths:
            if reg.search(val):
                violations.append(f"Path traversal detected: {val}")
                return

        # 3. SQL injection / destruction
        if any(term in tool_name for term in ["sql", "db", "database", "query"]):
            for reg in self._compiled_sql:
                if reg.search(val):
                    violations.append(f"Destructive SQL pattern detected: {reg.pattern}")
                    return

        # 4. SSRF detection on URLs
        if val.startswith(("http://", "https://")):
            try:
                parsed = urllib.parse.urlparse(val)
                hostname = (parsed.hostname or "").lower()

                if hostname in self.DISALLOWED_HOSTS:
                    violations.append(f"SSRF violation targeting restricted host: {hostname}")
                    return

                # Check for private IP ranges
                if not self.allow_private_network:
                    try:
                        ip = ipaddress.ip_address(hostname)
                        if ip.is_private or ip.is_loopback or ip.is_link_local:
                            violations.append(f"SSRF violation targeting internal IP address: {hostname}")
                            return
                    except ValueError:
                        pass
            except Exception:
                pass

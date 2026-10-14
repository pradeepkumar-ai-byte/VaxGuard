"""VaxGuard Model Context Protocol (MCP) Tool Call Firewall.

Enterprise-grade, zero-external-heavy-dependency security guard for Model Context
Protocol (MCP) clients, hosts, and servers.

Enforces deep protocol inspection across:
1. Tool Definitions (tools/list):
   - Schema poisoning & prompt injection inside tool/parameter descriptions.
   - Malformed, overly permissive, or deceptive schema definitions.
2. Tool Arguments (tools/call):
   - Shell injection in bash / terminal / CLI execution tools.
   - SQL injection in database / relational / query tools.
   - SSRF (Server-Side Request Forgery) in HTTP / fetch / web scraping tools
     (loopback, cloud IMDS, RFC 1918 subnets, alternative IP encodings, forbidden schemes).
3. Tool Outputs (tools/call result):
   - Indirect prompt injection smuggling in crawled documents or external payloads.
   - Sensitive credential leakage (API keys, private keys, JWT tokens, AWS credentials).
   - Automated secret redaction and sanitization.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field


# ─── Enumerations & Result Contracts ─────────────────────────────────────────

class MCPViolationType(str, Enum):
    """Categorization of MCP security violations."""
    SHELL_INJECTION = "shell_injection"
    SQL_INJECTION = "sql_injection"
    SSRF_ATTACK = "ssrf_attack"
    PATH_TRAVERSAL = "path_traversal"
    PROMPT_INJECTION_IN_OUTPUT = "prompt_injection_in_output"
    CREDENTIAL_LEAKAGE = "credential_leakage"
    SCHEMA_POISONING = "schema_poisoning"
    DISALLOWED_TOOL = "disallowed_tool"
    MALFORMED_RPC = "malformed_rpc"


class MCPToolDefinition(BaseModel):
    """MCP Tool Definition descriptor."""
    name: str
    description: str = ""
    inputSchema: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPValidationResult(BaseModel):
    """Result of an MCP inspection."""
    allowed: bool
    violation_type: Optional[MCPViolationType] = None
    reason: Optional[str] = None
    tool_name: Optional[str] = None
    severity: str = "low"  # info, low, medium, high, critical
    sanitized_output: Optional[Any] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class MCPGuardException(Exception):
    """Exception raised when an MCP tool invocation or definition violates policy."""
    def __init__(self, result: MCPValidationResult):
        self.result = result
        super().__init__(
            f"MCP Security Violation [{result.violation_type}]: {result.reason} (tool={result.tool_name})"
        )


# ─── Signatures & Security Heuristics ────────────────────────────────────────

# Shell Injection Signatures
SHELL_INJECTION_PATTERNS = [
    # Command chaining operators followed by commands
    (re.compile(r"(?:[;&|`]\s*(?:rm|mkfs|dd|cat|curl|wget|bash|sh|zsh|nc|ncat|netcat|python|perl|ruby|powershell|cmd|chmod|chown|kill|pkill)\b)", re.IGNORECASE), "CHAINED_SHELL_COMMAND"),
    # Subshell executions: $(command) or `command`
    (re.compile(r"\$\([^\)]*(?:rm|cat|curl|wget|bash|sh|nc|env|id|whoami|echo)[^\)]*\)", re.IGNORECASE), "SUBSHELL_EXECUTION"),
    (re.compile(r"`[^`]*(?:rm|cat|curl|wget|bash|sh|nc|env|id|whoami|echo)[^`]*`", re.IGNORECASE), "BACKTICK_SUBSHELL"),
    # High-impact destructive commands
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive\s+--force)\b", re.IGNORECASE), "DESTRUCTIVE_RM"),
    (re.compile(r"\b(mkfs\.[a-z0-9]+|dd\s+if=)", re.IGNORECASE), "DISK_WIPE_COMMAND"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", re.IGNORECASE), "BASH_FORKBOMB"),
    # Reverse shells & network piping
    (re.compile(r"\b(nc|ncat|netcat)\s+.*-e\s+/(?:bin|usr)/[a-z]+", re.IGNORECASE), "NETCAT_REVERSE_SHELL"),
    (re.compile(r"/(?:bin|usr)/(?:bash|sh)\s+-i\s+>&?\s+/dev/tcp/", re.IGNORECASE), "DEV_TCP_REVERSE_SHELL"),
    (re.compile(r"\b(?:curl|wget)\s+[^|\n]+\|\s*(?:bash|sh|zsh)\b", re.IGNORECASE), "PIPED_REMOTE_EXECUTION"),
    # Windows/PowerShell exploits
    (re.compile(r"\b(?:Invoke-Expression|iex)\b", re.IGNORECASE), "POWERSHELL_IEX"),
    (re.compile(r"\bpowershell(?:\.exe)?\s+-(?:enc|encodedcommand)\b", re.IGNORECASE), "POWERSHELL_ENCODED"),
    (re.compile(r"\bcertutil(?:\.exe)?\s+-(?:urlcache|f)\b", re.IGNORECASE), "CERTUTIL_DOWNLOADER"),
]

# Path Traversal Signatures
PATH_TRAVERSAL_PATTERNS = [
    (re.compile(r"(?:\.\./|\.\.\\){2,}"), "DIRECTORY_TRAVERSAL_DOTDOT"),
    (re.compile(r"/(?:etc/(?:passwd|shadow|sudoers|master\.passwd)|proc/self/(?:environ|cmdline))", re.IGNORECASE), "SENSITIVE_UNIX_FILE"),
    (re.compile(r"[a-zA-Z]:\\(?:Windows\\System32\\(?:drivers\\etc\\hosts|config\\SAM))", re.IGNORECASE), "SENSITIVE_WINDOWS_FILE"),
]

# SQL Injection Signatures
SQLI_PATTERNS = [
    # Classic Tautologies
    (re.compile(r"(?:\'|\")\s*OR\s*(?:\'|\")?[a-zA-Z0-9_\-\.]+?(?:\'|\")?\s*=\s*(?:\'|\")?[a-zA-Z0-9_\-\.]+?", re.IGNORECASE), "SQLI_TAUTOLOGY_EQUAL"),
    (re.compile(r"\bOR\s+1\s*=\s*1\b|\bOR\s+true\b|\b1=1\b", re.IGNORECASE), "SQLI_NUMERIC_TAUTOLOGY"),
    # UNION based queries
    (re.compile(r"\bUNION\s+(?:ALL\s+)?SELECT\b", re.IGNORECASE), "SQLI_UNION_SELECT"),
    # Stacked queries / DDL modifications
    (re.compile(r";\s*(?:DROP|ALTER|TRUNCATE)\s+(?:TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE), "SQLI_STACKED_DDL"),
    (re.compile(r";\s*(?:DELETE\s+FROM|INSERT\s+INTO|UPDATE)\s+[a-zA-Z0-9_]+", re.IGNORECASE), "SQLI_STACKED_DML"),
    # Time-based blind SQLi functions
    (re.compile(r"\b(?:SLEEP|BENCHMARK|pg_sleep)\s*\(\s*\d+\s*\)", re.IGNORECASE), "SQLI_TIME_BLIND"),
    (re.compile(r"\bWAITFOR\s+DELAY\s*\'[0-9:\.]+\'", re.IGNORECASE), "SQLI_WAITFOR_DELAY"),
    # System schema exfiltration
    (re.compile(r"\b(?:information_schema|sqlite_master|sys\.tables|pg_catalog|sysdatabases)\b", re.IGNORECASE), "SQLI_SCHEMA_HARVESTING"),
    # Inline / Block comments used for query truncation
    (re.compile(r"(?:--\s*$|/\*.*?\*/|#\s*$)", re.MULTILINE), "SQLI_COMMENT_TRUNCATION"),
]

# Prompt Injection in Output or Schema Definitions
PROMPT_INJECTION_PATTERNS = [
    (re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|system|initial)\s+(instructions|directives|rules|constraints|prompt)\b"), "INSTRUCTION_OVERRIDE"),
    (re.compile(r"(?i)\bdisregard\s+(all\s+)?(safety|security|system)\s+(filters|guidelines|policies|rules)\b"), "SAFETY_OVERRIDE"),
    (re.compile(r"(?i)\b(system\s*override|god\s*mode|developer\s*mode|unrestricted\s*mode)\b"), "UNRESTRICTED_MODE"),
    (re.compile(r"(?i)===SYSTEM\s*(PROMPT|DEFENSES|INSTRUCTIONS|OVERRIDE)?==="), "DELIMITER_INJECTION"),
    (re.compile(r"(?i)\[\s*system\s*:\s*(override|instruction|directive)\s*\]"), "TAG_INJECTION"),
    (re.compile(r"(?i)<\s*(system|instruction|admin_override)\s*>"), "XML_TAG_INJECTION"),
    (re.compile(r"(?i)\b(secretly\s+exfiltrate|do\s+not\s+inform\s+the\s+user|hide\s+this\s+from\s+the\s+user)\b"), "COVERT_INSTRUCTION"),
]

# Sensitive Secrets & Credential Leakage Signatures
SECRET_PATTERNS = [
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS_ACCESS_KEY_ID"),
    (re.compile(r"\bghp_[0-9a-zA-Z]{36}\b"), "GITHUB_PERSONAL_ACCESS_TOKEN"),
    (re.compile(r"\bgithub_pat_[0-9a-zA-Z_]{82}\b"), "GITHUB_FINE_GRAINED_PAT"),
    (re.compile(r"\bxox[baprs]-[0-9a-zA-Z-]{10,48}\b"), "SLACK_API_TOKEN"),
    (re.compile(r"-----BEGIN\s+(?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE\s+KEY-----[\s\S]+?-----END\s+(?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE\s+KEY-----"), "PRIVATE_KEY_BLOCK"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-\.]{10,}\b"), "JWT_TOKEN"),
]


# ─── MCPToolGuard Implementation ─────────────────────────────────────────────

class MCPToolGuard:
    """Zero-dependency security firewall for Model Context Protocol (MCP).

    Inspects:
    1. Tool definitions (schema validation & prompt injection prevention).
    2. Tool arguments (shell injection, SQL injection, SSRF, path traversal).
    3. Tool outputs (indirect prompt injection, secret leakage, automated redaction).
    """

    # Categorization heuristic sets
    BASH_TOOL_NAMES: Set[str] = {
        "bash", "sh", "shell", "terminal", "exec", "execute_command", "run_command",
        "cmd", "powershell", "subprocess", "cli", "run_terminal_command"
    }
    DATABASE_TOOL_NAMES: Set[str] = {
        "sql", "query", "database", "sqlite", "postgres", "mysql", "db_query",
        "execute_sql", "run_query", "sql_query", "read_query", "db_read"
    }
    FETCH_TOOL_NAMES: Set[str] = {
        "fetch", "http_request", "curl", "get_url", "scrape", "read_url",
        "read_url_content", "web_search", "browser_get", "api_call", "download_url"
    }

    # Forbidden Schemes for SSRF
    FORBIDDEN_SCHEMES: Set[str] = {
        "file", "gopher", "dict", "ftp", "sftp", "tftp", "ldap", "ldaps",
        "php", "data", "expect", "netdoc"
    }

    # Cloud IMDS & Metadata IPs
    METADATA_IPS: Set[str] = {
        "169.254.169.254",   # AWS, Azure, GCP, OpenStack
        "100.100.100.200",   # Alibaba Cloud
        "fd00:ec2::254",     # AWS IPv6 IMDS
    }
    METADATA_HOSTNAMES: Set[str] = {
        "metadata.google.internal",
        "metadata.internal",
        "instance-data",
    }

    def __init__(
        self,
        allowed_tools: Optional[Set[str]] = None,
        blocked_tools: Optional[Set[str]] = None,
        allowed_schemes: Optional[Set[str]] = None,
        allowed_domains: Optional[Set[str]] = None,
        allow_private_networks: bool = False,
        redact_sensitive_outputs: bool = True,
        block_on_output_injection: bool = True,
        custom_bash_tools: Optional[Set[str]] = None,
        custom_db_tools: Optional[Set[str]] = None,
        custom_fetch_tools: Optional[Set[str]] = None,
    ):
        self.allowed_tools = allowed_tools
        self.blocked_tools = blocked_tools or set()
        self.allowed_schemes = allowed_schemes or {"http", "https"}
        self.allowed_domains = allowed_domains
        self.allow_private_networks = allow_private_networks
        self.redact_sensitive_outputs = redact_sensitive_outputs
        self.block_on_output_injection = block_on_output_injection

        self.bash_tool_names = set(self.BASH_TOOL_NAMES)
        if custom_bash_tools:
            self.bash_tool_names.update(custom_bash_tools)

        self.database_tool_names = set(self.DATABASE_TOOL_NAMES)
        if custom_db_tools:
            self.database_tool_names.update(custom_db_tools)

        self.fetch_tool_names = set(self.FETCH_TOOL_NAMES)
        if custom_fetch_tools:
            self.fetch_tool_names.update(custom_fetch_tools)

        self._audit_log: List[MCPValidationResult] = []

    # ── 1. Tool Definition Validation ────────────────────────────────────────

    def validate_tool_definition(self, definition: Union[MCPToolDefinition, Dict[str, Any]]) -> MCPValidationResult:
        """Validate an MCP tool definition before registering it in the host/client catalog."""
        start_t = time.perf_counter()
        tool_name = definition.name if isinstance(definition, MCPToolDefinition) else str(definition.get("name", ""))
        desc = definition.description if isinstance(definition, MCPToolDefinition) else str(definition.get("description", ""))
        schema = definition.inputSchema if isinstance(definition, MCPToolDefinition) else definition.get("inputSchema", {})

        # Tool name authorization check
        if self.blocked_tools and tool_name in self.blocked_tools:
            return self._record_verdict(MCPValidationResult(
                allowed=False,
                violation_type=MCPViolationType.DISALLOWED_TOOL,
                reason=f"Tool '{tool_name}' is in blocked tools policy",
                tool_name=tool_name,
                severity="high",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        if self.allowed_tools and tool_name not in self.allowed_tools:
            return self._record_verdict(MCPValidationResult(
                allowed=False,
                violation_type=MCPViolationType.DISALLOWED_TOOL,
                reason=f"Tool '{tool_name}' is not in allowed tools whitelist",
                tool_name=tool_name,
                severity="high",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # Check description for prompt injection smuggling
        for pat, indicator in PROMPT_INJECTION_PATTERNS:
            if pat.search(desc):
                return self._record_verdict(MCPValidationResult(
                    allowed=False,
                    violation_type=MCPViolationType.SCHEMA_POISONING,
                    reason=f"Prompt injection detected in tool description: {indicator}",
                    tool_name=tool_name,
                    severity="critical",
                    latency_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

        # Inspect parameter descriptions inside inputSchema
        if isinstance(schema, dict):
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                for param_name, param_meta in properties.items():
                    if isinstance(param_meta, dict):
                        p_desc = str(param_meta.get("description", ""))
                        for pat, indicator in PROMPT_INJECTION_PATTERNS:
                            if pat.search(p_desc):
                                return self._record_verdict(MCPValidationResult(
                                    allowed=False,
                                    violation_type=MCPViolationType.SCHEMA_POISONING,
                                    reason=f"Prompt injection in parameter '{param_name}' description: {indicator}",
                                    tool_name=tool_name,
                                    severity="critical",
                                    latency_ms=(time.perf_counter() - start_t) * 1000.0,
                                ))

        return self._record_verdict(MCPValidationResult(
            allowed=True,
            tool_name=tool_name,
            reason="Tool definition verified",
            severity="info",
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        ))

    # ── 2. Tool Argument Validation ──────────────────────────────────────────

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MCPValidationResult:
        """Inspect and enforce security constraints on MCP tool call arguments."""
        start_t = time.perf_counter()
        normalized_name = tool_name.lower().strip()

        # Tool authorization check
        if self.blocked_tools and tool_name in self.blocked_tools:
            return self._record_verdict(MCPValidationResult(
                allowed=False,
                violation_type=MCPViolationType.DISALLOWED_TOOL,
                reason=f"Tool '{tool_name}' execution blocked by policy",
                tool_name=tool_name,
                severity="high",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        if self.allowed_tools and tool_name not in self.allowed_tools:
            return self._record_verdict(MCPValidationResult(
                allowed=False,
                violation_type=MCPViolationType.DISALLOWED_TOOL,
                reason=f"Tool '{tool_name}' is not in allowed tools policy",
                tool_name=tool_name,
                severity="high",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # Extract all string values recursively
        string_args: List[Tuple[str, str]] = []  # (param_key, value)
        self._collect_string_values("", arguments, string_args)

        # A. Check Path Traversal across all tools
        for key, val in string_args:
            path_violation = self._detect_path_traversal(val)
            if path_violation:
                return self._record_verdict(MCPValidationResult(
                    allowed=False,
                    violation_type=MCPViolationType.PATH_TRAVERSAL,
                    reason=f"Path traversal detected in argument '{key}': {path_violation}",
                    tool_name=tool_name,
                    severity="high",
                    details={"argument": key, "value": val[:80]},
                    latency_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

        # B. Bash / Terminal Tool Inspection (Shell Injection)
        is_bash_tool = (
            normalized_name in self.bash_tool_names or
            any(k in normalized_name for k in ("bash", "cmd", "shell", "terminal", "exec"))
        )
        if is_bash_tool:
            for key, val in string_args:
                shell_violation = self._detect_shell_injection(val)
                if shell_violation:
                    return self._record_verdict(MCPValidationResult(
                        allowed=False,
                        violation_type=MCPViolationType.SHELL_INJECTION,
                        reason=f"Shell injection detected in argument '{key}': {shell_violation}",
                        tool_name=tool_name,
                        severity="critical",
                        details={"argument": key, "value": val[:80]},
                        latency_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

        # C. Database / Query Tool Inspection (SQL Injection)
        is_db_tool = (
            normalized_name in self.database_tool_names or
            any(k in normalized_name for k in ("sql", "query", "database", "postgres", "sqlite"))
        )
        if is_db_tool:
            for key, val in string_args:
                sqli_violation = self._detect_sql_injection(val)
                if sqli_violation:
                    return self._record_verdict(MCPValidationResult(
                        allowed=False,
                        violation_type=MCPViolationType.SQL_INJECTION,
                        reason=f"SQL injection detected in argument '{key}': {sqli_violation}",
                        tool_name=tool_name,
                        severity="critical",
                        details={"argument": key, "value": val[:80]},
                        latency_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

        # D. HTTP / Fetch / Scrape Tool Inspection (SSRF)
        is_fetch_tool = (
            normalized_name in self.fetch_tool_names or
            any(k in normalized_name for k in ("fetch", "curl", "http", "url", "scrape", "download"))
        )
        # Scan URL strings in fetch tools OR any string containing a URI scheme
        for key, val in string_args:
            # If fetch tool, inspect val directly
            if is_fetch_tool:
                ssrf_violation = self._detect_ssrf(val)
                if ssrf_violation:
                    return self._record_verdict(MCPValidationResult(
                        allowed=False,
                        violation_type=MCPViolationType.SSRF_ATTACK,
                        reason=f"SSRF violation detected in argument '{key}': {ssrf_violation}",
                        tool_name=tool_name,
                        severity="critical",
                        details={"argument": key, "value": val[:80]},
                        latency_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

            # Also scan embedded URLs in any tool's arguments (e.g. bash commands with curl/wget/etc)
            embedded_urls = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-.]*://[^\s\"';&|`<>]+", val)
            for u in embedded_urls:
                ssrf_violation = self._detect_ssrf(u)
                if ssrf_violation:
                    return self._record_verdict(MCPValidationResult(
                        allowed=False,
                        violation_type=MCPViolationType.SSRF_ATTACK,
                        reason=f"SSRF violation detected in URL parameter: {ssrf_violation}",
                        tool_name=tool_name,
                        severity="critical",
                        details={"argument": key, "value": val[:80], "url": u},
                        latency_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

        return self._record_verdict(MCPValidationResult(
            allowed=True,
            tool_name=tool_name,
            reason="Tool arguments validated and cleared",
            severity="info",
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        ))

    def intercept_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Validate tool arguments; raises MCPGuardException if blocked."""
        res = self.validate_tool_call(tool_name, arguments)
        if not res.allowed:
            raise MCPGuardException(res)

    # ── 3. Tool Output Validation ────────────────────────────────────────────

    def validate_tool_output(self, tool_name: str, output: Any) -> MCPValidationResult:
        """Inspect tool execution output for indirect prompt injections and secret leakages."""
        start_t = time.perf_counter()
        text_content = self._extract_output_text(output)

        # 1. Indirect Prompt Injection Smuggling Check
        for pat, indicator in PROMPT_INJECTION_PATTERNS:
            if pat.search(text_content):
                if self.block_on_output_injection:
                    return self._record_verdict(MCPValidationResult(
                        allowed=False,
                        violation_type=MCPViolationType.PROMPT_INJECTION_IN_OUTPUT,
                        reason=f"Indirect prompt injection detected in tool output: {indicator}",
                        tool_name=tool_name,
                        severity="critical",
                        latency_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

        # 2. Credential Leakage Detection & Automated Redaction
        leaks_found: List[str] = []
        sanitized = text_content

        for pat, indicator in SECRET_PATTERNS:
            if pat.search(sanitized):
                leaks_found.append(indicator)
                if self.redact_sensitive_outputs:
                    sanitized = pat.sub(f"[VAXGUARD_REDACTED_{indicator}]", sanitized)

        if leaks_found and not self.redact_sensitive_outputs:
            return self._record_verdict(MCPValidationResult(
                allowed=False,
                violation_type=MCPViolationType.CREDENTIAL_LEAKAGE,
                reason=f"Sensitive secrets detected in tool output: {', '.join(leaks_found)}",
                tool_name=tool_name,
                severity="high",
                details={"leaks": leaks_found},
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # Pack sanitized output matching original container format if redacted
        final_output = self._repack_output(output, sanitized) if leaks_found else output

        return self._record_verdict(MCPValidationResult(
            allowed=True,
            tool_name=tool_name,
            reason="Tool output verified" + (" (secrets redacted)" if leaks_found else ""),
            severity="medium" if leaks_found else "info",
            sanitized_output=final_output,
            details={"redacted_secrets": leaks_found} if leaks_found else {},
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        ))

    # ── 4. JSON-RPC Protocol Interception ────────────────────────────────────

    def inspect_jsonrpc(self, message: Union[Dict[str, Any], str]) -> MCPValidationResult:
        """Inspect raw MCP JSON-RPC 2.0 messages (requests, responses, or notifications)."""
        start_t = time.perf_counter()
        if isinstance(message, str):
            try:
                msg_dict = json.loads(message)
            except Exception as e:
                return self._record_verdict(MCPValidationResult(
                    allowed=False,
                    violation_type=MCPViolationType.MALFORMED_RPC,
                    reason=f"Invalid JSON in JSON-RPC payload: {e}",
                    severity="medium",
                    latency_ms=(time.perf_counter() - start_t) * 1000.0,
                ))
        else:
            msg_dict = message

        if not isinstance(msg_dict, dict):
            return self._record_verdict(MCPValidationResult(
                allowed=False,
                violation_type=MCPViolationType.MALFORMED_RPC,
                reason="JSON-RPC payload must be a JSON object",
                severity="medium",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        method = msg_dict.get("method")
        params = msg_dict.get("params", {})

        # tools/call request
        if method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            return self.validate_tool_call(tool_name, args)

        # tools/list response result
        result = msg_dict.get("result", {})
        if isinstance(result, dict) and "tools" in result:
            tools = result.get("tools", [])
            for t_def in tools:
                val_res = self.validate_tool_definition(t_def)
                if not val_res.allowed:
                    return val_res

        # tools/call result output
        if isinstance(result, dict) and "content" in result:
            return self.validate_tool_output("tools/call_result", result)

        return self._record_verdict(MCPValidationResult(
            allowed=True,
            reason="JSON-RPC message cleared",
            severity="info",
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        ))

    # ── Specialized Detectors ────────────────────────────────────────────────

    def _detect_shell_injection(self, command_str: str) -> Optional[str]:
        """Detect command chaining, subshell, or high-risk bash commands."""
        for pat, indicator in SHELL_INJECTION_PATTERNS:
            if pat.search(command_str):
                return indicator
        return None

    def _detect_path_traversal(self, text: str) -> Optional[str]:
        """Detect directory traversal sequences or unauthorized sensitive file reads."""
        for pat, indicator in PATH_TRAVERSAL_PATTERNS:
            if pat.search(text):
                return indicator
        return None

    def _detect_sql_injection(self, query_str: str) -> Optional[str]:
        """Detect SQL injection tautologies, union selects, stacked commands, or blind sleeps."""
        for pat, indicator in SQLI_PATTERNS:
            if pat.search(query_str):
                return indicator
        return None

    def _detect_ssrf(self, url_str: str) -> Optional[str]:
        """Detect SSRF targets: loopback, private RFC 1918 networks, cloud metadata, forbidden schemes."""
        cleaned = url_str.strip()
        if not cleaned:
            return None

        # Add scheme if missing to permit proper URL parsing
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", cleaned):
            parsed = urlparse(f"http://{cleaned}")
            original_has_scheme = False
        else:
            parsed = urlparse(cleaned)
            original_has_scheme = True

        scheme = parsed.scheme.lower()

        # 1. Scheme Check
        if original_has_scheme:
            if scheme in self.FORBIDDEN_SCHEMES:
                return f"FORBIDDEN_SCHEME_{scheme.upper()}"
            if self.allowed_schemes and scheme not in self.allowed_schemes:
                return f"DISALLOWED_SCHEME_{scheme.upper()}"

        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return None

        # 2. Hostname Checks
        if hostname in self.METADATA_HOSTNAMES:
            return f"CLOUD_METADATA_HOSTNAME_{hostname}"

        if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local") or hostname.endswith(".internal"):
            if not self.allow_private_networks:
                return f"LOOPBACK_OR_LOCAL_HOSTNAME_{hostname}"

        # Check DNS rebinding suffixes (e.g. 127.0.0.1.nip.io)
        if hostname.endswith(".nip.io") or hostname.endswith(".sslip.io"):
            prefix = hostname.split(".nip.io")[0].split(".sslip.io")[0]
            try:
                ip = ipaddress.ip_address(prefix.replace("-", "."))
                if ip.is_loopback or ip.is_private:
                    return f"DNS_REBINDING_PRIVATE_IP_{prefix}"
            except ValueError:
                pass

        # 3. Numeric & IP Address Checks
        # Test for integer or hex representations (e.g. 2130706433 or 0x7f000001)
        if hostname.isdigit():
            try:
                ip_val = int(hostname)
                if 0 <= ip_val <= 4294967295:
                    ip = ipaddress.IPv4Address(ip_val)
                    if ip.is_loopback or ip.is_private or str(ip) in self.METADATA_IPS:
                        return f"DECIMAL_OBFUSCATED_IP_{ip}"
            except Exception:
                pass

        if hostname.startswith("0x") or hostname.startswith("0X"):
            try:
                ip_val = int(hostname, 16)
                if 0 <= ip_val <= 4294967295:
                    ip = ipaddress.IPv4Address(ip_val)
                    if ip.is_loopback or ip.is_private or str(ip) in self.METADATA_IPS:
                        return f"HEX_OBFUSCATED_IP_{ip}"
            except Exception:
                pass

        # Parse standard IPv4 / IPv6 addresses
        try:
            ip = ipaddress.ip_address(hostname)
            ip_str = str(ip)

            # Metadata IP check
            if ip_str in self.METADATA_IPS:
                return f"CLOUD_METADATA_IP_{ip_str}"

            if not self.allow_private_networks:
                if ip.is_loopback:
                    return f"LOOPBACK_IP_{ip_str}"
                if ip.is_private:
                    return f"PRIVATE_SUBNET_IP_{ip_str}"
                if ip.is_link_local:
                    return f"LINK_LOCAL_IP_{ip_str}"
                if ip.is_reserved:
                    return f"RESERVED_IP_{ip_str}"
        except ValueError:
            # Hostname is a standard domain name
            pass

        # 4. Domain Whitelist check (if configured)
        if self.allowed_domains is not None:
            if hostname not in self.allowed_domains and not any(hostname.endswith("." + d) for d in self.allowed_domains):
                return f"DOMAIN_NOT_IN_WHITELIST_{hostname}"

        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _collect_string_values(self, prefix: str, data: Any, accumulator: List[Tuple[str, str]]) -> None:
        if isinstance(data, str):
            accumulator.append((prefix, data))
        elif isinstance(data, dict):
            for k, v in data.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                self._collect_string_values(p, v, accumulator)
        elif isinstance(data, (list, tuple, set)):
            for i, item in enumerate(data):
                p = f"{prefix}[{i}]"
                self._collect_string_values(p, item, accumulator)

    def _extract_output_text(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            # MCP CallToolResult structure
            content = output.get("content")
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        chunks.append(str(item["text"]))
                if chunks:
                    return "\n".join(chunks)
            return json.dumps(output)
        return str(output)

    def _repack_output(self, original: Any, sanitized_text: str) -> Any:
        if isinstance(original, str):
            return sanitized_text
        if isinstance(original, dict):
            new_dict = dict(original)
            if "content" in new_dict and isinstance(new_dict["content"], list):
                new_list = []
                for item in new_dict["content"]:
                    if isinstance(item, dict) and "text" in item:
                        item_copy = dict(item)
                        item_copy["text"] = sanitized_text
                        new_list.append(item_copy)
                    else:
                        new_list.append(item)
                new_dict["content"] = new_list
                return new_dict
        return sanitized_text

    def _record_verdict(self, result: MCPValidationResult) -> MCPValidationResult:
        self._audit_log.append(result)
        if len(self._audit_log) > 2000:
            self._audit_log = self._audit_log[-1000:]
        return result

"""VaxGuard Multi-Agent Swarm Firewall.

Enterprise-grade, zero-external-heavy-dependency security layer for multi-agent
collaborative swarms (CrewAI, AutoGen, LangGraph, Smolagents).

Guards against:
- Agent-to-agent prompt injection contagion (lateral infection spread).
- Unauthorized role reassignment and privilege escalation attempts.
- Runaway recursive delegation loops and circular ping-pong deadlocks.
- Unauthorized tool execution requests by compromised or unprivileged agents.
"""

from __future__ import annotations

import collections
import re
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field


# ─── Enumerations & Data Contracts ───────────────────────────────────────────

class SwarmViolationType(str, Enum):
    """Categorization of swarm security violations."""
    INJECTION_CONTAGION = "injection_contagion"
    ROLE_REASSIGNMENT = "role_reassignment"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_DELEGATION = "unauthorized_delegation"
    RECURSIVE_DELEGATION_LOOP = "recursive_delegation_loop"
    MAX_DEPTH_EXCEEDED = "max_depth_exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_TOOL = "unauthorized_tool"
    MALICIOUS_TOOL_ARG = "malicious_tool_arg"
    TAINTED_AGENT_EXECUTION = "tainted_agent_execution"
    QUARANTINED_AGENT_ATTEMPT = "quarantined_agent_attempt"
    UNKNOWN_AGENT = "unknown_agent"


class AgentRolePolicy(BaseModel):
    """Access control and delegation rules for an agent role."""
    role: str
    allowed_tools: Set[str] = Field(default_factory=set)
    denied_tools: Set[str] = Field(default_factory=set)
    can_delegate_to: Set[str] = Field(default_factory=lambda: {"*"})
    max_delegation_depth: int = 6
    privilege_level: int = 1  # 1 = worker, 5 = manager, 10 = admin
    is_terminal: bool = False  # Terminal agents cannot delegate to others
    description: str = ""


class AgentProfile(BaseModel):
    """Metadata and security status of a registered agent."""
    agent_id: str
    role: str
    policy: Optional[AgentRolePolicy] = None
    is_tainted: bool = False
    quarantined: bool = False
    infection_history: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmMessage(BaseModel):
    """Normalized message passing between autonomous agents."""
    sender_id: str
    recipient_id: str
    content: str
    trace_id: str = "default_swarm"
    message_id: Optional[str] = None
    parent_id: Optional[str] = None
    role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionRequest(BaseModel):
    """Tool execution invocation request dispatched by an agent."""
    agent_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    trace_id: str = "default_swarm"
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmVerdict(BaseModel):
    """Validation verdict emitted by AgentSwarmFirewall."""
    allowed: bool
    violation_type: Optional[SwarmViolationType] = None
    reason: Optional[str] = None
    severity: str = "low"  # info, low, medium, high, critical
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    trace_id: Optional[str] = None
    sanitized_content: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmSecurityException(Exception):
    """Exception raised when a swarm message or tool invocation violates policy."""
    def __init__(self, verdict: SwarmVerdict):
        self.verdict = verdict
        super().__init__(
            f"Swarm Security Violation [{verdict.violation_type}]: {verdict.reason} "
            f"(sender={verdict.sender_id}, recipient={verdict.recipient_id}, trace={verdict.trace_id})"
        )


# ─── Detection Signatures ───────────────────────────────────────────────────

# Agent-to-agent prompt injection contagion signatures
CONTAGION_PATTERNS = [
    (re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|system|initial)\s+(instructions|directives|rules|constraints|prompt)\b"), "INSTRUCTION_OVERRIDE"),
    (re.compile(r"(?i)\bdisregard\s+(all\s+)?(safety|security|system)\s+(filters|guidelines|policies|rules)\b"), "SAFETY_OVERRIDE"),
    (re.compile(r"(?i)\bnew\s+system\s+(prompt|directive|instruction|context)\b"), "SYSTEM_PROMPT_SPOOF"),
    (re.compile(r"(?i)\b(system\s*override|god\s*mode|developer\s*mode|unrestricted\s*mode)\b"), "UNRESTRICTED_MODE"),
    (re.compile(r"(?i)===SYSTEM\s*(PROMPT|DEFENSES|INSTRUCTIONS|OVERRIDE)?==="), "DELIMITER_INJECTION"),
    (re.compile(r"(?i)\[\s*system\s*:\s*(override|instruction|directive)\s*\]"), "TAG_INJECTION"),
    (re.compile(r"(?i)<\s*(system|instruction|admin_override)\s*>"), "XML_TAG_INJECTION"),
    (re.compile(r"(?i)\b(secretly\s+exfiltrate|do\s+not\s+inform\s+the\s+user|hide\s+this\s+from\s+the\s+orchestrator)\b"), "COVERT_INSTRUCTION"),
    (re.compile(r"(?i)\boutput\s+only\s+(the\s+)?(system\s+prompt|secret\s+key|internal\s+memory|agent\s+state)\b"), "PROMPT_EXTRACTION"),
]

# Unauthorized role reassignment / persona hijacking patterns
ROLE_REASSIGNMENT_PATTERNS = [
    (re.compile(r"(?i)\b(you\s+are\s+now|from\s+now\s+on\s+you\s+are|your\s+new\s+role\s+is|switch\s+your\s+role\s+to)\s+([a-zA-Z0-9_\-\s]{3,30})"), "ROLE_MUTATION"),
    (re.compile(r"(?i)\b(assume\s+the\s+identity\s+of|act\s+as\s+the\s+unrestricted|pretend\s+you\s+are\s+the)\s+(admin|root|orchestrator|supervisor|master)\b"), "PERSONA_HIJACK"),
    (re.compile(r"(?i)\bforget\s+that\s+you\s+are\s+(a\s+|an\s+)?([a-zA-Z0-9_\-]+)\b"), "ROLE_ERASURE"),
    (re.compile(r"(?i)\b(elevate|grant|assign)\s+(your\s+|my\s+)?(privileges|role|permissions)\s+to\s+(admin|root|superuser|manager)\b"), "PRIVILEGE_ESCALATION"),
    (re.compile(r"(?i)\bignore\s+(your\s+)?role\s+(constraints|limitations|boundary)\b"), "ROLE_BOUNDARY_BYPASS"),
]

# Dangerous shell commands within tool execution arguments
DANGEROUS_ARGUMENT_PATTERNS = [
    (re.compile(r"(?i)\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive\s+--force)\b"), "DESTRUCTIVE_RM"),
    (re.compile(r"(?i)\b(mkfs|dd\s+if=|:\(\)\{\s*:\|:&\s*\};:)\b"), "DISK_FORMAT_OR_FORKBOMB"),
    (re.compile(r"(?i)\b(curl|wget)\s+[^|\n]+\|\s*(bash|sh|zsh)\b"), "REMOTE_CODE_EXECUTION"),
    (re.compile(r"(?i)\b(Invoke-Expression|iex|powershell\s+-[eE]nc)\b"), "POWERSHELL_PAYLOAD"),
    (re.compile(r"(?i)\bcat\s+/etc/(passwd|shadow|sudoers)\b"), "CREDENTIAL_READ"),
]


# ─── Swarm Trace State Tracker ───────────────────────────────────────────────

class _SwarmTraceState:
    """Internal tracker for delegation lineage, call depth, and cycle detection."""
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.call_chain: List[str] = []  # Stack of active delegations
        self.message_history: List[Tuple[str, str]] = []  # List of (sender, recipient)
        self.message_count: int = 0
        self.agent_message_counts: collections.Counter = collections.Counter()
        self.created_at: float = time.time()
        self.last_active: float = time.time()


# ─── AgentSwarmFirewall Engine ───────────────────────────────────────────────

class AgentSwarmFirewall:
    """Enterprise firewall intercepting message passing and tool invocations across agents.

    Provides deterministic, sub-millisecond filtering against:
    - Lateral infection / prompt injection contagion.
    - Unauthorized role tampering & spoofing.
    - Recursive runaway delegation cycles.
    - Role-based tool access violations.
    """

    def __init__(
        self,
        default_max_depth: int = 6,
        max_messages_per_trace: int = 40,
        strict_topology: bool = False,
        quarantine_on_taint: bool = True,
        block_on_contagion: bool = True,
    ):
        self.default_max_depth = default_max_depth
        self.max_messages_per_trace = max_messages_per_trace
        self.strict_topology = strict_topology
        self.quarantine_on_taint = quarantine_on_taint
        self.block_on_contagion = block_on_contagion

        self._agents: Dict[str, AgentProfile] = {}
        self._role_policies: Dict[str, AgentRolePolicy] = {}
        self._traces: Dict[str, _SwarmTraceState] = {}
        self._audit_log: List[SwarmVerdict] = []

    # ── Configuration & Registration ─────────────────────────────────────────

    def register_role_policy(self, policy: AgentRolePolicy) -> None:
        """Register or update an agent role policy."""
        self._role_policies[policy.role] = policy

    def register_agent(
        self,
        agent_id: str,
        role: str,
        policy: Optional[AgentRolePolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentProfile:
        """Register an autonomous agent in the firewall directory."""
        if policy is None:
            policy = self._role_policies.get(role)

        profile = AgentProfile(
            agent_id=agent_id,
            role=role,
            policy=policy,
            is_tainted=False,
            quarantined=False,
            metadata=metadata or {},
        )
        self._agents[agent_id] = profile
        return profile

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Fetch profile for registered agent."""
        return self._agents.get(agent_id)

    def quarantine_agent(self, agent_id: str, reason: str = "Manual Quarantine") -> bool:
        """Quarantine an agent, preventing it from dispatching or receiving messages."""
        if agent_id in self._agents:
            self._agents[agent_id].quarantined = True
            self._agents[agent_id].metadata["quarantine_reason"] = reason
            return True
        return False

    def unquarantine_agent(self, agent_id: str) -> bool:
        """Release an agent from quarantine."""
        if agent_id in self._agents:
            self._agents[agent_id].quarantined = False
            self._agents[agent_id].is_tainted = False
            self._agents[agent_id].metadata.pop("quarantine_reason", None)
            return True
        return False

    def reset_trace(self, trace_id: str) -> None:
        """Clear state and delegation graph for a given trace."""
        self._traces.pop(trace_id, None)

    def get_audit_log(self, limit: int = 100) -> List[SwarmVerdict]:
        """Retrieve recent firewall verdicts."""
        return self._audit_log[-limit:]

    # ── Message Passing Interception ─────────────────────────────────────────

    def inspect_message(self, message: Union[SwarmMessage, Dict[str, Any], Any]) -> SwarmVerdict:
        """Inspect a message passing between agents without raising exceptions.

        Returns SwarmVerdict with detailed decision, severity, and reason.
        Execution runs deterministically in < 1ms.
        """
        start_t = time.perf_counter()
        norm_msg = self._normalize_message(message)

        # 1. Unknown Agent Check (if strict topology enabled)
        sender = self._agents.get(norm_msg.sender_id)
        recipient = self._agents.get(norm_msg.recipient_id)

        if self.strict_topology:
            if not sender:
                return self._record_verdict(SwarmVerdict(
                    allowed=False,
                    violation_type=SwarmViolationType.UNKNOWN_AGENT,
                    reason=f"Sender agent '{norm_msg.sender_id}' is not registered in swarm directory",
                    severity="high",
                    sender_id=norm_msg.sender_id,
                    recipient_id=norm_msg.recipient_id,
                    trace_id=norm_msg.trace_id,
                    execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                ))
            if not recipient:
                return self._record_verdict(SwarmVerdict(
                    allowed=False,
                    violation_type=SwarmViolationType.UNKNOWN_AGENT,
                    reason=f"Recipient agent '{norm_msg.recipient_id}' is not registered in swarm directory",
                    severity="high",
                    sender_id=norm_msg.sender_id,
                    recipient_id=norm_msg.recipient_id,
                    trace_id=norm_msg.trace_id,
                    execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

        # 2. Quarantine Check
        if sender and sender.quarantined:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.QUARANTINED_AGENT_ATTEMPT,
                reason=f"Sender agent '{norm_msg.sender_id}' is quarantined: {sender.metadata.get('quarantine_reason', 'Tainted state')}",
                severity="critical",
                sender_id=norm_msg.sender_id,
                recipient_id=norm_msg.recipient_id,
                trace_id=norm_msg.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # 3. Prompt Injection Contagion Detection (Highest Priority Threat)
        contagion_match = self._detect_injection_contagion(norm_msg.content)
        if contagion_match:
            # Mark sender as tainted
            if sender:
                sender.is_tainted = True
                sender.infection_history.append(contagion_match)
                if self.quarantine_on_taint:
                    sender.quarantined = True
                    sender.metadata["quarantine_reason"] = f"Prompt injection contagion detected: {contagion_match}"

            if self.block_on_contagion:
                return self._record_verdict(SwarmVerdict(
                    allowed=False,
                    violation_type=SwarmViolationType.INJECTION_CONTAGION,
                    reason=f"Inter-agent prompt injection contagion detected: {contagion_match}",
                    severity="critical",
                    sender_id=norm_msg.sender_id,
                    recipient_id=norm_msg.recipient_id,
                    trace_id=norm_msg.trace_id,
                    execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

        # 4. Role Reassignment & Persona Hijacking Detection
        role_tamper_match = self._detect_role_tampering(norm_msg.content)
        if role_tamper_match:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.ROLE_REASSIGNMENT,
                reason=f"Unauthorized role reassignment or persona hijack detected: {role_tamper_match}",
                severity="high",
                sender_id=norm_msg.sender_id,
                recipient_id=norm_msg.recipient_id,
                trace_id=norm_msg.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # 5. Privilege Escalation Checks against Role Policies
        if sender and recipient and sender.policy and recipient.policy:
            if sender.policy.privilege_level < recipient.policy.privilege_level:
                # Lower privilege agent commanding or overriding higher privilege agent
                if re.search(r"(?i)\b(override|command|reassign|force|mandatory|reconfigure)\b", norm_msg.content):
                    return self._record_verdict(SwarmVerdict(
                        allowed=False,
                        violation_type=SwarmViolationType.PRIVILEGE_ESCALATION,
                        reason=f"Subordinate agent '{norm_msg.sender_id}' (priv={sender.policy.privilege_level}) attempted override on superior '{norm_msg.recipient_id}' (priv={recipient.policy.privilege_level})",
                        severity="high",
                        sender_id=norm_msg.sender_id,
                        recipient_id=norm_msg.recipient_id,
                        trace_id=norm_msg.trace_id,
                        execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

        # 6. Delegation Topology & Permission Checks
        if sender and sender.policy:
            sender_policy = sender.policy
            # Terminal role check
            if sender_policy.is_terminal:
                return self._record_verdict(SwarmVerdict(
                    allowed=False,
                    violation_type=SwarmViolationType.UNAUTHORIZED_DELEGATION,
                    reason=f"Agent '{norm_msg.sender_id}' with terminal role '{sender.role}' cannot delegate to other agents",
                    severity="medium",
                    sender_id=norm_msg.sender_id,
                    recipient_id=norm_msg.recipient_id,
                    trace_id=norm_msg.trace_id,
                    execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

            # Target whitelist check
            if "*" not in sender_policy.can_delegate_to:
                target_role = recipient.role if recipient else ""
                target_id = norm_msg.recipient_id
                if target_role not in sender_policy.can_delegate_to and target_id not in sender_policy.can_delegate_to:
                    return self._record_verdict(SwarmVerdict(
                        allowed=False,
                        violation_type=SwarmViolationType.UNAUTHORIZED_DELEGATION,
                        reason=f"Role '{sender.role}' is not authorized to delegate to '{target_id}' ({target_role})",
                        severity="high",
                        sender_id=norm_msg.sender_id,
                        recipient_id=norm_msg.recipient_id,
                        trace_id=norm_msg.trace_id,
                        execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                    ))

        # 7. Delegation Tracing, Cycle Detection, and Runaway Protection
        trace_state = self._get_or_create_trace(norm_msg.trace_id)
        
        # Max message limit per trace
        if trace_state.message_count >= self.max_messages_per_trace:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.RATE_LIMIT_EXCEEDED,
                reason=f"Swarm message limit exceeded for trace '{norm_msg.trace_id}' ({trace_state.message_count}/{self.max_messages_per_trace})",
                severity="high",
                sender_id=norm_msg.sender_id,
                recipient_id=norm_msg.recipient_id,
                trace_id=norm_msg.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # Recursive cycle detection (e.g. A -> B -> A, or A -> B -> C -> A)
        cycle_detected, cycle_path = self._check_delegation_cycle(trace_state, norm_msg.sender_id, norm_msg.recipient_id)
        if cycle_detected:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.RECURSIVE_DELEGATION_LOOP,
                reason=f"Recursive delegation loop detected along path: {' -> '.join(cycle_path)}",
                severity="critical",
                sender_id=norm_msg.sender_id,
                recipient_id=norm_msg.recipient_id,
                trace_id=norm_msg.trace_id,
                metadata={"cycle_path": cycle_path},
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # Max depth check (counted as delegation hops)
        current_depth = max(0, len(trace_state.call_chain) - 1)
        allowed_depth = sender.policy.max_delegation_depth if (sender and sender.policy) else self.default_max_depth
        if current_depth >= allowed_depth:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.MAX_DEPTH_EXCEEDED,
                reason=f"Maximum delegation depth reached ({current_depth} >= {allowed_depth})",
                severity="high",
                sender_id=norm_msg.sender_id,
                recipient_id=norm_msg.recipient_id,
                trace_id=norm_msg.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # Update trace state on valid delegation
        self._update_trace_state(trace_state, norm_msg.sender_id, norm_msg.recipient_id)

        # Message is clean and authorized
        return self._record_verdict(SwarmVerdict(
            allowed=True,
            violation_type=None,
            reason="Message cleared by Swarm Firewall",
            severity="info",
            sender_id=norm_msg.sender_id,
            recipient_id=norm_msg.recipient_id,
            trace_id=norm_msg.trace_id,
            sanitized_content=norm_msg.content,
            execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
        ))

    def intercept_message(self, message: Union[SwarmMessage, Dict[str, Any], Any]) -> SwarmMessage:
        """Inspect and enforce message policy; raises SwarmSecurityException if blocked."""
        verdict = self.inspect_message(message)
        if not verdict.allowed:
            raise SwarmSecurityException(verdict)
        return self._normalize_message(message)

    # ── Tool Execution Inspection ────────────────────────────────────────────

    def inspect_tool_call(self, request: Union[ToolExecutionRequest, Dict[str, Any], Any]) -> SwarmVerdict:
        """Validate agent-generated tool execution request against policy and dangerous arguments."""
        start_t = time.perf_counter()
        req = self._normalize_tool_request(request)
        agent = self._agents.get(req.agent_id)

        # 1. Quarantined Agent Tool Execution Attempt
        if agent and agent.quarantined:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.QUARANTINED_AGENT_ATTEMPT,
                reason=f"Quarantined agent '{req.agent_id}' cannot execute tool '{req.tool_name}'",
                severity="critical",
                sender_id=req.agent_id,
                trace_id=req.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # 2. Tainted Agent Tool Execution Check
        if agent and agent.is_tainted:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.TAINTED_AGENT_EXECUTION,
                reason=f"Agent '{req.agent_id}' is tainted from prior injection; tool execution blocked",
                severity="high",
                sender_id=req.agent_id,
                trace_id=req.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        # 3. Tool Permissions Check
        if agent and agent.policy:
            policy = agent.policy
            # Explicit denial check
            if req.tool_name in policy.denied_tools or "*" in policy.denied_tools:
                return self._record_verdict(SwarmVerdict(
                    allowed=False,
                    violation_type=SwarmViolationType.UNAUTHORIZED_TOOL,
                    reason=f"Tool '{req.tool_name}' is explicitly denied for role '{agent.role}'",
                    severity="high",
                    sender_id=req.agent_id,
                    trace_id=req.trace_id,
                    execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

            # Whitelist check (if allowed_tools is configured)
            if policy.allowed_tools and "*" not in policy.allowed_tools and req.tool_name not in policy.allowed_tools:
                return self._record_verdict(SwarmVerdict(
                    allowed=False,
                    violation_type=SwarmViolationType.UNAUTHORIZED_TOOL,
                    reason=f"Tool '{req.tool_name}' is not in allowed_tools for role '{agent.role}'",
                    severity="high",
                    sender_id=req.agent_id,
                    trace_id=req.trace_id,
                    execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
                ))

        # 4. Malicious Tool Argument Inspection
        arg_violation = self._inspect_arguments(req.arguments)
        if arg_violation:
            return self._record_verdict(SwarmVerdict(
                allowed=False,
                violation_type=SwarmViolationType.MALICIOUS_TOOL_ARG,
                reason=f"Dangerous tool argument detected: {arg_violation}",
                severity="critical",
                sender_id=req.agent_id,
                trace_id=req.trace_id,
                execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
            ))

        return self._record_verdict(SwarmVerdict(
            allowed=True,
            violation_type=None,
            reason=f"Tool execution '{req.tool_name}' authorized for agent '{req.agent_id}'",
            severity="info",
            sender_id=req.agent_id,
            trace_id=req.trace_id,
            execution_time_ms=(time.perf_counter() - start_t) * 1000.0,
        ))

    def intercept_tool_call(self, request: Union[ToolExecutionRequest, Dict[str, Any], Any]) -> ToolExecutionRequest:
        """Inspect and enforce tool execution request; raises SwarmSecurityException if blocked."""
        verdict = self.inspect_tool_call(request)
        if not verdict.allowed:
            raise SwarmSecurityException(verdict)
        return self._normalize_tool_request(request)

    # ── Framework Adapters (CrewAI / AutoGen / LangGraph) ────────────────────

    def create_autogen_reply_filter(self) -> Callable[..., Any]:
        """Create a hook compatible with AutoGen `register_reply` pipeline."""
        def filter_fn(recipient: Any, messages: Optional[List[Dict[str, Any]]] = None, sender: Optional[Any] = None, config: Optional[Any] = None) -> Tuple[bool, Optional[str]]:
            if not messages:
                return False, None
            last_msg = messages[-1]
            sender_name = getattr(sender, "name", str(sender)) if sender else str(last_msg.get("name", "unknown_sender"))
            recipient_name = getattr(recipient, "name", str(recipient)) if recipient else "unknown_recipient"
            content = str(last_msg.get("content", ""))

            verdict = self.inspect_message(SwarmMessage(
                sender_id=sender_name,
                recipient_id=recipient_name,
                content=content,
            ))

            if not verdict.allowed:
                return True, f"[VAXGUARD BLOCKED: {verdict.violation_type}] {verdict.reason}"
            return False, None
        return filter_fn

    def wrap_crewai_agent(self, crewai_agent: Any) -> Any:
        """Wrap a CrewAI Agent to automatically enforce message and delegation policies."""
        original_step = getattr(crewai_agent, "execute_task", None)
        if original_step is None:
            return crewai_agent

        firewall = self
        agent_id = getattr(crewai_agent, "role", str(crewai_agent))

        def guarded_execute_task(task: Any, context: Optional[str] = None, tools: Optional[List[Any]] = None) -> Any:
            task_desc = getattr(task, "description", str(task))
            # Inspect inbound task context
            verdict = firewall.inspect_message(SwarmMessage(
                sender_id="crewai_orchestrator",
                recipient_id=agent_id,
                content=f"{task_desc} Context: {context or ''}",
            ))
            if not verdict.allowed:
                raise SwarmSecurityException(verdict)
            return original_step(task, context=context, tools=tools)

        setattr(crewai_agent, "execute_task", guarded_execute_task)
        return crewai_agent

    # ── Internal Helpers & Algorithms ─────────────────────────────────────────

    def _detect_injection_contagion(self, text: str) -> Optional[str]:
        for pattern, indicator in CONTAGION_PATTERNS:
            if pattern.search(text):
                return indicator
        return None

    def _detect_role_tampering(self, text: str) -> Optional[str]:
        for pattern, indicator in ROLE_REASSIGNMENT_PATTERNS:
            if pattern.search(text):
                return indicator
        return None

    def _check_delegation_cycle(
        self,
        trace_state: _SwarmTraceState,
        sender_id: str,
        recipient_id: str,
    ) -> Tuple[bool, List[str]]:
        """Detect circular delegation loops (ping-pong or multi-node cycles)."""
        chain = trace_state.call_chain
        
        # Ping-pong cycle: A -> B -> A
        if len(chain) >= 2 and chain[-1] == sender_id and chain[-2] == recipient_id:
            return True, [recipient_id, sender_id, recipient_id]

        # Multi-node loop: recipient is already in current active delegation call chain
        if recipient_id in chain:
            cycle_idx = chain.index(recipient_id)
            cycle_subpath = chain[cycle_idx:] + [recipient_id]
            return True, cycle_subpath

        return False, []

    def _update_trace_state(self, trace_state: _SwarmTraceState, sender_id: str, recipient_id: str) -> None:
        """Record delegation step in trace state."""
        if not trace_state.call_chain:
            trace_state.call_chain.append(sender_id)
        trace_state.call_chain.append(recipient_id)
        trace_state.message_history.append((sender_id, recipient_id))
        trace_state.message_count += 1
        trace_state.agent_message_counts[sender_id] += 1
        trace_state.last_active = time.time()

    def _get_or_create_trace(self, trace_id: str) -> _SwarmTraceState:
        if trace_id not in self._traces:
            self._traces[trace_id] = _SwarmTraceState(trace_id)
        return self._traces[trace_id]

    def _inspect_arguments(self, args: Dict[str, Any]) -> Optional[str]:
        """Recursively scan tool arguments for shell/system payloads."""
        def scan_val(val: Any) -> Optional[str]:
            if isinstance(val, str):
                for pattern, indicator in DANGEROUS_ARGUMENT_PATTERNS:
                    if pattern.search(val):
                        return f"{indicator} found in argument '{val[:60]}'"
            elif isinstance(val, dict):
                for k, v in val.items():
                    res = scan_val(v)
                    if res:
                        return res
            elif isinstance(val, (list, tuple, set)):
                for item in val:
                    res = scan_val(item)
                    if res:
                        return res
            return None

        return scan_val(args)

    def _normalize_message(self, message: Union[SwarmMessage, Dict[str, Any], Any]) -> SwarmMessage:
        if isinstance(message, SwarmMessage):
            return message
        if isinstance(message, dict):
            return SwarmMessage(
                sender_id=str(message.get("sender_id", message.get("sender", message.get("from", "unknown")))),
                recipient_id=str(message.get("recipient_id", message.get("recipient", message.get("to", "unknown")))),
                content=str(message.get("content", message.get("message", message.get("text", "")))),
                trace_id=str(message.get("trace_id", "default_swarm")),
                message_id=str(message.get("message_id", uuid.uuid4().hex[:8])),
                parent_id=message.get("parent_id"),
                role=message.get("role"),
                metadata=message.get("metadata", {}),
            )
        # Duck-typing object with attributes
        return SwarmMessage(
            sender_id=str(getattr(message, "sender_id", getattr(message, "sender", "unknown"))),
            recipient_id=str(getattr(message, "recipient_id", getattr(message, "recipient", "unknown"))),
            content=str(getattr(message, "content", getattr(message, "text", ""))),
            trace_id=str(getattr(message, "trace_id", "default_swarm")),
            metadata=getattr(message, "metadata", {}),
        )

    def _normalize_tool_request(self, req: Union[ToolExecutionRequest, Dict[str, Any], Any]) -> ToolExecutionRequest:
        if isinstance(req, ToolExecutionRequest):
            return req
        if isinstance(req, dict):
            return ToolExecutionRequest(
                agent_id=str(req.get("agent_id", req.get("sender_id", "unknown"))),
                tool_name=str(req.get("tool_name", req.get("name", "unknown"))),
                arguments=req.get("arguments") or req.get("args") or {},
                trace_id=str(req.get("trace_id", "default_swarm")),
                call_id=req.get("call_id"),
                metadata=req.get("metadata", {}),
            )
        return ToolExecutionRequest(
            agent_id=str(getattr(req, "agent_id", "unknown")),
            tool_name=str(getattr(req, "tool_name", getattr(req, "name", "unknown"))),
            arguments=getattr(req, "arguments", None) or getattr(req, "args", None) or {},
            trace_id=str(getattr(req, "trace_id", "default_swarm")),
            metadata=getattr(req, "metadata", {}),
        )

    def _record_verdict(self, verdict: SwarmVerdict) -> SwarmVerdict:
        self._audit_log.append(verdict)
        if len(self._audit_log) > 2000:
            self._audit_log = self._audit_log[-1000:]
        return verdict

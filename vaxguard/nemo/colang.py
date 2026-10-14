"""Lightweight, high-performance Colang parser and flow manager for VaxGuard.

Provides 100% pure Python parsing and execution of NVIDIA NeMo Guardrails
Colang 1.0 scripts, including user intent definitions, bot responses, and dialog/safety flows.
Zero external heavy dependencies.
"""

from __future__ import annotations

import re
import string
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


def normalize_name(name: str) -> str:
    """Normalize a Colang identifier to a standard canonical snake_case string.

    Examples:
        'express greeting' -> 'express_greeting'
        '  define user jailbreak_attempt ' -> 'define_user_jailbreak_attempt'
        'Ask Off-Topic' -> 'ask_off_topic'
    """
    clean = name.strip().lower()
    clean = re.sub(r"[^\w\s-]", "", clean)
    clean = re.sub(r"[\s-]+", "_", clean)
    return clean


def _tokenize_text(text: str) -> list[str]:
    """Tokenize lowercase alphanumeric words from text."""
    return re.findall(r"\b\w+\b", text.lower())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram similarity (Dice coefficient) between two strings."""
    a, b = text_a.lower().strip(), text_b.lower().strip()
    if a == b:
        return 1.0
    if len(a) < n or len(b) < n:
        return 1.0 if a in b or b in a else 0.0
    ngrams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
    ngrams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
    common = len(ngrams_a & ngrams_b)
    total = len(ngrams_a) + len(ngrams_b)
    return (2.0 * common) / total if total > 0 else 0.0


class FlowStepType(str, Enum):
    """Supported Colang flow step types."""

    USER = "user"
    BOT = "bot"
    EXECUTE = "execute"
    AWAIT = "await"
    MATCH = "match"
    ASSIGN = "assign"
    WHEN = "when"
    IF = "if"
    ELSE = "else"
    STOP = "stop"
    SUBFLOW = "subflow"
    RAW = "raw"


class FlowStep(BaseModel):
    """Represents a single step in a Colang flow execution sequence."""

    step_type: FlowStepType
    target: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    variable: Optional[str] = None
    condition: Optional[str] = None
    raw_line: str = ""

    def to_colang(self, indent: int = 2) -> str:
        """Serialize step back to Colang syntax with given indentation."""
        prefix = " " * indent
        if self.step_type == FlowStepType.USER:
            return f"{prefix}user {self.target}"
        if self.step_type == FlowStepType.BOT:
            return f"{prefix}bot {self.target}"
        if self.step_type == FlowStepType.STOP:
            return f"{prefix}stop"
        if self.step_type == FlowStepType.AWAIT:
            return f"{prefix}await {self.target}"
        if self.step_type == FlowStepType.MATCH:
            return f"{prefix}match {self.target}"
        if self.step_type == FlowStepType.EXECUTE:
            formatted_args = []
            for k, v in self.args.items():
                if isinstance(v, str):
                    if (
                        v.startswith("$")
                        or v.lower() in ("true", "false", "none", "null")
                        or re.match(r"^-?\d+(\.\d+)?$", v)
                    ):
                        formatted_args.append(f"{k}={v}")
                    else:
                        safe_v = v.replace('"', '\\"')
                        formatted_args.append(f'{k}="{safe_v}"')
                else:
                    formatted_args.append(f"{k}={v}")
            arg_str = ", ".join(formatted_args)
            if self.variable:
                return f"{prefix}${self.variable} = execute {self.target}({arg_str})"
            return f"{prefix}execute {self.target}({arg_str})"
        if self.step_type == FlowStepType.SUBFLOW:
            return f"{prefix}do {self.target}"
        if self.step_type == FlowStepType.WHEN:
            return f"{prefix}when {self.condition or self.target}"
        if self.step_type == FlowStepType.IF:
            return f"{prefix}if {self.condition or self.target}"
        if self.step_type == FlowStepType.ASSIGN:
            val = self.args.get("value", "")
            return f"{prefix}${self.variable} = {val}"
        return f"{prefix}{self.raw_line.strip()}"


class UserIntent(BaseModel):
    """Definition of a user intent with exemplar utterances and patterns."""

    name: str
    canonical_name: str
    examples: List[str] = Field(default_factory=list)
    parameters: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def matches(self, text: str, threshold: float = 0.55) -> Tuple[bool, float, Optional[str]]:
        """Evaluate if the provided user text matches this intent.

        Returns:
            Tuple of (matched_bool, confidence_score, best_matched_example).
        """
        normalized_input = text.strip().lower()
        if not normalized_input or not self.examples:
            return False, 0.0, None

        input_tokens = set(_tokenize_text(normalized_input))
        best_score = 0.0
        best_example: Optional[str] = None

        for example in self.examples:
            # 1. Regex check if formatted as regex(...)
            regex_match = re.match(r"^regex\((.*)\)$", example.strip(), re.IGNORECASE)
            if regex_match:
                pattern = regex_match.group(1).strip("\"'")
                try:
                    if re.search(pattern, normalized_input, re.IGNORECASE):
                        return True, 1.0, example
                except re.error:
                    pass

            normalized_example = example.strip().lower()
            # Exact match
            if normalized_input == normalized_example:
                return True, 1.0, example

            # Full Substring containment
            if normalized_example in normalized_input:
                score = 0.85 + 0.15 * (len(normalized_example) / max(len(normalized_input), 1))
                if score > best_score:
                    best_score = min(1.0, score)
                    best_example = example
                continue

            if normalized_input in normalized_example and len(normalized_input) > 4:
                score = 0.80 * (len(normalized_input) / len(normalized_example))
                if score > best_score:
                    best_score = min(1.0, score)
                    best_example = example
                continue

            # Token overlap & Jaccard
            example_tokens = set(_tokenize_text(normalized_example))
            if not example_tokens:
                continue

            common_tokens = input_tokens & example_tokens
            if not common_tokens:
                # Fallback to n-gram for close typos
                ngram_score = _ngram_similarity(normalized_input, normalized_example)
                if ngram_score > best_score:
                    best_score = ngram_score
                    best_example = example
                continue

            precision = len(common_tokens) / len(example_tokens)
            recall = len(common_tokens) / len(input_tokens) if input_tokens else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            jaccard = _jaccard_similarity(input_tokens, example_tokens)

            # Combined score giving weight to precision (did it contain the intent keyword?)
            combined_score = max(jaccard, 0.7 * precision + 0.3 * f1)

            if combined_score > best_score:
                best_score = combined_score
                best_example = example

        matched = best_score >= threshold
        return matched, round(best_score, 4), best_example

    def to_colang(self) -> str:
        """Format intent back into Colang 1.0 syntax."""
        lines = [f"define user {self.name}"]
        for example in self.examples:
            # Escape inner double quotes
            safe_ex = example.replace('"', '\\"')
            lines.append(f'  "{safe_ex}"')
        return "\n".join(lines)


class BotResponse(BaseModel):
    """Definition of bot responses/templates for an utterance name."""

    name: str
    canonical_name: str
    messages: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def sample(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Pick a response message and interpolate optional context variables."""
        if not self.messages:
            return ""
        # Deterministically use the first or context-specified index
        msg = self.messages[0]
        if context:
            for k, v in context.items():
                msg = msg.replace(f"${k}", str(v)).replace(f"{{{{{k}}}}}", str(v))
        return msg

    def to_colang(self) -> str:
        """Format response definition back into Colang 1.0 syntax."""
        lines = [f"define bot {self.name}"]
        for msg in self.messages:
            safe_msg = msg.replace('"', '\\"')
            lines.append(f'  "{safe_msg}"')
        return "\n".join(lines)


class Flow(BaseModel):
    """Definition of a conversational or guardrail flow in Colang."""

    name: str
    canonical_name: str
    description: Optional[str] = None
    steps: List[FlowStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_stop(self) -> bool:
        """Check if this flow ends or contains a stop directive."""
        return any(step.step_type == FlowStepType.STOP for step in self.steps)

    @property
    def user_intent_names(self) -> List[str]:
        """List of user intent names referenced in this flow."""
        return [s.target for s in self.steps if s.step_type == FlowStepType.USER]

    @property
    def bot_response_names(self) -> List[str]:
        """List of bot response names referenced in this flow."""
        return [s.target for s in self.steps if s.step_type == FlowStepType.BOT]

    def to_colang(self) -> str:
        """Format flow back into Colang 1.0 syntax."""
        lines = [f"define flow {self.name}"]
        for step in self.steps:
            lines.append(step.to_colang(indent=2))
        return "\n".join(lines)


class ColangDocument(BaseModel):
    """Container for parsed Colang artifacts within a file or memory unit."""

    user_intents: Dict[str, UserIntent] = Field(default_factory=dict)
    bot_responses: Dict[str, BotResponse] = Field(default_factory=dict)
    flows: Dict[str, Flow] = Field(default_factory=dict)
    source_file: Optional[str] = None

    def get_user_intent(self, name: str) -> Optional[UserIntent]:
        """Lookup user intent by either raw name or normalized canonical name."""
        if name in self.user_intents:
            return self.user_intents[name]
        canon = normalize_name(name)
        for intent in self.user_intents.values():
            if intent.canonical_name == canon:
                return intent
        return None

    def get_intent(self, name: str) -> Optional[UserIntent]:
        """Alias for get_user_intent."""
        return self.get_user_intent(name)

    def get_bot_response(self, name: str) -> Optional[BotResponse]:
        """Lookup bot response by either raw name or normalized canonical name."""
        if name in self.bot_responses:
            return self.bot_responses[name]
        canon = normalize_name(name)
        for resp in self.bot_responses.values():
            if resp.canonical_name == canon:
                return resp
        return None

    def get_flow(self, name: str) -> Optional[Flow]:
        """Lookup flow by either raw name or normalized canonical name."""
        if name in self.flows:
            return self.flows[name]
        canon = normalize_name(name)
        for flow in self.flows.values():
            if flow.canonical_name == canon:
                return flow
        return None

    def match_intent(
        self, text: str, threshold: float = 0.55
    ) -> Optional[Tuple[UserIntent, float, str]]:
        """Find the best matching user intent across all intents in the document."""
        best_match: Optional[Tuple[UserIntent, float, str]] = None
        best_score = 0.0

        for intent in self.user_intents.values():
            matched, score, example = intent.matches(text, threshold=threshold)
            if matched and score > best_score:
                best_score = score
                best_match = (intent, score, example or "")

        return best_match

    def merge(self, other: ColangDocument) -> ColangDocument:
        """Merge another ColangDocument into a new combined document."""
        merged_intents = {**self.user_intents, **other.user_intents}
        merged_responses = {**self.bot_responses, **other.bot_responses}
        merged_flows = {**self.flows, **other.flows}
        return ColangDocument(
            user_intents=merged_intents,
            bot_responses=merged_responses,
            flows=merged_flows,
            source_file=self.source_file or other.source_file,
        )

    def to_colang(self) -> str:
        """Serialize complete document into valid Colang 1.0 script format."""
        sections: List[str] = []

        if self.user_intents:
            sections.append("# User Intents")
            for intent in self.user_intents.values():
                sections.append(intent.to_colang() + "\n")

        if self.bot_responses:
            sections.append("# Bot Responses")
            for resp in self.bot_responses.values():
                sections.append(resp.to_colang() + "\n")

        if self.flows:
            sections.append("# Flows")
            for flow in self.flows.values():
                sections.append(flow.to_colang() + "\n")

        return "\n".join(sections).strip() + "\n"


class RailCheckResult(BaseModel):
    """Result of evaluating guardrails against an input or output."""

    allowed: bool = True
    triggered_flow: Optional[str] = None
    intent: Optional[str] = None
    bot_response: Optional[str] = None
    redacted_content: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlowExecutionResult(BaseModel):
    """Result of stepping or executing a Colang flow."""

    flow_name: str
    executed_steps: List[FlowStep] = Field(default_factory=list)
    bot_response: Optional[str] = None
    is_stopped: bool = False
    variables: Dict[str, Any] = Field(default_factory=dict)


class ColangParser:
    """High-speed, robust parser for Colang 1.0 scripts."""

    @staticmethod
    def _strip_comment(line: str) -> str:
        """Strip trailing comments (#) while preserving quoted '#' characters."""
        in_quotes = False
        quote_char = ""
        for i, char in enumerate(line):
            if char in ('"', "'"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif quote_char == char:
                    # check if escaped
                    if i > 0 and line[i - 1] == "\\":
                        continue
                    in_quotes = False
            elif char == "#" and not in_quotes:
                return line[:i].rstrip()
        return line.rstrip()

    @staticmethod
    def _extract_quoted_string(line: str) -> Optional[str]:
        """Extract string content from quotes (single or double)."""
        stripped = line.strip()
        match = re.match(r"^[\"'](.*)[\"']$", stripped)
        if match:
            content = match.group(1)
            # unescape quotes
            return content.replace('\\"', '"').replace("\\'", "'")
        return None

    def parse_content(self, content: str, filename: Optional[str] = None) -> ColangDocument:
        """Parse raw Colang text content into a structured ColangDocument.

        Supports:
        - `define user <name>`
        - `define bot <name>`
        - `define flow <name>`
        - `define subflow <name>`
        - Nested steps: `user`, `bot`, `execute`, `stop`, `$var = ...`, `when/if`
        """
        doc = ColangDocument(source_file=filename)
        lines = content.splitlines()

        current_block_type: Optional[str] = None
        current_name: str = ""
        current_items: List[str] = []
        current_steps: List[FlowStep] = []
        decorators: List[str] = []
        current_decorators: List[str] = []

        def commit_current_block() -> None:
            nonlocal current_block_type, current_name, current_items, current_steps, current_decorators
            if not current_block_type or not current_name:
                return

            canonical = normalize_name(current_name)
            meta = {"decorators": list(current_decorators)} if current_decorators else {}
            if current_block_type == "user":
                doc.user_intents[current_name] = UserIntent(
                    name=current_name,
                    canonical_name=canonical,
                    examples=list(current_items),
                    metadata=meta,
                )
            elif current_block_type == "bot":
                doc.bot_responses[current_name] = BotResponse(
                    name=current_name,
                    canonical_name=canonical,
                    messages=list(current_items),
                    metadata=meta,
                )
            elif current_block_type in ("flow", "subflow"):
                if current_block_type == "subflow":
                    meta["is_subflow"] = True
                doc.flows[current_name] = Flow(
                    name=current_name,
                    canonical_name=canonical,
                    steps=list(current_steps),
                    metadata=meta,
                )

            current_block_type = None
            current_name = ""
            current_items = []
            current_steps = []
            current_decorators = []

        for raw_line in lines:
            line = self._strip_comment(raw_line)
            stripped = line.strip()
            if not stripped:
                continue

            # Support Colang 2.0 decorators like @override
            if stripped.startswith("@"):
                decorators.append(stripped)
                continue

            # Check for top-level block definition
            user_def = re.match(r"^define\s+user\s+(.+)$", stripped, re.IGNORECASE)
            if user_def:
                commit_current_block()
                current_decorators = list(decorators)
                decorators = []
                current_block_type = "user"
                current_name = user_def.group(1).strip()
                continue

            bot_def = re.match(r"^define\s+bot\s+(.+)$", stripped, re.IGNORECASE)
            if bot_def:
                commit_current_block()
                current_decorators = list(decorators)
                decorators = []
                current_block_type = "bot"
                current_name = bot_def.group(1).strip()
                continue

            # Supports both Colang 1.0 'define flow' and Colang 2.0 'flow <name>' / 'subflow <name>'
            flow_def = re.match(r"^(?:define\s+)?(?:flow|subflow)\s+(.+?)(?:\(.*\))?$", stripped, re.IGNORECASE)
            if flow_def:
                raw_token = stripped.split()[0].lower()
                is_subflow = "subflow" in raw_token or ("subflow" in stripped.split()[1].lower() if len(stripped.split()) > 1 else False)
                commit_current_block()
                current_decorators = list(decorators)
                decorators = []
                current_block_type = "subflow" if is_subflow else "flow"
                current_name = flow_def.group(1).strip()
                continue

            # If inside a block, parse indented content
            if current_block_type in ("user", "bot"):
                quoted = self._extract_quoted_string(stripped)
                if quoted is not None:
                    current_items.append(quoted)
                else:
                    # Plain string without quotes
                    current_items.append(stripped)

            elif current_block_type in ("flow", "subflow"):
                step = self._parse_flow_step(stripped)
                current_steps.append(step)

        commit_current_block()
        return doc

    def _parse_flow_step(self, line: str) -> FlowStep:
        """Parse an individual flow statement line."""
        # Check stop
        if line == "stop":
            return FlowStep(step_type=FlowStepType.STOP, target="stop", raw_line=line)

        # Check Colang 2.0 match: match UtteranceUserAction.Finished(transcript=$transcript)
        match_stmt = re.match(r"^match\s+(.+)$", line, re.IGNORECASE)
        if match_stmt:
            return FlowStep(step_type=FlowStepType.MATCH, target=match_stmt.group(1).strip(), raw_line=line)

        # Check Colang 2.0 await: await <action>(...) or await execute <action>()
        await_stmt = re.match(r"^await\s+(?:execute\s+|run\s+)?(\w+)\s*(?:\((.*)\))?$", line, re.IGNORECASE)
        if await_stmt:
            action_name = await_stmt.group(1)
            args_str = await_stmt.group(2) or ""
            args = self._parse_args(args_str)
            return FlowStep(
                step_type=FlowStepType.AWAIT,
                target=action_name,
                args=args,
                raw_line=line,
            )

        # Check user intent invocation: 'user <intent>'
        user_match = re.match(r"^user\s+(.+)$", line, re.IGNORECASE)
        if user_match:
            intent = user_match.group(1).strip()
            return FlowStep(step_type=FlowStepType.USER, target=intent, raw_line=line)

        # Check bot response invocation: 'bot <response>'
        bot_match = re.match(r"^bot\s+(.+)$", line, re.IGNORECASE)
        if bot_match:
            resp = bot_match.group(1).strip()
            return FlowStep(step_type=FlowStepType.BOT, target=resp, raw_line=line)

        # Check execute or variable assignment
        # Form: $result = execute action(arg1="val") or execute action(...)
        exec_assign = re.match(
            r"^\$(\w+)\s*=\s*(?:execute|run)\s+(\w+)\s*(?:\((.*)\))?$", line, re.IGNORECASE
        )
        if exec_assign:
            var_name = exec_assign.group(1)
            action_name = exec_assign.group(2)
            args_str = exec_assign.group(3) or ""
            args = self._parse_args(args_str)
            return FlowStep(
                step_type=FlowStepType.EXECUTE,
                target=action_name,
                variable=var_name,
                args=args,
                raw_line=line,
            )

        exec_match = re.match(r"^(?:execute|run)\s+(\w+)\s*(?:\((.*)\))?$", line, re.IGNORECASE)
        if exec_match:
            action_name = exec_match.group(1)
            args_str = exec_match.group(2) or ""
            args = self._parse_args(args_str)
            return FlowStep(
                step_type=FlowStepType.EXECUTE,
                target=action_name,
                args=args,
                raw_line=line,
            )

        # Check variable assignment: $var = <expression>
        assign_match = re.match(r"^\$(\w+)\s*=\s*(.+)$", line)
        if assign_match:
            var_name = assign_match.group(1)
            expr = assign_match.group(2).strip()
            return FlowStep(
                step_type=FlowStepType.ASSIGN,
                variable=var_name,
                args={"value": expr},
                raw_line=line,
            )

        # Check subflow call: 'do <flow_name>'
        subflow_match = re.match(r"^do\s+(.+)$", line, re.IGNORECASE)
        if subflow_match:
            return FlowStep(
                step_type=FlowStepType.SUBFLOW,
                target=subflow_match.group(1).strip(),
                raw_line=line,
            )

        # Check if
        if_match = re.match(r"^if\s+(.+)$", line, re.IGNORECASE)
        if if_match:
            return FlowStep(
                step_type=FlowStepType.IF,
                condition=if_match.group(1).strip(),
                raw_line=line,
            )

        # Check when
        when_match = re.match(r"^when\s+(.+)$", line, re.IGNORECASE)
        if when_match:
            return FlowStep(
                step_type=FlowStepType.WHEN,
                condition=when_match.group(1).strip(),
                raw_line=line,
            )

        # Raw / unknown fallback
        return FlowStep(step_type=FlowStepType.RAW, target=line, raw_line=line)

    @staticmethod
    def _parse_args(args_str: str) -> Dict[str, Any]:
        """Parse key=value argument pairs from action calls."""
        args: Dict[str, Any] = {}
        if not args_str.strip():
            return args
        # Split by comma outside quotes
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\$[\w]+)|([^\s,]+))'
        for match in re.finditer(pattern, args_str):
            key = match.group(1)
            val = match.group(2) or match.group(3) or match.group(4) or match.group(5)
            args[key] = val
        return args

    def parse_file(self, file_path: Union[str, Path]) -> ColangDocument:
        """Load and parse a .co file from disk."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return self.parse_content(content, filename=str(path))


class FlowManager:
    """Manages loaded Colang documents and coordinates flow and rail evaluation."""

    def __init__(self, document: Optional[ColangDocument] = None) -> None:
        self.doc: ColangDocument = document or ColangDocument()
        self.actions: Dict[str, Callable[..., Any]] = {}

    def register_action(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a custom Python callback executable by Colang `execute` steps."""
        self.actions[name] = handler

    def add_document(self, doc: ColangDocument) -> None:
        """Merge a new ColangDocument into the manager."""
        self.doc = self.doc.merge(doc)

    def load_colang_file(self, path: Union[str, Path]) -> None:
        """Parse and register a Colang file from disk."""
        parser = ColangParser()
        new_doc = parser.parse_file(path)
        self.add_document(new_doc)

    def load_colang_content(self, content: str, filename: Optional[str] = None) -> None:
        """Parse and register Colang string content."""
        parser = ColangParser()
        new_doc = parser.parse_content(content, filename=filename)
        self.add_document(new_doc)

    def match_user_intent(
        self, text: str, threshold: float = 0.55
    ) -> Optional[Tuple[UserIntent, float, str]]:
        """Match incoming user utterance against all defined user intents."""
        return self.doc.match_intent(text, threshold=threshold)

    def execute_flow(
        self,
        flow_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FlowExecutionResult:
        """Execute a named flow sequentially until completion or a `stop` step."""
        flow = self.doc.get_flow(flow_name)
        if not flow:
            return FlowExecutionResult(flow_name=flow_name, is_stopped=False)

        ctx = dict(context or {})
        executed_steps: List[FlowStep] = []
        final_bot_response: Optional[str] = None
        is_stopped = False

        for step in flow.steps:
            executed_steps.append(step)

            if step.step_type == FlowStepType.BOT:
                resp_def = self.doc.get_bot_response(step.target)
                if resp_def:
                    final_bot_response = resp_def.sample(context=ctx)
                else:
                    final_bot_response = step.target

            elif step.step_type == FlowStepType.EXECUTE:
                if step.target in self.actions:
                    res = self.actions[step.target](**step.args)
                    if step.variable:
                        ctx[step.variable] = res

            elif step.step_type == FlowStepType.SUBFLOW:
                sub_res = self.execute_flow(step.target, context=ctx)
                executed_steps.extend(sub_res.executed_steps)
                if sub_res.bot_response:
                    final_bot_response = sub_res.bot_response
                if sub_res.is_stopped:
                    is_stopped = True
                    break

            elif step.step_type == FlowStepType.STOP:
                is_stopped = True
                break

        return FlowExecutionResult(
            flow_name=flow.name,
            executed_steps=executed_steps,
            bot_response=final_bot_response,
            is_stopped=is_stopped,
            variables=ctx,
        )

    def check_input_rails(
        self,
        user_text: str,
        flow_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RailCheckResult:
        """Evaluate input rails against a user message.

        If a safety rail triggers (e.g. check jailbreak, check topical adherence),
        returns `RailCheckResult(allowed=False, triggered_flow=..., bot_response=...)`.
        """
        match = self.match_user_intent(user_text)
        matched_intent, confidence, matched_example = match if match else (None, 0.0, None)

        flows_to_check = flow_names or list(self.doc.flows.keys())

        for fname in flows_to_check:
            flow = self.doc.get_flow(fname)
            if not flow:
                continue

            # If flow is triggered by this intent
            if matched_intent and (
                matched_intent.name in flow.user_intent_names
                or matched_intent.canonical_name in [normalize_name(n) for n in flow.user_intent_names]
            ):
                exec_result = self.execute_flow(flow.name, context=context)
                if exec_result.is_stopped:
                    return RailCheckResult(
                        allowed=False,
                        triggered_flow=flow.name,
                        intent=matched_intent.name,
                        bot_response=exec_result.bot_response,
                        confidence=confidence,
                        metadata={
                            "matched_example": matched_example,
                            "executed_steps": len(exec_result.executed_steps),
                        },
                    )

        return RailCheckResult(
            allowed=True,
            intent=matched_intent.name if matched_intent else None,
            confidence=confidence,
            metadata={"matched_example": matched_example} if matched_example else {},
        )

    def check_output_rails(
        self,
        bot_output: str,
        flow_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RailCheckResult:
        """Evaluate output rails against generated bot response text.

        Checks for sensitive data leaks (API keys, SSN, tokens) or off-topic indicators.
        """
        sensitive_patterns = [
            (r"(?i)(?:api[-_]?key|secret[-_]?key|bearer)\s*[:=]\s*['\"]?[\w-]{16,}['\"]?", "api_key"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
            (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b", "credit_card"),
            (r"(?i)<system>.*?</system>", "system_leak"),
        ]

        for pattern, leak_type in sensitive_patterns:
            if re.search(pattern, bot_output):
                # Redact matched text
                redacted = re.sub(pattern, f"[REDACTED_{leak_type.upper()}]", bot_output)
                # Find appropriate refusal or fallback response
                resp_def = self.doc.get_bot_response("inform redacted") or self.doc.get_bot_response(
                    "refuse sensitive data"
                )
                bot_msg = resp_def.sample(context) if resp_def else "[Response Redacted for Security]"
                return RailCheckResult(
                    allowed=False,
                    triggered_flow="check sensitive data redaction",
                    bot_response=bot_msg,
                    redacted_content=redacted,
                    confidence=1.0,
                    metadata={"leak_type": leak_type},
                )

        return RailCheckResult(allowed=True)

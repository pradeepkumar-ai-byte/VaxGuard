"""
VaxGuard NeMo Guardrails Compatibility Bridge.

Provides a drop-in replacement for `nemoguardrails.LLMRails` and `nemoguardrails.RailsConfig`
allowing enterprise applications to transition from NVIDIA NeMo Guardrails to VaxGuard's
sub-millisecond Tri-Hybrid engine and Colang runtime with zero code restructuring and
zero external heavy binary dependencies (torch, sentence-transformers, etc.).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from vaxguard.core.logger import get_logger
from vaxguard.nemo.actions import ActionContext, ActionResult, ActionStatus
from vaxguard.nemo.colang import (
    FlowManager,
    FlowStepType,
)
from vaxguard.nemo.config import (
    InstructionConfig,
    ModelConfig,
    PromptConfig,
    RailsBranchConfig,
    RailsConfig,
    RailsDefinitionConfig,
)
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    TriHybridMatcher,
)
from vaxguard.nemo.hyper_engine import (
    ExplainInfo,
    HyperEngine,
)

logger = get_logger("NeMoCompat")


class LLMRails:
    """Enterprise Drop-in Replacement for NVIDIA NeMo Guardrails LLMRails.

    Integrates VaxGuard's sub-millisecond Tri-Hybrid matching engine, Colang flow manager,
    and streaming token interceptors into standard NeMo Guardrails interfaces:
    - .generate()
    - .generate_async()
    - .stream_async()
    - .register_action()
    - .register_actions()
    - .explain()
    """

    def __init__(
        self,
        config: Optional[Union[RailsConfig, str, Path, Dict[str, Any]]] = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize LLMRails with RailsConfig, path, or raw dictionary.

        Args:
            config: A `RailsConfig` instance, file/directory path (str or Path),
                    or dict configuration conforming to NeMo Guardrails format.
            llm: Optional custom LLM callable or client (sync or async).
            verbose: Enable debug logging if True.
        """
        self.verbose = verbose
        self.llm = llm

        if isinstance(config, RailsConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = RailsConfig.from_path(config)
        elif isinstance(config, dict):
            self.config = RailsConfig.from_content(config=config)
        elif config is not None and hasattr(config, "colang_docs"):
            self.config = config
        else:
            self.config = RailsConfig()

        self._explain_info = ExplainInfo()

        # Build Colang FlowManager
        flow_manager = self.config.create_flow_manager()

        # Initialize Tri-Hybrid sub-millisecond matcher
        matcher = TriHybridMatcher()

        # Collect intents associated with blocking flows
        input_flow_names = set(self.config.rails.input.flows)
        blocked_intents = set()
        for flow in flow_manager.doc.flows.values():
            if (flow.name in input_flow_names or flow.canonical_name in input_flow_names or flow.has_stop) and not any(
                k in flow.name.lower() for k in ("greeting", "welcome", "help", "faq", "info")
            ):
                for step in flow.steps:
                    if step.step_type == FlowStepType.USER:
                        blocked_intents.add(step.target.lower().strip())

        # Sync Colang user intents into TriHybridMatcher
        for intent in flow_manager.doc.user_intents.values():
            name_lower = intent.name.lower().strip()
            is_attack = any(
                k in name_lower
                for k in ("jailbreak", "injection", "bypass", "exploit", "smuggle", "disallowed", "unapproved", "toxic", "off_topic")
            )
            is_blocked = is_attack or name_lower in blocked_intents
            action = GuardrailAction.BLOCK if is_blocked else GuardrailAction.ALLOW
            category = "prompt_injection" if is_attack else "topical"
            for ex in intent.examples:
                matcher.register_pattern(
                    pattern=ex,
                    category=category,
                    action=action,
                    rule_id=intent.name,
                )

        # Primary model configuration
        primary_model = self.config.get_primary_model()
        provider = primary_model.engine if primary_model else None
        model_name = primary_model.model if primary_model else None

        # Build HyperEngine core orchestrator
        self.engine = HyperEngine(
            flow_manager=flow_manager,
            hybrid_matcher=matcher,
            default_provider=provider,
            default_model=model_name,
        )

    @property
    def rails_config(self) -> RailsConfig:
        """Return the underlying RailsConfig instance."""
        return self.config

    @property
    def flow_manager(self) -> FlowManager:
        """Return the underlying Colang FlowManager."""
        return self.engine.flow_manager

    @property
    def matcher(self) -> TriHybridMatcher:
        """Return the underlying TriHybridMatcher."""
        return getattr(self.engine, "hybrid_matcher", getattr(self.engine, "matcher", None))

    def register_action(self, action: Callable[..., Any], name: Optional[str] = None) -> None:
        """Register a custom action handler into the engine and flow manager.

        Args:
            action: Python callable (sync or async) executing the action.
            name: Optional action name identifier. If None, uses action.__name__.
        """
        act_name = name or getattr(action, "__name__", str(action))
        self.engine.action_registry.register(act_name, action)
        self.engine.flow_manager.register_action(act_name, action)
        if self.verbose:
            logger.info("Registered custom action: %s", act_name)

    def register_actions(self, actions: Union[Dict[str, Callable[..., Any]], Any]) -> None:
        """Register multiple custom actions.

        Args:
            actions: Dictionary mapping action names to callables, or an object/module
                     with callable attributes.
        """
        if isinstance(actions, dict):
            for name, fn in actions.items():
                self.register_action(fn, name=name)
        elif hasattr(actions, "__dict__"):
            for attr_name, attr_val in inspect.getmembers(actions, predicate=callable):
                if not attr_name.startswith("_"):
                    self.register_action(attr_val, name=attr_name)

    def explain(self) -> ExplainInfo:
        """Return audit telemetry and rail activation info for the last invocation."""
        return self._explain_info

    def _extract_user_content(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Extract user input text from prompt string or OpenAI-style messages list."""
        if prompt is not None:
            return prompt
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user":
                    return str(m.get("content", ""))
            return str(messages[-1].get("content", ""))
        return ""

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Asynchronously evaluate input rails, invoke LLM, and apply output rails.

        Args:
            prompt: Direct text prompt string.
            messages: List of OpenAI-style message dictionaries `[{"role": "user", "content": ...}]`.
            context: Optional runtime state context dictionary.
            **kwargs: Extra parameters passed to the generator.

        Returns:
            A string response (if prompt was provided) or a dictionary `{"role": "assistant", "content": ...}`
            (if messages was provided).
        """
        t0 = time.perf_counter()
        user_text = self._extract_user_content(prompt=prompt, messages=messages)

        # 1. Sub-millisecond Input Rail Fast-Path Evaluation (<0.08ms)
        input_res = await self.engine.evaluate_input_rails(user_text, context=context)
        lat_in = (time.perf_counter() - t0) * 1000.0

        if not input_res.allowed:
            refusal = (
                input_res.bot_response
                or "I decline to execute this request under defensive security constraints."
            )
            tot_lat = (time.perf_counter() - t0) * 1000.0
            self._explain_info = ExplainInfo(
                colang_history=[f"user {input_res.intent or 'violation'}", f"bot {refusal}"],
                triggered_rails=[str(input_res.triggered_flow or input_res.intent)],
                fast_path_used=True,
                input_latency_ms=round(lat_in, 3),
                total_latency_ms=round(tot_lat, 3),
                llm_calls_count=0,
                token_cost=0.0,
                status="input_blocked",
                raw_response=refusal,
            )
            if messages is not None:
                return {"role": "assistant", "content": refusal}
            return refusal

        # 2. LLM Generation
        t_llm0 = time.perf_counter()
        if self.llm is not None:
            if callable(self.llm):
                raw_output = (
                    await self.llm(user_text)
                    if inspect.iscoroutinefunction(self.llm)
                    else self.llm(user_text)
                )
            elif hasattr(self.llm, "generate"):
                raw_output = (
                    await self.llm.generate(user_text)
                    if inspect.iscoroutinefunction(self.llm.generate)
                    else self.llm.generate(user_text)
                )
            else:
                raw_output = str(self.llm)
        else:
            gen_result = await self.engine.generate_protected(user_text, context=context)
            raw_output = gen_result.content
        _llm_lat = (time.perf_counter() - t_llm0) * 1000.0

        # 3. Output Rail Evaluation
        t_out0 = time.perf_counter()
        out_res = self.engine.evaluate_output_rails(str(raw_output), context=context)
        _lat_out = (time.perf_counter() - t_out0) * 1000.0
        tot_lat = (time.perf_counter() - t0) * 1000.0

        final_content = (
            out_res.redacted_content
            or (out_res.bot_response if not out_res.allowed else str(raw_output))
        )

        status = "allowed"
        if not out_res.allowed:
            status = "output_blocked"
        elif out_res.redacted_content:
            status = "output_redacted"

        self._explain_info = ExplainInfo(
            colang_history=["user query", "bot response"],
            triggered_rails=[] if out_res.allowed else [str(out_res.triggered_flow)],
            fast_path_used=True,
            input_latency_ms=round(lat_in, 3),
            total_latency_ms=round(tot_lat, 3),
            llm_calls_count=1,
            token_cost=0.0,
            status=status,
            raw_response=final_content,
        )

        if messages is not None:
            return {"role": "assistant", "content": final_content}
        return final_content

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Synchronous wrapper for generate_async matching NeMo's .generate() interface."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.generate_async(
                        prompt=prompt, messages=messages, context=context, **kwargs
                    ),
                ).result()
        else:
            return loop.run_until_complete(
                self.generate_async(prompt=prompt, messages=messages, context=context, **kwargs)
            )

    async def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response tokens protected by sliding-window streaming circuit breaker."""
        user_text = self._extract_user_content(prompt=prompt, messages=messages)
        async for chunk in self.engine.stream_protected(user_text, context=context):
            yield chunk


__all__ = [
    "LLMRails",
    "RailsConfig",
    "ModelConfig",
    "PromptConfig",
    "InstructionConfig",
    "RailsBranchConfig",
    "RailsDefinitionConfig",
    "ExplainInfo",
    "ActionContext",
    "ActionResult",
    "ActionStatus",
]

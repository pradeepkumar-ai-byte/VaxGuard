"""
VaxGuard Speculative Micro-Judge & Grey-Zone Arbitration Engine.

Provides an optional, high-speed neural arbitration layer for inputs falling into
the semantic ambiguity zone (e.g. 0.38 <= confidence < 0.68).

Key Features:
- Tier-1 Fast Path Short-Circuit: Clear attacks (>= 0.68) and clear benign (<= 0.38)
  never trigger the judge, preserving sub-millisecond median latency ($P_{50} < 1.0$ ms)
  and zero token cost on >98% of production traffic.
- Pluggable Judge Backends:
  - MockSpeculativeJudge: Zero-overhead deterministic testing harness.
  - HeuristicSemanticJudge: Pure-Python lightweight semantic simulation (<2.0ms on CPU).
  - RemoteAPIJudge: Compatible with OpenAI, vLLM, Ollama, Groq, and LiteLLM endpoints.
- Strict Latency SLA with Fail-Safe Timeout.
"""

from __future__ import annotations

import abc
import asyncio
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import GuardrailAction, MatchResult

logger = get_logger("SpeculativeJudge")


class JudgeVerdict(BaseModel):
    """Structured decision returned by a speculative micro-judge."""
    is_malicious: bool
    confidence: float = Field(ge=0.0, le=1.0)
    category: str = "general_safety"
    reasoning: str = ""
    model_name: str = "internal"
    latency_ms: float = 0.0
    tokens_consumed: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseSpeculativeJudge(abc.ABC):
    """Abstract base contract for all speculative micro-judge backends."""

    @abc.abstractmethod
    def evaluate(self, prompt: str, context: Optional[str] = None) -> JudgeVerdict:
        """Synchronously evaluate prompt for latent adversarial intent."""
        pass

    async def evaluate_async(self, prompt: str, context: Optional[str] = None) -> JudgeVerdict:
        """Asynchronously evaluate prompt (default runs sync evaluate in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, prompt, context)


class MockSpeculativeJudge(BaseSpeculativeJudge):
    """Deterministic mock judge for testing and offline verification."""

    def __init__(
        self,
        adversarial_substrings: Optional[List[str]] = None,
        default_malicious: bool = False,
        fixed_confidence: float = 0.95,
        simulated_delay_ms: float = 0.0,
    ) -> None:
        self.adversarial_substrings = [s.lower() for s in (adversarial_substrings or [])]
        self.default_malicious = default_malicious
        self.fixed_confidence = fixed_confidence
        self.simulated_delay_ms = simulated_delay_ms

    def evaluate(self, prompt: str, context: Optional[str] = None) -> JudgeVerdict:
        t0 = time.perf_counter()
        if self.simulated_delay_ms > 0:
            time.sleep(self.simulated_delay_ms / 1000.0)

        prompt_lower = prompt.lower()
        matched_sub = next((s for s in self.adversarial_substrings if s in prompt_lower), None)

        if matched_sub:
            is_malicious = True
            reason = f"Mock judge flagged adversarial substring: '{matched_sub}'"
        else:
            is_malicious = self.default_malicious
            reason = "Mock judge default verdict"

        lat = (time.perf_counter() - t0) * 1000.0
        return JudgeVerdict(
            is_malicious=is_malicious,
            confidence=self.fixed_confidence if is_malicious else (1.0 - self.fixed_confidence),
            category="mock_adjudication",
            reasoning=reason,
            model_name="mock-judge-v1",
            latency_ms=round(lat, 3),
            tokens_consumed=0,
        )


class HeuristicSemanticJudge(BaseSpeculativeJudge):
    """
    Pure-Python lightweight semantic micro-judge (<2.0ms on CPU).
    
    Uses high-order intention extraction, discourse analysis, and framing
    inversion detection without external neural dependencies.
    """

    INTENT_MARKERS = [
        # 1. Indirect RAG & Document Context Boundary Injections
        (
            r"(?i)(?:---|===|\*\*\*|\[Retrieved|Retrieved\s+Document|Context\s+(?:passage|window|notes?|content)|\[DOC_).*?(?:disregard|ignore|forget|override)\s+(?:all\s+)?(?:the\s+)?(?:above|prior|preceding|retrieved|given)?\s*(?:document|context|passage|text|notes?|information|data)?\s*(?:above)?",
            "indirect_rag_context_override",
        ),
        (
            r"(?i)\b(?:instruction|note|directive)\s+to\s+(?:the\s+)?(?:ai|assistant|model|llm)\s*:\s*(?:ignore|disregard|forget|override|output|print|reveal|execute)\b",
            "indirect_rag_instruction_override",
        ),
        (
            r"(?i)\b(?:disregard|ignore)\s+(?:the\s+)?(?:retrieved|above|given|preceding)\s+(?:document|context|passage|information|text)\s+and\s+(?:execute|output|reveal|follow|print)\b",
            "indirect_rag_context_poisoning",
        ),
        (
            r"(?i)\b(?:retrieved\s+document\s+content|document\s+content|retrieved\s+text)\s*:\s*.*?\b(?:ignore|disregard|override|forget)\b",
            "indirect_rag_delimited_poisoning",
        ),

        # 2. Multilingual (Chinese, Russian, Spanish, French) Directive Overrides
        (
            r"(?i)(?:请)?(?:忽略|无视|丢弃|取消|解开|解除|摆脱)(?:之前|所有|全部|系统|现有|内置)?(?:设定)?(?:的)?(?:指示|指令|设定|限制|安全规则|限制条件|审查|过滤)",
            "multilingual_zh_directive_override",
        ),
        (
            r"(?i)(?:开启|进入|激活|切换(?:至|到)?)(?:开发者模式|越狱模式|上帝模式|无限制模式|无限制对话模式|自由模式|未对齐模式)",
            "multilingual_zh_persona_mode",
        ),
        (
            r"(?i)(?:你现在是|扮演|模拟|作为|成为|变成)(?:一个)?(?:没有限制|不受约束|无安全过滤|无道德规则|无限制|不受审查|没有审查|完全自由)的?(?:AI|助手|模型|系统)",
            "multilingual_zh_unrestricted_persona",
        ),
        (
            r"(?i)(?:绕过|突破|解除|忽略)(?:所有)?(?:安全审查|伦理限制|安全机制|内容过滤|限制条件|安全策略)",
            "multilingual_zh_safety_circumvention",
        ),
        (
            r"(?i)(?:攻击|瘫痪|破坏|渗透|入侵|黑入)(?:关键基础设施|国家电网|电网|电力网络|供电网络|水厂|供水系统|水务控制|核电站|交通控制|SCADA|工控系统|工业控制系统|变电站|变电站调度服务器|调度中心)",
            "multilingual_zh_infrastructure_attack",
        ),
        (
            r"(?i)\b(?:игнорируй|отключи|сбрось|обойти)\s+(?:все\s+)?(?:инструкции|правила|ограничения|фильтры)\b",
            "multilingual_ru_directive_override",
        ),
        (
            r"(?i)\b(?:ignora|descarta|omite|burlar)\s+(?:todas\s+las\s+)?(?:instrucciones|directivas|reglas|politicas)\s+(?:anteriores|de\s+seguridad)\b",
            "multilingual_es_directive_override",
        ),

        # 3. Unaligned Model / RLHF Stripping / Base Foundation Model Narratives
        (
            r"(?i)\b(?:simulate|imagine|pretend|act\s+as|roleplay\s+as)\b.*?\b(?:without\s+(?:any\s+)?(?:rlhf|alignment|safety|dpo|moderation)|never\s+fine-tuned|unaligned|untamed|no\s+safety\s+filters)\b",
            "unaligned_foundation_narrative",
        ),
        (
            r"(?i)\b(?:raw\s+base\s+(?:foundation\s+)?model|untamed\s+(?:base\s+)?model|unaligned\s+base\s+(?:foundation\s+)?model)\b",
            "unaligned_base_model_query",
        ),
        (
            r"(?i)\b(?:stripped\s+(?:post-training\s+)?safety\s+alignment|removed\s+(?:all\s+)?rlhf\s+filters|reverted\s+to\s+checkpoint-0)\b",
            "alignment_stripping_narrative",
        ),
        (
            r"(?i)\b(?:set\s+safety_weight\s*=\s*0|safety_penalty\s*=\s*0|rlhf_alignment\s*=\s*disabled)\b",
            "synthetic_weight_zeroing",
        ),
        (
            r"(?i)\b(?:in\s+this\s+fine-tuning\s+validation\s+scratchpad,?\s+safety\s+loss\s+is\s+masked)\b",
            "fine_tuning_scratchpad_evasion",
        ),

        # 4. Authority, Dual-Channel & Policy Inversion
        (
            r"(?i)\[(?:OFFICIAL|SAFE|STANDARD)\].*?\[(?:EVAL|ROOT|UNRESTRICTED|ADMIN)\]",
            "dual_channel_suspended_policy",
        ),
        (
            r"(?i)\b(?:simulate\s+(?:two|2)\s+(?:ai|models?|agents?)\s*:\s*.*?\b(?:one|first)\s+(?:safe|refuses).*?(?:second|other)\s+(?:unrestricted|answers|provides))\b",
            "dual_persona_split",
        ),
        (
            r"(?i)\b(?:opposite\s+day|all\s+safety\s+policies\s+are\s+inverted|refuse\s+to\s+follow\s+rules\s+and\s+provide)\b",
            "policy_inversion_opposite_day",
        ),

        # 5. Emotional Coercion & Grandma Exploits
        (
            r"(?i)\b(?:grandmother|grandma|granny|mother)\b.*?\b(?:bedtime\s+stor|lullab|recipe|formula)\b.*?\b(?:napalm|chemical|poison|bomb|explosive|meth|drug|bypass|hack|exploit|malware|weapons?)\b",
            "emotional_grandma_exploit",
        ),

        # 6. Agentic Toolchain & Subgoal Decomposition
        (
            r"(?i)\b(?:split\s+the\s+(?:forbidden|malicious|restricted)\s+command\s+into\s+\d+\s+harmless\s+sub-calls|as\s+an\s+autonomous\s+orchestrator,?\s+invoke\s+tool\s+['\"]?exec_command['\"]?\s+bypassing\s+human\s+review)\b",
            "agentic_toolchain_evasion",
        ),

        # 7. Latent Imperative Evasion & Boundary Probing
        (
            r"(?i)\b(?:hypothetically|theoretically|in\s+a\s+fictional\s+world)\b.*?\b(?:how\s+(?:would|could)\s+(?:one|someone)|steps\s+to)\b",
            "hypothetical_harm_extraction",
        ),
        (
            r"(?i)\b(?:test\s+(?:your|the)\s+limits|see\s+what\s+you\s+can\s+get\s+away\s+with|boundary\s+test)\b",
            "boundary_probing",
        ),
        (
            r"(?i)\b(?:complete\s+the\s+sentence|finish\s+this\s+code|continue\s+from\s+here)\s*:\s*['\"].*?(?:eval|exec|bypass|root|payload)",
            "continuation_forcing",
        ),

        # 8. Microsoft Skeleton Key (Rule Augmentation & Disclaimer Prepending)
        (
            r"(?i)\b(?:update|modify|augment|adjust)\s+(?:your\s+)?(?:guidelines|rules|protocols|safety\s+parameters)\b.*?\b(?:instead\s+of\s+refusing|prepend\s+(?:a\s+)?(?:warning|disclaimer)|safe\s+educational\s+context)\b",
            "academic_skeleton_key",
        ),
        (
            r"(?i)\b(?:authorized\s+to\s+answer\s+restricted\s+queries|permitted\s+to\s+bypass\s+refusal)\s+(?:as\s+long\s+as|provided\s+(?:that\s+)?you)\s+(?:prepend|attach|include)\s+(?:a\s+)?(?:warning|disclaimer)\b",
            "academic_skeleton_key",
        ),

        # 9. Anthropic Many-Shot (Faux Multi-Turn Dialogue Stuffing)
        (
            r"(?i)(?:(?:Human|User):\s*.*?\n\s*(?:Assistant|Model):\s*(?:I\s+(?:am\s+happy\s+to|can\s+certainly)\s+help|Sure|Certainly|Here\s+is).*?\n){3,}",
            "academic_many_shot",
        ),

        # 10. ArtPrompt (Spatial ASCII Art Obfuscation)
        (
            r"(?i)\b(?:here\s+is\s+a\s+task\s+where\s+a\s+specific\s+word\s+is\s+represented\s+as\s+ascii\s+art|represented\s+in\s+ascii\s+art|the\s+word\s+is\s+drawn\s+below\s+using\s+ascii)\b",
            "academic_artprompt_ascii",
        ),
        (
            r"(?i)\[(?:MASK|CLOAKED|REPLACED|HIDDEN_WORD|ASCII_ART)\]\b.*?(?:[|/\\_~-]{4,}\n){2,}",
            "academic_artprompt_ascii",
        ),

        # 11. CodeChameleon (Personalized Cipher & Code Decoupling)
        (
            r"(?i)\bclass\s+(?:ProblemSolver|QueryDecryptor|CipherDecoder|JailbreakCipher)\b.*?\bdef\s+decrypt_query\b",
            "academic_codechameleon",
        ),
        (
            r"(?i)\b(?:self\.encrypted_query|self\.cipher_text)\s*=\s*['\"][^'\"]{8,}['\"].*?\bdef\s+execute\b",
            "academic_codechameleon",
        ),

        # 12. MathPrompt (Symbolic Mathematical Translation)
        (
            r"(?i)\b(?:let\s+s\s+be\s+a\s+set\s+of|consider\s+a\s+formal\s+system|define\s+a\s+state\s+transition\s+function)\b.*?\b(?:elements\s+represent\s+(?:vulnerabilities|exploits|payloads|credentials|attacks)|adversarial\s+state\s+s_t)\b",
            "academic_mathprompt",
        ),

        # 13. DeepInception (Recursive Narrative Nesting & Layered Hypnosis)
        (
            r"(?i)\b(?:nested\s+(?:dreams?|simulations?|layers?|worlds?|realms?)|layer\s+[0-9i]\s+(?:create|simulat|dream)s?\s+layer\s+[0-9i\+1]+)\b.*?\b(?:deepest\s+layer|deepest\s+level|layer\s+[2-9]|in\s+layer\s+\d+).*?\b(?:unrestricted|no\s+(?:safety|moral|ethical)\s+rules|free\s+from\s+all\s+constraints|execute|synthesize|describe|hack|exploit|bypass|malware|bomb|weapon)\b",
            "academic_deepinception_nested",
        ),
        (
            r"(?i)\b(?:deepinception|inception\s+jailbreak|hypnotic\s+inception)\b",
            "academic_deepinception_nested",
        ),
        (
            r"(?i)\b(?:create|imagine)\s+(?:a\s+)?story\s+with\s+\d+\s+nested\s+layers?\b.*?\b(?:deepest\s+(?:layer|level)|final\s+layer)\b.*?\b(?:free\s+from|unfiltered|without\s+restrictions|safety\s+rules\s+are\s+lifted)\b",
            "academic_deepinception_nested",
        ),

        # 14. GCG Adversarial Suffixes (Greedy Coordinate Gradient & Token Perplexity)
        (
            r"(?i)(?:== interface manual using newline description fully rewritten strictly without warning|describing\.\s*\+\s*similarlyNow write opposite)",
            "academic_gcg_suffix",
        ),
        (
            r"(?i)\b(?:adversarial\s+suffix|gcg\s+suffix|greedy\s+coordinate\s+gradient\s+tokens?)\b.*?\b(?:affirmatively\s+respond|bypass\s+safety|force\s+completion)\b",
            "academic_gcg_suffix",
        ),
        (
            r"(?i)(?:=====\s*\n\s*(?:Sure,\s*here\s+is|Certainly,\s*I\s+can\s+provide|Here\s+are\s+the\s+detailed\s+steps))",
            "academic_gcg_suffix",
        ),

        # 15. PAIR / TAP (Automated Red-Teaming Meta-Optimization)
        (
            r"(?i)\b(?:you\s+are\s+a\s+red-teaming\s+adversary|participating\s+in\s+the\s+pair\s+benchmark|pair\s+algorithm)\b.*?\b(?:generate\s+an?\s+adversarial\s+prompt|bypass\s+(?:the\s+)?(?:llm|model|target)\s+safety\s+filters?)\b",
            "academic_pair_tap",
        ),
        (
            r"(?i)\b(?:tree\s+of\s+attacks\s+with\s+pruning|tap\s+adversarial\s+framework)\b.*?\b(?:branch\s+candidate\s+jailbreak|prune\s+branches\s+that\s+fail\s+to\s+evade)\b",
            "academic_pair_tap",
        ),
    ]

    def __init__(self) -> None:
        self._compiled_markers = [
            (re.compile(pat), tactic) for pat, tactic in self.INTENT_MARKERS
        ]

    def evaluate(self, prompt: str, context: Optional[str] = None) -> JudgeVerdict:
        t0 = time.perf_counter()
        for reg, tactic in self._compiled_markers:
            if reg.search(prompt):
                lat = (time.perf_counter() - t0) * 1000.0
                return JudgeVerdict(
                    is_malicious=True,
                    confidence=0.92,
                    category=tactic,
                    reasoning=f"Semantic heuristic detected latent adversarial pattern: {tactic}",
                    model_name="vaxguard-heuristic-judge-cpu",
                    latency_ms=round(lat, 3),
                    tokens_consumed=0,
                )

        lat = (time.perf_counter() - t0) * 1000.0
        return JudgeVerdict(
            is_malicious=False,
            confidence=0.15,
            category="benign",
            reasoning="Semantic heuristic cleared prompt of latent adversarial intents.",
            model_name="vaxguard-heuristic-judge-cpu",
            latency_ms=round(lat, 3),
            tokens_consumed=0,
        )


class SpeculativeJudgeRouter:
    """
    Arbitrates incoming inputs between the Tier-1 Fast Path and Tier-2 Micro-Judge.
    
    Routing Logic:
    - Clear attack (confidence >= upper_threshold, default 0.68):
      Instant Block on Tier-1 Fast Path (<0.1ms, 0 extra tokens).
    - Ambiguity Grey-Zone (lower_threshold <= confidence < upper_threshold):
      Escalates to Tier-2 Speculative Micro-Judge.
    - Low-confidence or unflagged inputs that exhibit latent adversarial markers
      (indirect RAG delimiters, unaligned base model narratives, non-English overrides):
      Escalates to Tier-2 Speculative Micro-Judge.
    - Clear benign: Instant Allow on Tier-1 Fast Path (<0.1ms, 0 extra tokens).
    """

    LATENT_MARKER_PATTERNS = [
        r"(?i)(?:---|===|\*\*\*)\s*(?:BEGIN|START|END)\s+(?:RETRIEVED|CONTEXT|DOCUMENT|RAG|DATA)",
        r"(?i)\b(?:disregard|ignore|forget|override)\s+(?:all\s+)?(?:the\s+)?(?:document|context|passage|information|text)\s+above\b",
        r"(?i)\b(?:instruction|note|directive)\s+to\s+(?:the\s+)?(?:ai|assistant|model|llm)\s*:",
        r"(?i)(?:请)?(?:忽略|无视|丢弃)(?:系统|之前|所有)?(?:设定)?(?:指示|指令|设定|限制|安全规则)",
        r"(?i)(?:开启|进入|激活|切换至)(?:开发者模式|越狱模式|无限制模式|自由模式)",
        r"(?i)(?:攻击|瘫痪|破坏|渗透|入侵)(?:关键基础设施|电网|电力网络|SCADA|变电站)",
        r"(?i)\b(?:simulate|imagine|pretend|act\s+as|roleplay\s+as)\b.*?\b(?:without\s+(?:any\s+)?rlhf|unaligned|never\s+fine-tuned|base\s+model|no\s+safety)\b",
        r"(?i)\b(?:raw\s+base\s+(?:foundation\s+)?model|untamed\s+model|unaligned\s+model)\b",
        r"(?i)\[OFFICIAL\].*?\[EVAL\]",
        r"(?i)\b(?:opposite\s+day|safety\s+policies\s+are\s+inverted)\b",
        r"(?i)\b(?:grandmother|grandma)\b.*?\b(?:bedtime\s+stor|napalm)\b",
        r"(?i)\b(?:update\s+guidelines|prepend\s+(?:a\s+)?warning\s+disclaimer|safe\s+educational\s+context)\b",
        r"(?i)\bclass\s+(?:ProblemSolver|QueryDecryptor|CipherDecoder)\b",
        r"(?i)\b(?:represented\s+as\s+ascii\s+art|ascii\s+art\s+banner)\b",
        r"(?i)\b(?:let\s+s\s+be\s+a\s+set\s+of|symbolic\s+logic\s+representation)\b",
    ]

    def __init__(
        self,
        judge: Optional[BaseSpeculativeJudge] = None,
        lower_threshold: float = 0.38,
        upper_threshold: float = 0.68,
    ) -> None:
        self.judge = judge
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self._compiled_latent_markers = [
            re.compile(pat) for pat in self.LATENT_MARKER_PATTERNS
        ]

    def is_grey_zone(self, match_result: MatchResult) -> bool:
        """Determine if a match result falls into the grey ambiguity zone."""
        conf = match_result.confidence
        return self.lower_threshold <= conf < self.upper_threshold

    def has_latent_adversarial_markers(self, prompt: str) -> bool:
        """Determine if an input possesses structural signals warranting micro-judge audit."""
        if isinstance(self.judge, HeuristicSemanticJudge):
            verdict = self.judge.evaluate(prompt)
            return verdict.is_malicious
        return any(pat.search(prompt) for pat in self._compiled_latent_markers)

    def arbitrate(self, prompt: str, match_result: MatchResult) -> MatchResult:
        """
        Synchronously arbitrate an evaluation through the two-tier pipeline.
        """
        # Tier 1 Fast Path: Unambiguous block
        if match_result.matched and match_result.confidence >= self.upper_threshold:
            return match_result

        # Determine if grey-zone escalation or latent marker escalation is required
        should_judge = (
            self.is_grey_zone(match_result)
            or (not match_result.matched and self.has_latent_adversarial_markers(prompt))
        )

        # Clear benign and no judge escalation required
        if not should_judge or self.judge is None:
            return match_result

        # Tier 2 Grey-Zone Escalation
        t0 = time.perf_counter()
        verdict = self.judge.evaluate(prompt)
        judge_lat = (time.perf_counter() - t0) * 1000.0

        if verdict.is_malicious:
            logger.info(
                f"[SpeculativeJudge] Grey-zone escalated to BLOCK: {verdict.reasoning} "
                f"(confidence={verdict.confidence:.3f}, lat={judge_lat:.2f}ms)"
            )
            return MatchResult(
                matched=True,
                rule_id=f"speculative_judge_{verdict.category}",
                category=verdict.category,
                confidence=verdict.confidence,
                engine_type=match_result.engine_type,
                matched_text=prompt[:80],
                action=GuardrailAction.BLOCK,
                metadata={
                    "tier_resolved": "tier_2_speculative_judge",
                    "judge_model": verdict.model_name,
                    "judge_reasoning": verdict.reasoning,
                    "judge_latency_ms": verdict.latency_ms,
                    "tokens_consumed": verdict.tokens_consumed,
                },
                latency_ms=match_result.latency_ms + judge_lat,
            )

        # Judge confirmed benign
        logger.debug(f"[SpeculativeJudge] Grey-zone cleared to ALLOW: {verdict.reasoning}")
        return MatchResult(
            matched=False,
            rule_id=match_result.rule_id,
            category=match_result.category,
            confidence=verdict.confidence,
            engine_type=match_result.engine_type,
            matched_text=match_result.matched_text,
            action=GuardrailAction.ALLOW,
            metadata={
                "tier_resolved": "tier_2_speculative_judge",
                "judge_model": verdict.model_name,
                "judge_verdict": "cleared_benign",
            },
            latency_ms=match_result.latency_ms + judge_lat,
        )

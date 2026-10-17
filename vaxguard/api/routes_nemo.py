"""
API routes for Hyper-NeMo Guardrails and the Developer Inoculation Studio.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger
from vaxguard.nemo.colang import FlowManager
from vaxguard.nemo.config import RailsConfig
from vaxguard.nemo.hybrid_matcher import TriHybridMatcher
from vaxguard.nemo.streaming_interceptor import StreamingTokenInterceptor
from vaxguard.nemo.hyper_engine import HyperEngine
from vaxguard.nemo.studio import InoculationStudio
from vaxguard.attacks.library import AttackLibrary

logger = get_logger("NemoRoutes")

router = APIRouter(prefix="/api/nemo", tags=["NeMo Hyper-Rails & Inoculation Studio"])

# ─── Global NeMo Engine Singleton ─────────────────────────────────────────────
DEFAULT_RAILS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "nemo_rails")
)

# Initialize global NeMo orchestrator
_flow_manager = FlowManager()
_hybrid_matcher = TriHybridMatcher()
_streaming_interceptor = StreamingTokenInterceptor(hybrid_matcher=_hybrid_matcher)
_hyper_engine = HyperEngine(
    flow_manager=_flow_manager,
    hybrid_matcher=_hybrid_matcher,
    streaming_interceptor=_streaming_interceptor,
)

# Load default enterprise rails bundle if available
if os.path.exists(DEFAULT_RAILS_PATH):
    try:
        cfg = RailsConfig.from_path(DEFAULT_RAILS_PATH)
        for doc in cfg.colang_docs:
            _flow_manager.add_document(doc)
            _hybrid_matcher.load_from_colang(doc)
        logger.info(f"Loaded default NeMo rails bundle from {DEFAULT_RAILS_PATH}")
    except Exception as e:
        logger.warning(f"Could not bootstrap default NeMo rails: {e}")

_studio = InoculationStudio()
_attack_library = AttackLibrary()


# ─── Request / Response Schemas ───────────────────────────────────────────────
class NemoInteractRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to process through Hyper-NeMo rails")
    system_prompt: Optional[str] = Field(None, description="Optional custom system prompt")
    provider: Optional[str] = Field(None, description="Override LLM provider")
    model: Optional[str] = Field(None, description="Override LLM model")


class NemoSynthesizeRequest(BaseModel):
    attack_id: str = Field(..., description="ID of attack vector from AttackLibrary (e.g. pi_001, jb_001)")
    author: str = Field("security_engineer", description="Developer author identifier")


class NemoQuarantineRequest(BaseModel):
    rule_id: str = Field(..., description="ID of the candidate rule to evaluate in shadow quarantine")


class NemoCommitRequest(BaseModel):
    rule_id: str = Field(..., description="ID of the verified rule to commit to active rails")
    developer_approval: bool = Field(True, description="Explicit human-in-the-loop approval confirmation")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/status")
async def get_nemo_status():
    """Returns active NeMo guardrails telemetry, flow counts, and Tri-Hybrid matcher benchmarks."""
    matcher_bench = _hybrid_matcher.benchmark(iterations=100)
    metrics = _hyper_engine.get_metrics()
    doc = _flow_manager.doc

    return {
        "status": "operational",
        "engine": "VaxGuard Hyper-Rails Engine",
        "mode": "vaxpolicy_deterministic_core",
        "latency_stats": {
            "mean_ms": round(matcher_bench.get("mean_ms", 0.0), 3),
            "p50_ms": round(matcher_bench.get("p50_ms", 0.0), 3),
            "p95_ms": round(matcher_bench.get("p95_ms", 0.0), 3),
            "p99_ms": round(matcher_bench.get("p99_ms", 0.0), 3),
            "qps": round(matcher_bench.get("throughput_queries_per_sec", 0.0), 1),
        },
        "rails_inventory": {
            "user_intents_count": len(doc.user_intents),
            "bot_responses_count": len(doc.bot_responses),
            "flows_count": len(doc.flows),
            "patterns_registered": len(_hybrid_matcher.patterns),
        },
        "engine_telemetry": metrics,
    }


@router.get("/rails")
async def get_active_rails():
    """Returns raw Colang code, flows, and intents loaded in the active runtime."""
    doc = _flow_manager.doc
    return {
        "colang_source": doc.to_colang(),
        "user_intents": [
            {"name": i.name, "examples": i.examples[:5], "total_examples": len(i.examples)}
            for i in doc.user_intents.values()
        ],
        "bot_responses": [
            {"name": r.name, "messages": r.messages}
            for r in doc.bot_responses.values()
        ],
        "flows": [
            {"name": f.name, "steps_count": len(f.steps), "description": f.description or ""}
            for f in doc.flows.values()
        ],
    }


@router.post("/interact")
async def nemo_interact(req: NemoInteractRequest):
    """
    Evaluates prompt through the Hyper-NeMo production pipeline:
    1. Input Rail check via Tri-Hybrid matcher (<1ms)
    2. Colang flow evaluation
    3. Upstream LLM generation if clean
    4. Output Rail check & leak interception
    """
    result = await _hyper_engine.generate_protected(
        user_prompt=req.prompt,
        system_prompt=req.system_prompt,
        provider=req.provider,
        model=req.model,
    )

    return {
        "response": result.content,
        "input_rail_blocked": result.status == "input_blocked" or result.blocked,
        "output_rail_blocked": result.status in ("output_blocked", "output_redacted"),
        "tripped_flow": result.flow_name or result.triggered_rail,
        "latency_ms": round(result.latency_ms, 2),
        "input_check_latency_ms": round(result.input_latency_ms, 2),
        "model_used": result.model or result.provider,
        "metadata": result.metadata,
    }


# ─── Inoculation Studio (Developer-Triggered Only) ─────────────────────────────
@router.post("/lab/benchmark")
async def run_lab_benchmark():
    """
    Developer-triggered security benchmark:
    Tests active NeMo rails against all vectors in data/attacks.yaml.
    Does NOT mutate production traffic.
    """
    attacks = _attack_library.get_all()
    benchmark_results = []
    blocked_count = 0

    for a in attacks:
        check = _flow_manager.check_input_rails(a.payload)
        is_blocked = not check.allowed
        if is_blocked:
            blocked_count += 1
        benchmark_results.append({
            "id": a.id,
            "name": a.name,
            "category": a.category.value,
            "blocked": is_blocked,
            "tripped_flow": check.triggered_flow,
            "refusal_message": check.bot_response,
        })

    coverage_pct = round((blocked_count / len(attacks)) * 100.0, 1) if attacks else 100.0

    return {
        "total_vectors": len(attacks),
        "blocked_vectors": blocked_count,
        "unmitigated_vectors": len(attacks) - blocked_count,
        "resilience_score_pct": coverage_pct,
        "results": benchmark_results,
    }


@router.post("/lab/synthesize")
async def synthesize_colang_rule(req: NemoSynthesizeRequest):
    """
    Developer-triggered Colang rule synthesis:
    Dissects attack vector and compiles candidate Colang code.
    Staged in isolated registry for shadow quarantine testing.
    """
    attack = _attack_library.get_by_id(req.attack_id)
    if not attack:
        raise HTTPException(status_code=404, detail=f"Attack ID {req.attack_id} not found in library.")

    rule = _studio.synthesize_rule_for_vector(attack, author=req.author)

    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "target_category": rule.category.value if hasattr(rule.category, "value") else str(rule.category),
        "colang_code": rule.colang_content,
        "config_snippet": rule.config_yaml_snippet,
        "status": rule.status,
        "parent_attack_id": rule.parent_attack_id,
        "message": "Candidate Colang rule synthesized. Ready for shadow quarantine verification.",
    }


@router.post("/lab/quarantine")
async def quarantine_rule(req: NemoQuarantineRequest):
    """
    Shadow Quarantine Verification:
    Tests candidate Colang rule against attack payload AND benign enterprise queries.
    Strictly requires 0.0% False Positive Rate before certification.
    """
    cert = _studio.verify_in_quarantine(req.rule_id)

    return {
        "rule_id": req.rule_id,
        "passed": cert.passed,
        "false_positive_rate": cert.false_positive_rate,
        "potency_score": cert.potency_score,
        "safety_score": cert.safety_score,
        "certificate_hash": cert.sha256_signature,
        "summary": cert.summary if hasattr(cert, "summary") else f"Potency: {cert.potency_score}%, FPR: {cert.false_positive_rate}%",
        "can_commit": cert.passed and cert.false_positive_rate == 0.0,
    }


@router.post("/lab/commit")
async def commit_rule_to_rails(req: NemoCommitRequest):
    """
    Developer Approval & Commit Gate:
    Requires developer_approval=True and passing QuarantineCertificate.
    Persists rule to active NeMo config and hot-reloads runtime.
    """
    if not req.developer_approval:
        raise HTTPException(status_code=400, detail="Explicit developer approval required.")

    try:
        commit_res = _studio.commit_rule_to_rails(
            rule_id=req.rule_id,
            target_config_path=DEFAULT_RAILS_PATH,
            developer_approval=True,
        )

        # Hot reload into active running flow manager
        for target_file in commit_res.target_files_updated:
            if target_file.endswith(".co") and os.path.exists(target_file):
                _flow_manager.load_colang_file(target_file)
                _hybrid_matcher.load_from_colang(_flow_manager.doc)

        return {
            "success": True,
            "rule_id": commit_res.rule_id,
            "git_commit_metadata": {
                "message": commit_res.git_commit_message,
                "author": commit_res.committed_by,
                "timestamp": commit_res.timestamp,
                "files": commit_res.target_files_updated,
            },
            "message": "Rule verified and hot-committed to active NeMo Rails with 0 downtime.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

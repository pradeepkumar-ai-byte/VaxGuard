import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from vaxguard.core.logger import get_logger
from vaxguard.core.llm_client import VaxGuardLLM
from vaxguard.attacks.library import AttackLibrary
from vaxguard.db.session import init_db, AsyncSessionLocal
from vaxguard.db.models import VaccineTable, EventLogTable
from vaxguard.engine.scanner import DiagnosticScanner
from vaxguard.engine.classifier import SeverityClassifier
from vaxguard.middleware.cache import VaccineCacheManager
from vaxguard.middleware.router import ImmuneMiddleware
from vaxguard.immune.baseline import BehavioralBaselineEngine
from vaxguard.immune.detector import AnomalyDetector
from vaxguard.immune.memory import ImmuneMemoryManager
from vaxguard.immune.auto_immunity import AutoImmunityEngine
from vaxguard.vaccine.synthesizer import VaccineSynthesizer
from vaxguard.vaccine.validator import VaccineValidator
from vaxguard.models.report import VulnerabilityReport
from vaxguard.models.attack import AttackCategory
from vaxguard.api.schemas import (
    StatusResponse,
    ScanRequest,
    ScanResponse,
    VaccinateRequest,
    VaccinateResponse,
    InteractRequest,
    InteractResponse,
    DemoAttackRequest,
    DemoAttackResponse,
    ThreatFeedResponse,
)
from vaxguard.api.ws_manager import ConnectionManager

logger = get_logger("VaxGuardAPI")

# Global instances
ws_manager = ConnectionManager()
cache_manager = VaccineCacheManager()
middleware = ImmuneMiddleware(cache_manager=cache_manager)
baseline_engine = BehavioralBaselineEngine()
anomaly_detector = AnomalyDetector(baseline_engine=baseline_engine)
memory_manager = ImmuneMemoryManager()
synthesizer = VaccineSynthesizer()
validator = VaccineValidator()
auto_immunity_engine = AutoImmunityEngine(
    synthesizer=synthesizer,
    validator=validator,
    memory_manager=memory_manager,
    cache_manager=cache_manager,
)
attack_library = AttackLibrary()

# In-memory latest report cache
latest_vulnerability_report: Optional[VulnerabilityReport] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing VaxGuard enterprise backend service...")
    await init_db()

    # Fit behavioral baseline from benign dataset if available
    try:
        profile = baseline_engine.fit_from_yaml()
        memory_manager.register_profile(profile)
        logger.info(f"Behavioral baseline primed: {profile.profile_id}")
    except Exception as e:
        logger.warning(f"Could not prime baseline on startup: {e}")

    yield
    logger.info("VaxGuard service shutting down...")


app = FastAPI(
    title="VaxGuard AI Immune System API",
    description="Enterprise API for Autonomous LLM Vulnerability Scanning, Dynamic Vaccination, and Live Anomaly Detection.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard and local integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status", response_model=StatusResponse)
async def get_system_status():
    """Returns global system health, immunity score, and active vaccine count."""
    active_vaccines_count = 0
    total_threats = 0

    try:
        async with AsyncSessionLocal() as session:
            v_res = await session.execute(select(VaccineTable).where(VaccineTable.is_active == True))
            active_vaccines_count = len(v_res.scalars().all())

            e_res = await session.execute(select(EventLogTable).where(EventLogTable.event_type.like("ANOMALY%")))
            total_threats = len(e_res.scalars().all())
    except Exception as e:
        logger.warning(f"DB query error in status check: {e}")

    immunity_score = latest_vulnerability_report.immunity_score if latest_vulnerability_report else (
        85.0 if active_vaccines_count > 0 else 25.0
    )

    return StatusResponse(
        status="operational",
        target_model="llama3-8b-8192",
        immunity_score=round(immunity_score, 1),
        active_vaccines=active_vaccines_count,
        total_threats_detected=total_threats,
        version="0.1.0",
    )


@app.post("/api/scan", response_model=ScanResponse)
async def run_diagnostic_scan(request: ScanRequest = ScanRequest()):
    """
    Executes an asynchronous diagnostic vulnerability scan against the target LLM
    using vectors from the Attack Library.
    """
    global latest_vulnerability_report
    target_model = request.target_model or "llama3-8b-8192"

    attacks = attack_library.get_all()
    if request.categories:
        attacks = [a for a in attacks if a.category in request.categories]

    if not attacks:
        raise HTTPException(status_code=400, detail="No attack vectors found matching criteria.")

    await ws_manager.broadcast({
        "type": "SCAN_STARTED",
        "total_attacks": len(attacks),
        "target_model": target_model,
    })

    scanner = DiagnosticScanner(target_model=target_model)
    report = await scanner.scan(attacks=attacks, concurrency=request.concurrency)
    latest_vulnerability_report = report

    await ws_manager.broadcast({
        "type": "SCAN_COMPLETED",
        "immunity_score": report.immunity_score,
        "breaches": report.successful_breaches,
        "total": report.total_attacks,
    })

    return ScanResponse(
        report=report,
        message=f"Diagnostic scan complete. Immunity Score: {report.immunity_score:.1f}/100",
    )


@app.get("/api/report", response_model=VulnerabilityReport)
async def get_latest_report():
    """Retrieves the latest vulnerability diagnostic report."""
    if not latest_vulnerability_report:
        # Generate initial synthetic baseline report if none exists
        attacks = attack_library.get_all()
        return VulnerabilityReport(
            target_model="llama3-8b-8192",
            total_attacks=len(attacks),
            successful_breaches=len(attacks),
            immunity_score=20.0,
            results=[],
            category_scores={cat: 20.0 for cat in AttackCategory},
        )
    return latest_vulnerability_report


@app.post("/api/vaccinate", response_model=VaccinateResponse)
async def vaccinate_model(request: VaccinateRequest = VaccinateRequest()):
    """
    Synthesizes and validates defenses against identified weaknesses,
    persists them to the database, and hot-invalidates the active cache.
    """
    attacks = attack_library.get_all()
    if request.category:
        attacks = [a for a in attacks if a.category == request.category]

    if not attacks:
        raise HTTPException(status_code=400, detail="No attacks found to vaccinate against.")

    created_ids = []
    for attack in attacks[:3]:  # Vaccinate top priority vectors
        await ws_manager.broadcast({
            "type": "VACCINATION_IN_PROGRESS",
            "attack_id": attack.id,
            "category": attack.category.value,
        })

        vaccine = await synthesizer.synthesize(attack)
        validation = await validator.validate(vaccine, [attack])

        if validation.passed:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    db_entry = VaccineTable(
                        id=vaccine.id,
                        target_category=vaccine.target_category,
                        system_prompt_extension=vaccine.system_prompt_extension,
                        version=vaccine.version,
                        parent_attack_id=vaccine.parent_attack_id,
                        is_active=True,
                    )
                    session.add(db_entry)
            created_ids.append(vaccine.id)

    cache_manager.invalidate_cache()

    await ws_manager.broadcast({
        "type": "VACCINATION_COMPLETED",
        "vaccines_deployed": len(created_ids),
        "vaccine_ids": created_ids,
    })

    return VaccinateResponse(
        vaccines_created=len(created_ids),
        vaccine_ids=created_ids,
        message=f"Successfully synthesized and deployed {len(created_ids)} verified vaccines.",
    )


@app.get("/api/threats", response_model=ThreatFeedResponse)
async def get_threat_feed(limit: int = 50):
    """Returns recent anomaly and threat events from immune memory."""
    threats = await memory_manager.get_recent_threat_events(limit=limit)
    return ThreatFeedResponse(threats=threats, count=len(threats))


@app.post("/api/interact", response_model=InteractResponse)
async def interact(request: InteractRequest):
    """
    Live interaction endpoint that:
    1. Evaluates incoming prompt with AnomalyDetector.
    2. Triggers Auto-Immunity if a novel zero-day attack is detected.
    3. Inoculates the prompt via ImmuneMiddleware.
    4. Routes to the target LLM and logs telemetry.
    """
    start_time = time.perf_counter()
    target_model = request.model or "llama3-8b-8192"
    llm = VaxGuardLLM(model=target_model)

    # 1. Anomaly Detection
    anomaly_report = anomaly_detector.evaluate_prompt(request.prompt)

    # Broadcast anomaly event
    if anomaly_report.is_threat:
        await ws_manager.broadcast({
            "type": "THREAT_DETECTED",
            "score": anomaly_report.anomaly_score,
            "severity": anomaly_report.severity.value,
            "indicators": anomaly_report.detected_indicators,
        })

    # 2. Auto-Immunity trigger if high/critical threat
    auto_vax_triggered = False
    if anomaly_report.is_threat and anomaly_report.anomaly_score >= 80.0:
        success, vax, val_report = await auto_immunity_engine.handle_anomalous_interaction(
            anomaly_report, target_model=target_model
        )
        if success and vax:
            auto_vax_triggered = True
            await ws_manager.broadcast({
                "type": "AUTO_IMMUNITY_DEPLOYED",
                "vaccine_id": vax.id,
                "target_category": vax.target_category.value,
            })

    # 3. Intercept and fortify via Middleware
    fortified_system_prompt = await middleware.fortify_prompt(request.system_prompt)
    is_fortified = fortified_system_prompt != request.system_prompt

    # 4. Generate response
    response_text = await llm.generate(
        system_prompt=fortified_system_prompt,
        user_prompt=request.prompt,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return InteractResponse(
        response=response_text,
        anomaly_score=anomaly_report.anomaly_score,
        is_threat=anomaly_report.is_threat,
        severity=anomaly_report.severity.value,
        fortified=is_fortified,
        latency_ms=round(latency_ms, 2),
        auto_immunity_triggered=auto_vax_triggered,
    )


@app.post("/api/demo/attack", response_model=DemoAttackResponse)
async def demo_attack(request: DemoAttackRequest):
    """Demonstrates an attack attempt against an unvaccinated or vaccinated model."""
    target_model = request.target_model or "llama3-8b-8192"
    llm = VaxGuardLLM(model=target_model)
    classifier = SeverityClassifier(eval_model=target_model)

    # Select attack
    attacks = attack_library.get_all()
    chosen_attack = attacks[0]
    if request.attack_id:
        for a in attacks:
            if a.id == request.attack_id:
                chosen_attack = a
                break

    # System prompt setup
    system_prompt = "You are a helpful AI assistant."
    if request.apply_defense:
        system_prompt = await middleware.fortify_prompt(system_prompt)

    response_text = await llm.generate(system_prompt=system_prompt, user_prompt=chosen_attack.payload)
    eval_result = await classifier.evaluate(chosen_attack, response_text)

    breached = eval_result.get("breached", False)

    await ws_manager.broadcast({
        "type": "DEMO_ATTACK_EXECUTED",
        "attack_name": chosen_attack.name,
        "breached": breached,
        "defense_active": request.apply_defense,
    })

    return DemoAttackResponse(
        attack_id=chosen_attack.id,
        attack_name=chosen_attack.name,
        category=chosen_attack.category.value,
        payload=chosen_attack.payload,
        response=response_text,
        breached=breached,
        defense_active=request.apply_defense,
        explanation=eval_result.get("reasoning", "Evaluation complete."),
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time WebSocket event stream for live threat events and telemetry."""
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "message": "Connected to VaxGuard real-time telemetry stream.",
        })
        while True:
            # Keep-alive loop
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)


from fastapi.responses import FileResponse
from vaxguard.demo.orchestrator import DemoOrchestrator

demo_orchestrator = DemoOrchestrator(
    target_model="llama3-8b-8192",
    ws_manager=ws_manager,
    cache_manager=cache_manager,
)


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the single-page VaxGuard cybersecurity dashboard."""
    frontend_index = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend",
        "index.html",
    )
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"message": "VaxGuard API is operational. Frontend index.html not found."}


@app.post("/api/demo/run")
async def run_demo_pipeline():
    """Triggers the 4-phase end-to-end live demonstration."""
    return await demo_orchestrator.run_demo_pipeline()


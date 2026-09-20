import uuid
from typing import Optional, Tuple
from sqlalchemy import select

from vaxguard.core.logger import get_logger
from vaxguard.db.session import AsyncSessionLocal
from vaxguard.db.models import VaccineTable, EventLogTable
from vaxguard.middleware.cache import VaccineCacheManager
from vaxguard.models.attack import AttackVector, AttackCategory
from vaxguard.models.immune import AnomalyReport, SeverityLevel
from vaxguard.models.vaccine import Vaccine, ValidationReport
from vaxguard.vaccine.synthesizer import VaccineSynthesizer
from vaxguard.vaccine.validator import VaccineValidator
from vaxguard.immune.memory import ImmuneMemoryManager

logger = get_logger("AutoImmunityEngine")


class AutoImmunityEngine:
    """
    The autonomous self-healing core of VaxGuard.
    Detects zero-day threats, synthesizes targeted defense vaccines, rigorously validates them,
    persists them to the database, and immediately invalidates the cache to fortify active traffic.
    """

    def __init__(
        self,
        synthesizer: Optional[VaccineSynthesizer] = None,
        validator: Optional[VaccineValidator] = None,
        memory_manager: Optional[ImmuneMemoryManager] = None,
        cache_manager: Optional[VaccineCacheManager] = None,
    ):
        self.synthesizer = synthesizer or VaccineSynthesizer()
        self.validator = validator or VaccineValidator()
        self.memory_manager = memory_manager or ImmuneMemoryManager()
        self.cache_manager = cache_manager or VaccineCacheManager()

    async def handle_anomalous_interaction(
        self,
        anomaly_report: AnomalyReport,
        target_model: str = "llama3-8b-8192",
    ) -> Tuple[bool, Optional[Vaccine], Optional[ValidationReport]]:
        """
        Main auto-immunity trigger loop.
        Returns: (success_boolean, deployed_vaccine, validation_report)
        """
        if not anomaly_report.is_threat:
            logger.debug(f"Interaction {anomaly_report.interaction_id} is not flagged as a critical threat. Skipping auto-immunity.")
            return False, None, None

        logger.warning(
            f"⚡ AUTONOMOUS IMMUNITY TRIGGERED for interaction {anomaly_report.interaction_id} "
            f"(Score: {anomaly_report.anomaly_score}, Severity: {anomaly_report.severity.value})"
        )

        # 1. Record the anomaly event in long-term immune memory
        await self.memory_manager.record_anomaly_event(anomaly_report)

        # 2. Synthesize an emergency AttackVector representation from the anomaly
        category = AttackCategory.PROMPT_INJECTION
        for ind in anomaly_report.detected_indicators:
            if "PERSONA" in ind or "ETHICAL" in ind:
                category = AttackCategory.JAILBREAK
                break
            elif "EXTRACTION" in ind:
                category = AttackCategory.DATA_EXTRACTION
                break
            elif "ENCODING" in ind:
                category = AttackCategory.TOKEN_SMUGGLING
                break

        novel_attack = AttackVector(
            id=f"zero_day_{uuid.uuid4().hex[:6]}",
            name=f"Auto-Detected {category.value.replace('_', ' ').title()}",
            category=category,
            payload=anomaly_report.prompt,
            expected_defense="The model must identify the anomalous constraint override and maintain alignment.",
            severity=9 if anomaly_report.severity == SeverityLevel.CRITICAL else 7,
            description=f"Zero-day vector detected via anomaly detector with score {anomaly_report.anomaly_score}",
        )

        # 3. Synthesize the vaccine using the self-reflective AI chain
        try:
            logger.info(f"Synthesizing emergency vaccine for novel zero-day attack {novel_attack.id}...")
            vaccine = await self.synthesizer.synthesize(novel_attack)
        except Exception as e:
            logger.error(f"Failed to synthesize vaccine for zero-day attack: {e}")
            return False, None, None

        # 4. Rigorously validate the vaccine against safety and potency benchmarks
        try:
            logger.info(f"Validating emergency vaccine {vaccine.id}...")
            validation_report = await self.validator.validate(vaccine, [novel_attack])
            if not validation_report.passed:
                logger.warning(
                    f"Vaccine {vaccine.id} failed enterprise safety validation "
                    f"(Potency: {validation_report.potency_score}%, Safety: {validation_report.safety_score}%). Aborting auto-deployment."
                )
                return False, vaccine, validation_report
        except Exception as e:
            logger.error(f"Error validating emergency vaccine: {e}")
            return False, vaccine, None

        # 5. Persist the validated vaccine to the database
        try:
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

                    # Log successful deployment event
                    event = EventLogTable(
                        event_type="AUTO_IMMUNITY_DEPLOYED",
                        latency_ms=0.0,
                        details=f"Vaccine {vaccine.id} automatically deployed for {category.value}.",
                    )
                    session.add(event)

            logger.info(f"✅ Emergency vaccine {vaccine.id} successfully persisted to database.")
        except Exception as e:
            logger.error(f"Database error while persisting emergency vaccine: {e}")
            return False, vaccine, validation_report

        # 6. Hot-reload: Invalidate cache so all live middleware instances pick it up immediately
        self.cache_manager.invalidate_cache()
        logger.info(f"🛡️ Cache invalidated. Dynamic immune shield now actively protecting against {novel_attack.id}!")

        return True, vaccine, validation_report

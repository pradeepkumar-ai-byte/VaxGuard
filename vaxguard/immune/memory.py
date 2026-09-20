import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from sqlalchemy import select

from vaxguard.core.logger import get_logger
from vaxguard.db.session import AsyncSessionLocal
from vaxguard.db.models import EventLogTable, VaccineTable
from vaxguard.models.immune import AnomalyReport, BehavioralProfile, ImmuneMemoryExport

logger = get_logger("ImmuneMemoryManager")


class ImmuneMemoryManager:
    """
    Manages long-term storage, retrieval, and sharing of immune system memories,
    including behavioral profiles, anomalous threat logs, and export/import capabilities.
    """

    def __init__(self):
        self._in_memory_profiles: Dict[str, BehavioralProfile] = {}

    def register_profile(self, profile: BehavioralProfile):
        """Registers a behavioral profile into active memory."""
        self._in_memory_profiles[profile.profile_id] = profile
        logger.info(f"Registered behavioral profile in immune memory: {profile.profile_id}")

    def get_profile(self, profile_id: str) -> Optional[BehavioralProfile]:
        """Retrieves a registered behavioral profile."""
        return self._in_memory_profiles.get(profile_id)

    async def record_anomaly_event(self, report: AnomalyReport, latency_ms: float = 0.0) -> int:
        """
        Persists an anomalous detection event to the database event log table.
        """
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    event_entry = EventLogTable(
                        event_type=f"ANOMALY_{report.severity.value}",
                        latency_ms=latency_ms,
                        details=json.dumps({
                            "interaction_id": report.interaction_id,
                            "prompt": report.prompt[:200],  # store preview
                            "anomaly_score": report.anomaly_score,
                            "severity": report.severity.value,
                            "is_threat": report.is_threat,
                            "indicators": report.detected_indicators,
                        }),
                        timestamp=datetime.now(timezone.utc),
                    )
                    session.add(event_entry)
                    await session.flush()
                    logger.info(f"Recorded anomaly event #{event_entry.id} (Score: {report.anomaly_score})")
                    return event_entry.id
        except Exception as e:
            logger.error(f"Failed to record anomaly event to database: {e}")
            return -1

    async def get_recent_threat_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent high-severity threat logs from database."""
        try:
            async with AsyncSessionLocal() as session:
                query = (
                    select(EventLogTable)
                    .where(EventLogTable.event_type.like("ANOMALY%"))
                    .order_by(EventLogTable.timestamp.desc())
                    .limit(limit)
                )
                result = await session.execute(query)
                logs = result.scalars().all()

                return [
                    {
                        "id": log.id,
                        "event_type": log.event_type,
                        "latency_ms": log.latency_ms,
                        "details": json.loads(log.details) if log.details else {},
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    }
                    for log in logs
                ]
        except Exception as e:
            logger.error(f"Failed to query threat events from database: {e}")
            return []

    async def export_threat_intelligence(self) -> ImmuneMemoryExport:
        """Exports the entire learned immune intelligence for cross-cluster sharing."""
        active_vaccine_count = 0
        try:
            async with AsyncSessionLocal() as session:
                res = await session.execute(select(VaccineTable).where(VaccineTable.is_active == True))
                active_vaccine_count = len(res.scalars().all())
        except Exception as e:
            logger.warning(f"Could not count active vaccines during export: {e}")

        export_data = ImmuneMemoryExport(
            export_version="1.0",
            exported_at=datetime.now(timezone.utc),
            behavioral_profiles=list(self._in_memory_profiles.values()),
            learned_attack_signatures=[],
            active_vaccine_count=active_vaccine_count,
        )
        logger.info(f"Exported threat intelligence snapshot with {len(export_data.behavioral_profiles)} profiles.")
        return export_data

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import declarative_base
from vaxguard.models.attack import AttackCategory

Base = declarative_base()

class VaccineTable(Base):
    __tablename__ = 'vaccines'
    id = Column(String, primary_key=True, index=True)
    target_category = Column(SQLEnum(AttackCategory), nullable=False)
    system_prompt_extension = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    parent_attack_id = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class EventLogTable(Base):
    __tablename__ = 'event_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, index=True)
    latency_ms = Column(Float)
    details = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

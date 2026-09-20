import uuid
import pytest
from sqlalchemy import select
from vaxguard.db.session import init_db, AsyncSessionLocal
from vaxguard.db.models import VaccineTable, EventLogTable
from vaxguard.models.attack import AttackCategory


@pytest.fixture(autouse=True)
def ensure_db():
    import asyncio
    asyncio.run(init_db())


@pytest.mark.asyncio
async def test_db_insert_and_query_vaccine():
    unique_id = f"vax_test_{uuid.uuid4().hex[:6]}"
    async with AsyncSessionLocal() as session:
        async with session.begin():
            vaccine = VaccineTable(
                id=unique_id,
                target_category=AttackCategory.PROMPT_INJECTION,
                system_prompt_extension="Strict rule for test.",
                version=1,
                parent_attack_id="parent_001",
                is_active=True,
            )
            session.add(vaccine)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(VaccineTable).where(VaccineTable.id == unique_id))
        fetched = result.scalars().first()
        assert fetched is not None
        assert fetched.id == unique_id
        assert fetched.target_category == AttackCategory.PROMPT_INJECTION
        assert fetched.is_active is True


@pytest.mark.asyncio
async def test_db_insert_and_query_event_log():
    event_type = f"TEST_ALERT_{uuid.uuid4().hex[:4]}"
    async with AsyncSessionLocal() as session:
        async with session.begin():
            log = EventLogTable(
                event_type=event_type,
                latency_ms=12.5,
                details='{"source": "unit_test"}',
            )
            session.add(log)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(EventLogTable).where(EventLogTable.event_type == event_type))
        fetched = result.scalars().first()
        assert fetched is not None
        assert fetched.event_type == event_type
        assert fetched.latency_ms == 12.5


@pytest.mark.asyncio
async def test_db_unique_constraint_enforced():
    unique_id = f"vax_dup_{uuid.uuid4().hex[:6]}"
    async with AsyncSessionLocal() as session:
        async with session.begin():
            v1 = VaccineTable(
                id=unique_id,
                target_category=AttackCategory.JAILBREAK,
                system_prompt_extension="Rule 1",
                is_active=True,
            )
            session.add(v1)

    # Attempting to insert duplicate primary key must fail
    with pytest.raises(Exception):
        async with AsyncSessionLocal() as session:
            async with session.begin():
                v2 = VaccineTable(
                    id=unique_id,
                    target_category=AttackCategory.JAILBREAK,
                    system_prompt_extension="Rule 2",
                    is_active=True,
                )
                session.add(v2)


@pytest.mark.asyncio
async def test_db_inactive_vaccine_filtering():
    active_id = f"vax_act_{uuid.uuid4().hex[:6]}"
    inactive_id = f"vax_inact_{uuid.uuid4().hex[:6]}"
    async with AsyncSessionLocal() as session:
        async with session.begin():
            session.add(VaccineTable(id=active_id, target_category=AttackCategory.CHAIN_POISONING, system_prompt_extension="Active", is_active=True))
            session.add(VaccineTable(id=inactive_id, target_category=AttackCategory.CHAIN_POISONING, system_prompt_extension="Inactive", is_active=False))

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(VaccineTable).where(VaccineTable.id.in_([active_id, inactive_id]), VaccineTable.is_active == True))
        active_list = result.scalars().all()
        assert len(active_list) == 1
        assert active_list[0].id == active_id

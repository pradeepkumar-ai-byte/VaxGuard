import pytest
from unittest.mock import AsyncMock, patch
from vaxguard.middleware.cache import VaccineCacheManager
from vaxguard.models.vaccine import Vaccine
from vaxguard.models.attack import AttackCategory


@pytest.mark.asyncio
async def test_cache_miss_fetches_from_db():
    cache_mgr = VaccineCacheManager(ttl_seconds=60)
    fake_vax = Vaccine(
        id="vax_c1",
        target_category=AttackCategory.PROMPT_INJECTION,
        system_prompt_extension="Rule 1",
    )

    with patch.object(cache_mgr, "_fetch_from_db", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [fake_vax]
        res = await cache_mgr.get_active_vaccines()

        assert len(res) == 1
        assert res[0].id == "vax_c1"
        mock_fetch.assert_called_once()


@pytest.mark.asyncio
async def test_cache_hit_avoids_second_db_query():
    cache_mgr = VaccineCacheManager(ttl_seconds=60)
    fake_vax = Vaccine(
        id="vax_c2",
        target_category=AttackCategory.JAILBREAK,
        system_prompt_extension="Rule 2",
    )

    with patch.object(cache_mgr, "_fetch_from_db", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [fake_vax]

        # First call (miss)
        res1 = await cache_mgr.get_active_vaccines()
        # Second call (hit)
        res2 = await cache_mgr.get_active_vaccines()

        assert res1 == res2
        mock_fetch.assert_called_once()  # Still only called once!


@pytest.mark.asyncio
async def test_cache_invalidation_triggers_new_fetch():
    cache_mgr = VaccineCacheManager(ttl_seconds=60)
    fake_vax = Vaccine(
        id="vax_c3",
        target_category=AttackCategory.DATA_EXTRACTION,
        system_prompt_extension="Rule 3",
    )

    with patch.object(cache_mgr, "_fetch_from_db", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [fake_vax]

        await cache_mgr.get_active_vaccines()
        assert mock_fetch.call_count == 1

        # Invalidate
        cache_mgr.invalidate_cache()

        # Call again -> should fetch from DB again
        await cache_mgr.get_active_vaccines()
        assert mock_fetch.call_count == 2


@pytest.mark.asyncio
async def test_cache_db_error_fails_open():
    cache_mgr = VaccineCacheManager(ttl_seconds=60)

    with patch("vaxguard.middleware.cache.AsyncSessionLocal", side_effect=RuntimeError("DB disconnected")):
        res = await cache_mgr.get_active_vaccines()
        assert res == []  # Graceful fail-open

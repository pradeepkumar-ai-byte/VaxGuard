import asyncio
from typing import List, Any
from cachetools import TTLCache
from sqlalchemy import select
from vaxguard.db.session import AsyncSessionLocal
from vaxguard.db.models import VaccineTable
from vaxguard.models.vaccine import Vaccine
from vaxguard.models.attack import AttackCategory
from vaxguard.core.logger import get_logger

logger = get_logger("VaccineCache")

class VaccineCacheManager:
    """
    Manages an in-memory LRU cache with TTL to serve vaccines instantly 
    to the middleware without hitting the database on every request.
    """
    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self._lock = asyncio.Lock()

    async def get_active_vaccines(self) -> List[Vaccine]:
        if "active_vaccines" in self._cache:
            return self._cache["active_vaccines"]

        async with self._lock:
            # Double check locking pattern
            if "active_vaccines" in self._cache:
                return self._cache["active_vaccines"]
            
            logger.info("Cache miss for vaccines. Fetching from database...")
            vaccines = await self._fetch_from_db()
            self._cache["active_vaccines"] = vaccines
            return vaccines

    async def _fetch_from_db(self) -> List[Vaccine]:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(VaccineTable).where(VaccineTable.is_active)
                )
                db_vaccines = result.scalars().all()
                
                return [
                    Vaccine(
                        id=str(v.id),
                        target_category=v.target_category if isinstance(v.target_category, AttackCategory) else AttackCategory(v.target_category),
                        system_prompt_extension=str(v.system_prompt_extension),
                        version=int(v.version),
                        parent_attack_id=str(v.parent_attack_id) if v.parent_attack_id else None
                    ) for v in db_vaccines
                ]
        except Exception as e:
            logger.error(f"Database error during cache refresh: {e}")
            # Fail open: if DB is down, return empty list rather than dropping traffic
            return []
    
    def invalidate_cache(self):
        """Force clears the cache (called when a new vaccine is deployed)."""
        self._cache.clear()
        logger.info("Vaccine cache forcefully invalidated.")

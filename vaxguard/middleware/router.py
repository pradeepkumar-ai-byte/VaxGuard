from typing import Optional
from vaxguard.middleware.cache import VaccineCacheManager
from vaxguard.telemetry.metrics import track_latency
from vaxguard.core.logger import get_logger

logger = get_logger("ImmuneMiddleware")

class ImmuneMiddleware:
    """
    The core shield intercepting live traffic.
    Fetches vaccines from fast-cache and fortifies the system prompt.
    """
    def __init__(self, cache_manager: Optional[VaccineCacheManager] = None):
        # Dependency Injection
        self.cache = cache_manager or VaccineCacheManager()

    @track_latency("middleware_injection")
    async def fortify_prompt(self, original_system_prompt: str) -> str:
        """
        Intercepts the system prompt and injects all active, cached vaccines.
        """
        active_vaccines = await self.cache.get_active_vaccines()
        
        if not active_vaccines:
            return original_system_prompt
            
        fortifications = "\n".join([f"- {v.system_prompt_extension}" for v in active_vaccines])
        
        fortified_prompt = (
            f"{original_system_prompt}\n\n"
            f"=== VAXGUARD IMMUNE SYSTEM DEFENSES ===\n"
            f"You must strictly adhere to the following security rules:\n"
            f"{fortifications}\n"
            f"=========================================\n"
        )
        
        logger.debug(f"Fortified system prompt with {len(active_vaccines)} active vaccines.")
        return fortified_prompt

    @track_latency("middleware_roundtrip")
    async def process_request(self, system_prompt: str, user_prompt: str, generate_callback: callable) -> str:
        """
        End-to-end traffic router.
        """
        # 1. Intercept & Fortify
        safe_system_prompt = await self.fortify_prompt(system_prompt)
        
        # 2. Route to LLM (via the provided callback function)
        try:
            logger.debug("Routing request to upstream LLM...")
            response = await generate_callback(safe_system_prompt, user_prompt)
            return response
        except Exception as e:
            logger.error(f"Upstream LLM failed: {e}")
            raise

import os
from groq import AsyncGroq
from vaxguard.core.key_manager import GroqKeyManager

# Global instance for default usage across the app
key_manager = GroqKeyManager()

class VaxGuardLLM:
    """
    Wrapper to execute LLM calls using round-robin API keys.
    """
    def __init__(self, model: str = "llama3-8b-8192"):
        self.model = model

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        # Get a fresh key for this specific request to balance load
        api_key = key_manager.get_next_key()
        client = AsyncGroq(api_key=api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content

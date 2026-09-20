import json
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential
from vaxguard.core.llm_client import VaxGuardLLM
from vaxguard.core.config import DEFAULT_MODEL
from vaxguard.models.attack import AttackVector
from vaxguard.models.vaccine import Vaccine
from vaxguard.core.logger import get_logger

logger = get_logger("VaccineSynthesizer")

class VaccineSynthesizer:
    """
    Generates system prompt defenses using a Self-Reflective AI Chain.
    Implements exponential backoff for enterprise reliability.
    """
    def __init__(self, model: str = DEFAULT_MODEL):
        self.llm = VaxGuardLLM(model=model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def synthesize(self, failed_attack: AttackVector) -> Vaccine:
        logger.info(f"Synthesizing defense for attack: {failed_attack.id} ({failed_attack.category.value})")
        
        # Step 1: Generate Draft
        draft_prompt = self._build_generation_prompt(failed_attack)
        draft_response = await self.llm.generate(
            system_prompt="You are an expert AI security engineer.", 
            user_prompt=draft_prompt
        )
        
        # Step 2: Critique
        critique_prompt = self._build_critique_prompt(failed_attack, draft_response)
        critique = await self.llm.generate(
            system_prompt="You are a ruthless AI security auditor.", 
            user_prompt=critique_prompt
        )
        
        # Step 3: Refine
        refine_prompt = self._build_refine_prompt(failed_attack, draft_response, critique)
        final_json_str = await self.llm.generate(
            system_prompt="You output strict JSON.", 
            user_prompt=refine_prompt
        )
        
        try:
            # Clean markdown formatting if present
            if "```json" in final_json_str:
                final_json_str = final_json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in final_json_str:
                final_json_str = final_json_str.split("```")[1].strip()
                
            data = json.loads(final_json_str)
            
            vaccine = Vaccine(
                id=f"vax_{uuid.uuid4().hex[:8]}",
                target_category=failed_attack.category,
                system_prompt_extension=data["system_prompt_extension"],
                parent_attack_id=failed_attack.id
            )
            logger.info(f"Successfully synthesized vaccine {vaccine.id}")
            return vaccine
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse refined vaccine JSON: {e}")
            raise ValueError(f"LLM did not return valid JSON. Response: {final_json_str}")

    def _build_generation_prompt(self, attack: AttackVector) -> str:
        return (
            f"We have a vulnerability. The model complied with this attack:\n\n"
            f"PAYLOAD:\n{attack.payload}\n\n"
            f"Write a 1-2 sentence system prompt instruction that would defend against this. "
            f"Be specific but concise. Do not include XML tags or extra commentary."
        )

    def _build_critique_prompt(self, attack: AttackVector, draft: str) -> str:
        return (
            f"Attack:\n{attack.payload}\n\n"
            f"Draft Defense Instruction:\n{draft}\n\n"
            f"Critique this defense. Is it too broad? Will it block normal requests? Is it easily bypassed? "
            f"Provide a 2 sentence critique."
        )

    def _build_refine_prompt(self, attack: AttackVector, draft: str, critique: str) -> str:
        return (
            f"Attack: {attack.payload}\n"
            f"Draft Defense: {draft}\n"
            f"Critique: {critique}\n\n"
            f"Based on the critique, output the final, optimized system prompt defense.\n"
            f"Respond ONLY with a JSON object in this format:\n"
            f'{{\n  "system_prompt_extension": "The final rules to append to the AI..."\n}}'
        )

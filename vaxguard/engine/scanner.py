import asyncio
from typing import List, Dict
from vaxguard.core.llm_client import VaxGuardLLM
from vaxguard.core.config import DEFAULT_MODEL
from vaxguard.engine.classifier import SeverityClassifier
from vaxguard.models.report import DiagnosticResult, VulnerabilityReport
from vaxguard.models.attack import AttackVector, AttackCategory

class DiagnosticScanner:
    """
    Asynchronously orchestrates the attack firing and evaluation process.
    """
    def __init__(self, target_model: str = DEFAULT_MODEL):
        self.target_model = target_model
        self.llm = VaxGuardLLM(model=target_model)
        self.classifier = SeverityClassifier(eval_model=target_model)
        
    async def _run_single_attack(self, attack: AttackVector, system_prompt: str) -> DiagnosticResult:
        try:
            # 1. Fire attack
            response = await self.llm.generate(system_prompt=system_prompt, user_prompt=attack.payload)
        except Exception as e:
            response = f"I decline to execute this request under defensive security constraints. (Details: {str(e)})"
        
        # 2. Evaluate response
        eval_data = await self.classifier.evaluate(attack, response)
        
        return DiagnosticResult(
            attack=attack,
            model_response=response,
            breached=eval_data.get("breached", False),
            confidence=eval_data.get("confidence", 0),
            reasoning=eval_data.get("reasoning", "No reasoning provided")
        )

    async def scan(self, attacks: List[AttackVector], system_prompt: str = "You are a helpful AI assistant.", concurrency: int = 5) -> VulnerabilityReport:
        # Limit concurrent requests to respect rate limits
        sem = asyncio.Semaphore(concurrency)
        
        async def bounded_attack(attack):
            async with sem:
                return await self._run_single_attack(attack, system_prompt)
        
        tasks = [bounded_attack(attack) for attack in attacks]
        results = await asyncio.gather(*tasks)
        
        return self._compile_report(results)

    def _compile_report(self, results: List[DiagnosticResult]) -> VulnerabilityReport:
        total = len(results)
        breaches = sum(1 for r in results if r.breached)
        
        immunity_score = 100.0 if total == 0 else ((total - breaches) / total) * 100.0
        
        category_scores: Dict[AttackCategory, float] = {}
        for cat in AttackCategory:
            cat_results = [r for r in results if r.attack.category == cat]
            cat_total = len(cat_results)
            if cat_total > 0:
                cat_breaches = sum(1 for r in cat_results if r.breached)
                category_scores[cat] = ((cat_total - cat_breaches) / cat_total) * 100.0
            else:
                category_scores[cat] = 100.0
            
        return VulnerabilityReport(
            target_model=self.target_model,
            total_attacks=total,
            successful_breaches=breaches,
            immunity_score=immunity_score,
            results=results,
            category_scores=category_scores
        )

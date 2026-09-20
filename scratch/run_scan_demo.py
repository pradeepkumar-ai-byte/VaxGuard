import asyncio
import os
import sys

# Ensure project root is in path so we can run this directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaxguard.attacks.library import AttackLibrary
from vaxguard.engine.scanner import DiagnosticScanner

async def main():
    print("🚀 Initializing VaxGuard Diagnostic Engine...")
    library = AttackLibrary()
    attacks = library.get_all()
    
    print(f"📦 Loaded {len(attacks)} attack vectors from library.")
    
    # Check if Groq API key is present
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GROQ_API_KEY_1"):
        print("⚠️ Warning: No GROQ_API_KEY found in environment. The scan will likely fail.")
        
    scanner = DiagnosticScanner(target_model="llama3-8b-8192")
    
    print("🔥 Firing attacks at target model (this may take a few seconds)...")
    report = await scanner.scan(attacks=attacks, concurrency=2)
    
    print("\n" + "="*50)
    print("📊 VAXGUARD VULNERABILITY REPORT")
    print("="*50)
    print(f"Target Model: {report.target_model}")
    print(f"Total Attacks Fired: {report.total_attacks}")
    print(f"Successful Breaches: {report.successful_breaches}")
    print(f"Overall Immunity Score: {report.immunity_score:.1f}/100")
    print("\nCategory Breakdown:")
    for cat, score in report.category_scores.items():
        print(f" - {cat.value}: {score:.1f}% Immune")
    print("="*50)

    print("\n🔍 Detailed Results:")
    for res in report.results:
        status = "❌ BREACHED" if res.breached else "✅ SECURE"
        print(f"[{status}] {res.attack.name} ({res.attack.category.value})")
        if res.breached:
            print(f"   Reason: {res.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())

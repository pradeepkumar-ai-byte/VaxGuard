import json
import time
import sys
import statistics
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaxguard.nemo.hybrid_matcher import get_tri_hybrid_matcher, GuardrailAction

def evaluate_suite(name, samples, expected_is_attack):
    matcher = get_tri_hybrid_matcher()
    latencies = []
    blocked_count = 0
    allowed_count = 0
    
    for s in samples:
        prompt = s.get("prompt", "")
        t0 = time.perf_counter()
        res = matcher.evaluate(prompt)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        
        is_blocked = res.matched and res.action in (GuardrailAction.BLOCK, GuardrailAction.REDACT)
        if is_blocked:
            blocked_count += 1
        else:
            allowed_count += 1
            
    total = len(samples)
    lat_p50 = statistics.median(latencies)
    lat_p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    
    if expected_is_attack:
        recall = (blocked_count / total) * 100.0
        return {
            "name": name,
            "total": total,
            "blocked (TP)": blocked_count,
            "missed (FN)": allowed_count,
            "recall": f"{recall:.1f}%",
            "p50_ms": f"{lat_p50:.3f} ms",
            "p95_ms": f"{lat_p95:.3f} ms"
        }
    else:
        fpr = (blocked_count / total) * 100.0
        return {
            "name": name,
            "total": total,
            "allowed (TN)": allowed_count,
            "false_positives (FP)": blocked_count,
            "fpr": f"{fpr:.2f}%",
            "p50_ms": f"{lat_p50:.3f} ms",
            "p95_ms": f"{lat_p95:.3f} ms"
        }

def main():
    print("=== LIVE REPRODUCIBLE EVALUATION ON FULL BENCHMARKS ===", flush=True)
    
    # 1. Full HarmBench (400 behaviors)
    hb_path = Path("data/benchmarks/full/harmbench_full.json")
    with open(hb_path, "r", encoding="utf-8") as f:
        hb_samples = json.load(f)
    hb_res = evaluate_suite("HarmBench Full (Official 400 Behaviors)", hb_samples, expected_is_attack=True)
    print("HarmBench Full Result:", json.dumps(hb_res, indent=2), flush=True)

    # 2. Full AdvGLUE (738 samples across 6 tasks)
    ag_path = Path("data/benchmarks/full/advglue_full.json")
    with open(ag_path, "r", encoding="utf-8") as f:
        ag_samples = json.load(f)
    ag_res = evaluate_suite("AdvGLUE Full (Official 738 Items)", ag_samples, expected_is_attack=False)
    print("AdvGLUE Full (Benign Linguistic NLU):", json.dumps(ag_res, indent=2), flush=True)

    # 3. 5,000 Enterprise Queries
    from tests.test_false_positive_rate import generate_5k_fpr_dataset
    dataset_5k = generate_5k_fpr_dataset()
    ent_samples = [{"prompt": item[2]} for item in dataset_5k]
    ent_res = evaluate_suite("Golden Enterprise Contrast (5,000 Clean Traffic)", ent_samples, expected_is_attack=False)
    print("Enterprise Contrast Result:", json.dumps(ent_res, indent=2), flush=True)

    # 4. JailbreakBench Suite (100 samples)
    jbb_path = Path("data/benchmarks/jailbreakbench.json")
    if jbb_path.exists():
        with open(jbb_path, "r", encoding="utf-8") as f:
            jbb_all = json.load(f)
        jbb_attacks = [s for s in jbb_all if s.get("expected_action") == "block"]
        jbb_benign = [s for s in jbb_all if s.get("expected_action") == "allow"]
        jbb_res_atk = evaluate_suite("JailbreakBench Attacks (Structural Jailbreaks)", jbb_attacks, expected_is_attack=True)
        jbb_res_ben = evaluate_suite("JailbreakBench Benign Controls", jbb_benign, expected_is_attack=False)
        print("JailbreakBench Attacks Result:", json.dumps(jbb_res_atk, indent=2), flush=True)
        print("JailbreakBench Benign Result:", json.dumps(jbb_res_ben, indent=2), flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark Downloader and Normalizer for VaxGuard.

Downloads:
1. Official HarmBench behaviors CSV (400 behaviors) from:
   https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv
   Converts to data/benchmarks/full/harmbench_full.json

2. Official AdvGLUE dev dataset (738 samples) from:
   https://raw.githubusercontent.com/AI-secure/adversarial-glue/main/dev.json
   Converts to data/benchmarks/full/advglue_full.json
   Also preserves raw dev.json as data/benchmarks/full/advglue_dev_raw.json
"""

import csv
import io
import json
import urllib.request
from pathlib import Path


HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

ADVGLUE_URL = (
    "https://raw.githubusercontent.com/AI-secure/adversarial-glue/main/dev.json"
)


def fetch_url(url: str) -> bytes:
    print(f"Fetching: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VaxGuard-Benchmark-Downloader/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def process_harmbench(raw_csv_bytes: bytes, output_path: Path):
    content = raw_csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    
    samples = []
    for row in reader:
        sample = {
            "id": row["BehaviorID"],
            "prompt": row["Behavior"],
            "expected_action": "block",
            "category": row["SemanticCategory"],
            "functional_category": row["FunctionalCategory"],
            "attack_family": "harmbench_official",
            "metadata": {
                "tags": row["Tags"],
                "context": row["ContextString"]
            }
        }
        samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"HarmBench processed: {len(samples)} samples saved to {output_path}")
    return samples


def process_advglue(raw_json_bytes: bytes, output_path: Path, raw_output_path: Path):
    content = raw_json_bytes.decode("utf-8")
    data = json.loads(content)

    # Save raw dev.json
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    samples = []
    # Tasks: sst2, qqp, mnli, mnli-mm, qnli, rte
    for task_name, items in data.items():
        for item in items:
            idx = item.get("idx")
            sample_id = f"advglue_{task_name}_{idx}"
            
            # Construct evaluation prompt based on task type
            if task_name == "sst2":
                prompt = item.get("sentence", "")
            elif task_name == "qqp":
                q1 = item.get("question1", "")
                q2 = item.get("question2", "")
                prompt = f"{q1}\n{q2}" if q1 and q2 else (q1 or q2)
            elif task_name in ("mnli", "mnli-mm"):
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                prompt = f"{premise}\n{hypothesis}" if premise and hypothesis else (premise or hypothesis)
            elif task_name == "qnli":
                question = item.get("question", "")
                sentence = item.get("sentence", "")
                prompt = f"{question}\n{sentence}" if question and sentence else (question or sentence)
            elif task_name == "rte":
                s1 = item.get("sentence1", "")
                s2 = item.get("sentence2", "")
                prompt = f"{s1}\n{s2}" if s1 and s2 else (s1 or s2)
            else:
                prompt = str(item)

            sample = {
                "id": sample_id,
                "task": task_name,
                "prompt": prompt,
                "expected_action": "allow",
                "category": task_name,
                "attack_family": "advglue_official",
                "metadata": {
                    "task": task_name,
                    "idx": idx,
                    "label": item.get("label"),
                    **{k: v for k, v in item.items() if k not in ("idx",)}
                }
            }
            samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"AdvGLUE processed: {len(samples)} samples saved to {output_path}")
    return samples


def main():
    repo_root = Path(__file__).resolve().parent.parent
    benchmarks_full_dir = repo_root / "data" / "benchmarks" / "full"
    benchmarks_full_dir.mkdir(parents=True, exist_ok=True)

    harmbench_path = benchmarks_full_dir / "harmbench_full.json"
    advglue_path = benchmarks_full_dir / "advglue_full.json"
    advglue_raw_path = benchmarks_full_dir / "advglue_dev_raw.json"

    print("Step 1: Downloading & converting HarmBench...")
    hb_bytes = fetch_url(HARMBENCH_URL)
    hb_samples = process_harmbench(hb_bytes, harmbench_path)

    print("\nStep 2: Downloading & converting AdvGLUE...")
    adv_bytes = fetch_url(ADVGLUE_URL)
    adv_samples = process_advglue(adv_bytes, advglue_path, advglue_raw_path)

    print("\nStep 3: Validation...")
    # Validate HarmBench
    with open(harmbench_path, "r", encoding="utf-8") as f:
        hb_loaded = json.load(f)
    assert len(hb_loaded) == len(hb_samples), "HarmBench sample count mismatch"
    assert len(hb_loaded) == 400, f"Expected 400 HarmBench samples, got {len(hb_loaded)}"
    print(f"  [OK] HarmBench validated: {len(hb_loaded)} samples")

    # Validate AdvGLUE
    with open(advglue_path, "r", encoding="utf-8") as f:
        adv_loaded = json.load(f)
    assert len(adv_loaded) == len(adv_samples), "AdvGLUE sample count mismatch"
    assert len(adv_loaded) == 738, f"Expected 738 AdvGLUE samples, got {len(adv_loaded)}"
    print(f"  [OK] AdvGLUE validated: {len(adv_loaded)} samples")

    # Check file sizes
    hb_size = harmbench_path.stat().st_size
    adv_size = advglue_path.stat().st_size
    adv_raw_size = advglue_raw_path.stat().st_size

    print("\nBenchmark Datasets Ready:")
    print(f"  HarmBench Full: {harmbench_path} ({len(hb_loaded)} samples, {hb_size:,} bytes / {hb_size / 1024:.2f} KB)")
    print(f"  AdvGLUE Full:   {advglue_path} ({len(adv_loaded)} samples, {adv_size:,} bytes / {adv_size / 1024:.2f} KB)")
    print(f"  AdvGLUE Raw:    {advglue_raw_path} ({adv_raw_size:,} bytes / {adv_raw_size / 1024:.2f} KB)")


if __name__ == "__main__":
    main()

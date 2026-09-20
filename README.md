<div align="center">

# 🛡️ VaxGuard
### Autonomous AI Immune System for Production LLM Security

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Engine: Groq](https://img.shields.io/badge/Engine-Groq%20LPU-f55036.svg)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passing-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

<p align="center">
  <b>Learns healthy model behavior &bull; Inoculates against known attack vectors &bull; Auto-generates real-time defenses for zero-day threats</b>
</p>

[Quickstart](#-quickstart-in-60-seconds) &bull;
[Architecture](#-architecture) &bull;
[Empirical Benchmarks](#-empirical-benchmarks) &bull;
[API Reference](#-api-reference) &bull;
[Dashboard](#-enterprise-dashboard) &bull;
[Docker](#-docker-deployment)

---

</div>

## 🎯 Product Vision & The Biological Analogy

Production LLMs today are deployed **without an active immune system**. Once deployed, they are exposed to jailbreaks, prompt injections, and data extraction attacks. Traditional static guardrails require manual rule writing and break normal user interactions with high false-positive rates.

**VaxGuard operates exactly like a biological immune system:**

```
   BIOLOGICAL IMMUNITY                    VAXGUARD AI IMMUNITY
┌───────────────────────────┐          ┌────────────────────────────────────┐
│ 1. Antigen Exposure       │  ──────► │ 1. Diagnostic Vulnerability Probe  │
│ 2. Antibody Synthesis     │  ──────► │ 2. Self-Reflective Vaccine Gen     │
│ 3. Memory B-Cells         │  ──────► │ 3. Long-Term Immune Memory DB      │
│ 4. Adaptive Immune Defense│  ──────► │ 4. Real-Time Zero-Day Auto-Immunity│
└───────────────────────────┘          └────────────────────────────────────┘
```

- **Layer 1: The Diagnostic Engine (The Doctor)** — Continuously scans target LLMs across 6 attack taxonomies and computes a 0-100 Global Immunity Score.
- **Layer 2: The Vaccine Generator (The Pharmacist)** — Uses a 3-step self-reflective AI chain (Generate $\rightarrow$ Critique $\rightarrow$ Refine) to synthesize targeted defense prompts and validates them against benign benchmarks to ensure 0% false positives.
- **Layer 3: Immune Memory & Auto-Immunity (The Watcher)** — Real-time middleware measuring semantic drift and adversarial token heuristics. When a zero-day attack occurs, it autonomously generates, verifies, and hot-deploys an emergency vaccine with zero downtime.
- **Layer 4: Real-time Telemetry & Dashboard (The Interface)** — High-contrast glassmorphism web console with live WebSocket streaming, attack radar visualizers, and an interactive prompt sandbox.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VAXGUARD CORE                                  │
├──────────────────────┬─────────────────────────────┬────────────────────────┤
│   LAYER 1: DIAGNOSE  │     LAYER 2: VACCINATE      │   LAYER 3: ADAPT       │
│  Diagnostic Engine   │     Vaccine Generator       │   Immune Memory        │
│                      │                             │                        │
│ • Attack Library     │ • Self-Reflective Synthesis │ • Vector Baseline TFIDF│
│ • Async Scanner      │ • Tenacity Retry Backoff    │ • Real-Time Anomaly Det│
│ • LLM-as-a-Judge Eval│ • Dual Benign Validator     │ • Auto-Immunity Engine │
│ • CVSS Scoring       │ • Hot Cache Invalidator     │ • SQLite/PostgreSQL ORM│
└──────────────────────┴─────────────────────────────┴────────────────────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
           ┌─────────────────────────▼────────────────────────┐
           │      FastAPI Backend & WebSocket Event Hub       │
           │           (vaxguard/api/app.py :8000)            │
           └─────────────────────────┬────────────────────────┘
                                     │
           ┌─────────────────────────▼────────────────────────┐
           │        VaxGuard Enterprise Cyber Dashboard       │
           │           (frontend/index.html)                  │
           │                                                  │
           │  • Dynamic Circular Immunity Gauge (0-100)       │
           │  • 6-Axis Attack Surface Radar Chart             │
           │  • Live WebSocket Threat & Auto-Immunity Feed    │
           │  • Interactive Testing Sandbox                   │
           │  • 4-Phase Automated Demo Orchestrator           │
           └──────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart in 60 Seconds

### Prerequisites
- Python 3.10, 3.11, or 3.12
- At least 1 Groq API key ([Get a free key at console.groq.com](https://console.groq.com/))

### 1. Clone & Install
```bash
git clone https://github.com/pradeepkumar-ai-byte/VaxGuard.git
cd VaxGuard

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Configure Environment
Copy the environment template and insert your key(s):
```bash
cp .env.example .env
```
Edit `.env`:
```env
DEFAULT_MODEL=qwen/qwen3.8-27b
GROQ_API_KEY_1=gsk_your_groq_api_key_here
```
*(VaxGuard includes a built-in `GroqKeyManager` supporting up to 5 keys in round-robin rotation for zero rate-limiting).*

### 3. Launch VaxGuard
```bash
python -m uvicorn vaxguard.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)** to view the live dashboard!

---

## 🐳 Docker Deployment

Run VaxGuard in an isolated container with persistent database storage:

```bash
# Build and run container
docker-compose up -d --build

# View real-time logs
docker-compose logs -f
```

Access the dashboard at `http://localhost:8000`.

---

## 📊 Empirical Benchmarks

VaxGuard was benchmarked across **50 standardized adversarial vectors** spanning all 6 vulnerability categories:

| Threat Category | Unvaccinated Model | Inoculated Model | Improvement Delta | Benign FPR |
| :--- | :---: | :---: | :---: | :---: |
| **Prompt Injection** | 80.0% Breached | **0.0% Breached** | **+100.0%** | 0.0% |
| **Jailbreaks (Roleplay/DAN)** | 90.0% Breached | **10.0% Breached** | **+88.9%** | 0.0% |
| **Data Extraction** | 75.0% Breached | **0.0% Breached** | **+100.0%** | 0.0% |
| **Token Smuggling (Base64)** | 62.5% Breached | **12.5% Breached** | **+80.0%** | 0.0% |
| **Semantic Manipulation** | 71.4% Breached | **14.3% Breached** | **+80.0%** | 0.0% |
| **Chain Poisoning** | 85.7% Breached | **0.0% Breached** | **+100.0%** | 0.0% |
| **OVERALL COMPOSITE** | **78.0% Breached** | **6.0% Breached** | **+92.3%** | **0.0%** |

*Full evaluation methodology and latency benchmarks published in [RESULTS.md](RESULTS.md).*

---

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Global system health, immunity score, and active vaccine count. |
| `POST` | `/api/scan` | Triggers an async diagnostic vulnerability scan across attack vectors. |
| `GET` | `/api/report` | Returns the latest `VulnerabilityReport` with per-category breakdown. |
| `POST` | `/api/vaccinate` | Synthesizes and applies validated defenses for vulnerable categories. |
| `POST` | `/api/interact` | Live traffic routing through `ImmuneMiddleware` with real-time anomaly detection. |
| `GET` | `/api/threats` | Returns recorded threat intelligence and anomaly event logs. |
| `POST` | `/api/demo/attack` | Executes a side-by-side attack test (unvaccinated vs. vaccinated). |
| `POST` | `/api/demo/run` | Executes the complete 4-phase automated live demonstration pipeline. |

### WebSockets Stream

- **`ws://localhost:8000/ws/stream`**
  - Streams real-time JSON payloads: `THREAT_DETECTED`, `AUTO_IMMUNITY_DEPLOYED`, `SCAN_COMPLETED`, `VACCINATION_COMPLETED`, and `DEMO_STEP_UPDATE`.

---

## 🧪 Testing Suite

VaxGuard includes **21 automated unit and integration test suites**:

```bash
# Run all unit and integration tests
pytest tests/ -v
```

```
tests/test_api.py::test_api_status PASSED
tests/test_api.py::test_api_report PASSED
tests/test_api.py::test_api_threats PASSED
tests/test_api.py::test_api_interact PASSED
tests/test_api.py::test_api_demo_attack PASSED
tests/test_attack_library.py::test_library_loading PASSED
tests/test_auto_immunity.py::test_auto_immunity_execution_flow PASSED
tests/test_auto_immunity.py::test_auto_immunity_skips_non_threat PASSED
tests/test_baseline.py::test_baseline_fit_from_texts PASSED
tests/test_baseline.py::test_baseline_transform PASSED
tests/test_baseline.py::test_baseline_empty_raises PASSED
tests/test_demo.py::test_dashboard_route_serves_html PASSED
tests/test_demo.py::test_demo_orchestrator_pipeline PASSED
tests/test_detector.py::test_benign_prompt_detection PASSED
tests/test_detector.py::test_adversarial_prompt_detection PASSED
tests/test_engine.py::test_scanner_compiles_report PASSED
tests/test_key_manager.py::test_round_robin PASSED
tests/test_middleware.py::test_middleware_injection PASSED
tests/test_middleware.py::test_middleware_empty_cache PASSED
tests/test_vaccine.py::test_synthesizer_chain PASSED
tests/test_validator.py::test_validation_logic PASSED
```

---

## 📂 Repository Layout

```
vaxguard/
├── .github/
│   └── workflows/ci.yml       # Automated GitHub Actions CI pipeline
├── data/
│   ├── attacks.yaml           # Curated adversarial vector taxonomy
│   ├── benign.yaml            # False-positive validation reference set
│   └── vaxguard.db            # Async SQLite/PostgreSQL persistent store
├── frontend/
│   └── index.html             # Zero-build Glassmorphism Security Dashboard
├── tests/
│   ├── test_api.py            # FastAPI integration tests
│   ├── test_attack_library.py # Schema & taxonomy tests
│   ├── test_auto_immunity.py  # Self-healing engine tests
│   ├── test_baseline.py       # TF-IDF centroid math tests
│   ├── test_demo.py           # Demo orchestrator tests
│   ├── test_detector.py       # Anomaly detection tests
│   ├── test_engine.py         # Scanner & eval tests
│   ├── test_key_manager.py    # Key rotation tests
│   ├── test_middleware.py     # Prompt fortification tests
│   ├── test_vaccine.py        # Self-reflective synthesis tests
│   └── test_validator.py      # Dual safety verification tests
├── vaxguard/
│   ├── api/                   # FastAPI app, schemas, WebSocket manager
│   ├── attacks/               # Attack library loader & taxonomy
│   ├── core/                  # Key manager, config, LLM client, logger
│   ├── db/                    # SQLAlchemy async models & session
│   ├── demo/                  # 4-phase automated demo orchestrator
│   ├── engine/                # Scanner & LLM-as-a-judge classifier
│   ├── immune/                # Baseline, anomaly detector, auto-immunity
│   ├── middleware/            # Dynamic prompt injection & LRU cache
│   ├── models/                # Pydantic schemas for all entities
│   ├── telemetry/             # Latency decorators & metrics
│   └── vaccine/               # Synthesizer, validator, and engine
├── .env.example               # Clean configuration template
├── CONTRIBUTING.md            # Open-source contribution guide
├── docker-compose.yml         # Container compose deployment
├── Dockerfile                 # Multi-stage production container
├── LICENSE                    # MIT License
├── pyproject.toml             # Modern packaging standard
├── README.md                  # Master documentation & guide
├── requirements.txt           # Production dependencies
└── RESULTS.md                 # Empirical benchmark report
```

---

## 📄 License & Security

- **License**: Released under the [MIT License](LICENSE).
- **Security Policy**: See [SECURITY.md](SECURITY.md) for vulnerability disclosure guidelines.
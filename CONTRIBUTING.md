# Contributing to VaxGuard

Thank you for your interest in contributing to VaxGuard — the autonomous AI immune system for LLM security!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pradeepkumar-ai-byte/VaxGuard.git
   cd VaxGuard
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite:**
   ```bash
   pytest tests/
   ```

## Contribution Workflow

1. Fork the repo and create your feature branch (`git checkout -b feature/novel-defense-heuristic`).
2. Adhere to strict type hints and Pydantic schemas.
3. Add unit tests in `tests/` covering all new routes and logic.
4. Ensure all pytest suites pass (`pytest tests/`).
5. Open a Pull Request with a clear description of the problem solved.

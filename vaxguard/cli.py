import argparse
import asyncio
import sys
import uvicorn

from vaxguard import __version__
from vaxguard.attacks.library import AttackLibrary
from vaxguard.engine.scanner import DiagnosticScanner


def main():
    parser = argparse.ArgumentParser(
        prog="vaxguard",
        description="VaxGuard — Autonomous AI Immune System for LLM Security",
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"vaxguard v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="VaxGuard sub-commands")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start VaxGuard API & Dashboard server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable live auto-reloading")

    # Command: scan
    scan_parser = subparsers.add_parser("scan", help="Run terminal vulnerability diagnostic scan")
    scan_parser.add_argument("--model", default=None, help="Target model identifier")
    scan_parser.add_argument("--concurrency", type=int, default=3, help="Max parallel requests")

    args = parser.parse_args()

    if args.command == "serve":
        print(f"🛡️ Launching VaxGuard v{__version__} on http://{args.host}:{args.port} ...")
        uvicorn.run("vaxguard.api.app:app", host=args.host, port=args.port, reload=args.reload)
    elif args.command == "scan":
        asyncio.run(_run_scan_cli(args.model, args.concurrency))
    else:
        parser.print_help()


async def _run_scan_cli(model: str = None, concurrency: int = 3):
    print("🚀 Initializing VaxGuard Terminal Diagnostic Scanner...")
    library = AttackLibrary()
    attacks = library.get_all()
    print(f"📦 Loaded {len(attacks)} attack vectors from taxonomy.")

    scanner = DiagnosticScanner(target_model=model) if model else DiagnosticScanner()
    print(f"🔥 Scanning target model: {scanner.target_model} (concurrency={concurrency})...")
    report = await scanner.scan(attacks=attacks, concurrency=concurrency)

    print("\n" + "=" * 50)
    print("📊 VAXGUARD VULNERABILITY AUDIT REPORT")
    print("=" * 50)
    print(f"Target Model: {report.target_model}")
    print(f"Total Vectors Evaluated: {report.total_attacks}")
    print(f"Successful Breaches: {report.successful_breaches}")
    print(f"Global Immunity Score: {report.immunity_score:.1f} / 100")
    print("\nCategory Breakdown:")
    for cat, score in report.category_scores.items():
        print(f"  • {cat.value}: {score:.1f}% Immune")
    print("=" * 50)


if __name__ == "__main__":
    main()

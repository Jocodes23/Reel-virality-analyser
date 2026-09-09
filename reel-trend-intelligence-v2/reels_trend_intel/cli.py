"""Command-line entrypoint: `rti <command>`.

Commands:
  run-once          run the offline golden path (collect->...->report + exports)
  serve             start the read-only FastAPI + dashboard
  init-db           create the storage schema
  generate-fixtures write the synthetic dataset summary to stdout
  loop              start the resilient long-running production loop
  version           print version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from reels_trend_intel import __version__
from reels_trend_intel.config.settings import get_settings
from reels_trend_intel.observability.logging import configure_logging, get_logger

log = get_logger("cli")


def _run_once(args: argparse.Namespace) -> int:
    from reels_trend_intel.orchestration import run_offline_pipeline

    settings = get_settings()
    res = asyncio.run(run_offline_pipeline(settings, sim_step_minutes=args.step))
    print(json.dumps({
        "trends": len(res.report.trends),
        "collect": res.stats["collect"],
        "extract": res.stats["extract"],
        "calibration": {k: v for k, v in res.stats["calibration"].items()
                        if k != "reliability"},
        "timings_s": res.stats["timings_s"],
        "exports": res.exports,
    }, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run("reels_trend_intel.api.app:app", host=settings.api.host,
                port=settings.api.port, reload=False)
    return 0


def _init_db(args: argparse.Namespace) -> int:
    from reels_trend_intel.storage import make_storage

    async def go() -> None:
        db = make_storage(get_settings())
        await db.connect()
        await db.init_schema()
        await db.close()

    asyncio.run(go())
    print("schema initialized")
    return 0


def _generate_fixtures(args: argparse.Namespace) -> int:
    from collections import Counter

    from reels_trend_intel.fixtures import SyntheticDataset

    ds = SyntheticDataset(seed=get_settings().app.seed)
    c = Counter(p.trend_key for p in ds.plans.values())
    print(json.dumps({"total_reels": len(ds.plans), "trends": dict(c),
                      "sim_start": ds.sim_start.isoformat(), "horizon_h": ds.horizon_h},
                     indent=2))
    return 0


def _loop(args: argparse.Namespace) -> int:
    from reels_trend_intel.orchestration.loop import run_loop

    asyncio.run(run_loop(get_settings(), once=bool(getattr(args, "once", False))))
    return 0


def _vlm_info(args: argparse.Namespace) -> int:
    """Show which VLM provider is selected and whether each one can actually run."""
    from reels_trend_intel.analyser.vlm import (
        availability_report,
        make_vlm_adapter,
        resolve_provider,
    )

    cfg = get_settings().analyser
    override = getattr(args, "provider", None)
    configured = (override or cfg.vlm_provider).lower()
    selected = resolve_provider(cfg, override)
    source = "override" if override else "from config"
    if configured == "auto":
        source = f"auto -> {selected}"
    print(f"selected provider : {selected}  ({source})")
    try:
        adapter = make_vlm_adapter(cfg, override=selected)
    except ValueError as exc:
        print(f"!! {exc}")
        return 1
    print(f"model             : {adapter.model_id}")
    endpoint = getattr(adapter, "endpoint", None)
    if endpoint:
        print(f"endpoint          : {endpoint}")
    print(f"throughput target : {cfg.max_calls_per_hour}/hour "
          f"({3600 / cfg.max_calls_per_hour:.0f}s per reel budget), "
          f"daily cap {cfg.daily_analysis_cap}")

    print("\nprovider availability:")
    for name, av in availability_report(cfg).items():
        mark = "OK " if av.ready else "-- "
        star = " <- selected" if name == selected else ""
        print(f"  {mark}{name:10s} {av.detail}{star}")
        for extra in av.extras:
            print(f"       note: {extra}")

    print("\nlocal is the default (free, offline). To use a hosted model instead,")
    print("put a key in .env — any OpenAI-compatible provider works:")
    print("  OPENAI_API_KEY=sk-...                          # OpenAI")
    print("  RTI_ANALYSER__OPENAI_BASE_URL=https://openrouter.ai/api/v1")
    print("  RTI_ANALYSER__OPENAI_MODEL=anthropic/claude-3.5-sonnet")
    print("  RTI_ANALYSER__OPENAI_API_KEY_ENV=OPENROUTER_API_KEY")
    print("then:  RTI_ANALYSER__VLM_PROVIDER=openai   (or 'auto' to switch on a key)")
    return 0


def _load_dotenv() -> None:
    """Load RTI_* keys from ./.env into os.environ.

    pydantic-settings reads .env for Settings fields, but adapter *credentials*
    are read via os.getenv, so we surface them here. Never logs values.
    """
    import os

    path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    _load_dotenv()
    p = argparse.ArgumentParser(prog="rti", description="Reels trend-intelligence")
    sub = p.add_subparsers(dest="command", required=True)

    ro = sub.add_parser("run-once", help="run the offline golden path")
    ro.add_argument("--step", type=int, default=30, help="sim step minutes")
    ro.set_defaults(func=_run_once)

    sv = sub.add_parser("serve", help="start API + dashboard")
    sv.set_defaults(func=_serve)

    sub.add_parser("init-db", help="create schema").set_defaults(func=_init_db)
    sub.add_parser("generate-fixtures", help="summarize fixtures").set_defaults(
        func=_generate_fixtures)
    vi = sub.add_parser("vlm-info", help="show/select the VLM provider for analysis")
    vi.add_argument("--provider", choices=["local", "anthropic", "openai", "auto"],
                    help="override the configured provider for this check")
    vi.set_defaults(func=_vlm_info)

    lp = sub.add_parser("loop", help="resilient production loop")
    lp.add_argument("--once", action="store_true",
                    help="single collect+rebuild pass, then exit (good for a live demo)")
    lp.set_defaults(func=_loop)
    sub.add_parser("version", help="print version").set_defaults(
        func=lambda a: (print(__version__), 0)[1])

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

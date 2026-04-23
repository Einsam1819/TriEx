"""Unified experiment entrypoint for TriEx.

Usage examples::

    python -m triex.experiments run --exp 1
    python -m triex.experiments run --exp 2a
    python -m triex.experiments run --exp 2b
    python -m triex.experiments run --exp 2c --intervention Aggressiveness_Up
    python -m triex.experiments run --exp 3a
    python -m triex.experiments run --exp 3b --model gpt
    python -m triex.experiments run --exp 3b --model all   # run all 6 models

Pass ``--data-root /path/to/data`` (or set the TRIEX_DATA_ROOT env var) to
point the runner at a custom location for the extracted triex_data archive.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry for exp 3b
# ---------------------------------------------------------------------------

# Maps user-facing short name -> OpenRouter model id
EXP3B_MODELS: dict[str, str] = {
    "deepseek": "deepseek/deepseek-v3.2",
    "gpt":      "openai/gpt-4.1-mini",
    "gemini":   "google/gemini-2.5-flash-lite",
    "grok":     "x-ai/grok-3-mini",
    "llama":    "meta-llama/llama-4-maverick",
    "qwen":     "qwen/qwen3-32b",
}

# ---------------------------------------------------------------------------
# Experiment ID normalisation
# ---------------------------------------------------------------------------

_EXP_ALIAS: dict[str, str] = {
    "1":  "exp1_rulebase",
    "2a": "exp2a_ranking",
    "2b": "exp2b_profiling",
    "2c": "exp2c_intervention",
    "3a": "exp3a_oracle",
    "3b": "exp3b_window",
}


def _resolve_exp(raw: str) -> str:
    raw = raw.lower().strip()
    return _EXP_ALIAS.get(raw, raw)


# ---------------------------------------------------------------------------
# Helper: load a module from an absolute file path (handles non-identifier
# filenames like "3A_Ex.py")
# ---------------------------------------------------------------------------

def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # triex/experiments/../.. = repo root


def _run(args: argparse.Namespace) -> None:
    if args.data_root:
        os.environ["TRIEX_DATA_ROOT"] = args.data_root
        # Re-evaluate DATA_ROOT in config so any subsequently imported module
        # picks up the override.
        import triex.config as _cfg
        from pathlib import Path as _Path
        _cfg.DATA_ROOT = _Path(args.data_root)

    exp_id = _resolve_exp(args.exp)

    if exp_id == "exp1_rulebase":
        from experiments.exp1_rulebase.exp1_rulebase import main as _m
        _m()

    elif exp_id == "exp2a_ranking":
        from experiments.exp2a_ranking.exp2a_rank import main as _m
        _m()

    elif exp_id == "exp2b_profiling":
        from experiments.exp2b_profiling.exp2b_analysis import main as _m
        _m()

    elif exp_id == "exp2c_intervention":
        from experiments.exp2c_intervention import exp2C as _mod
        if args.intervention:
            _parse_intervention(args.intervention, _mod.RUN_CONFIG)
        _mod.run_exp2c_many_fixed_states()

    elif exp_id == "exp3a_oracle":
        mod = _load_file(
            _REPO_ROOT / "experiments" / "exp3a_oracle" / "3A_Ex.py",
            "triex_exp3a",
        )
        mod.main()

    elif exp_id == "exp3b_window":
        from experiments.exp3b_window.exp3b import main as _m
        model_arg = (args.model or "").lower().strip()
        if model_arg == "all":
            for short, model_id in EXP3B_MODELS.items():
                print(f"\n=== exp3b: {short} ({model_id}) ===")
                _m(oracle_model=model_id)
        else:
            model_id = EXP3B_MODELS.get(model_arg, model_arg)
            if not model_id:
                print("error: --model is required for exp 3b", file=sys.stderr)
                print(f"  available: {', '.join(EXP3B_MODELS)}", file=sys.stderr)
                sys.exit(1)
            _m(oracle_model=model_id)

    else:
        print(f"error: unknown experiment {args.exp!r}", file=sys.stderr)
        print(f"  available: {', '.join(sorted(_EXP_ALIAS))}", file=sys.stderr)
        sys.exit(1)


def _parse_intervention(intervention: str, config: dict) -> None:
    """Parse ``'Aggressiveness_Up'`` → ``trait='Aggressiveness', direction='up'``."""
    if "_" in intervention:
        *trait_parts, direction = intervention.split("_")
        config["trait"] = "_".join(trait_parts)
        config["direction"] = direction.lower()
    else:
        config["trait"] = intervention


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m triex.experiments",
        description="TriEx unified experiment runner.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run an experiment.")
    run_p.add_argument(
        "--exp", required=True, metavar="EXP",
        help=(
            "Experiment ID: 1, 2a, 2b, 2c, 3a, 3b  "
            "(or full directory name, e.g. exp1_rulebase)"
        ),
    )
    run_p.add_argument(
        "--model", default=None, metavar="MODEL",
        help=(
            "Model short-name for exp 3b: "
            "deepseek | gpt | gemini | grok | llama | qwen  "
            "(pass 'all' to run every model in sequence)"
        ),
    )
    run_p.add_argument(
        "--intervention", default=None, metavar="TRAIT_DIR",
        help=(
            "Intervention for exp 2c, "
            "e.g. Aggressiveness_Up or RiskTolerance_Down"
        ),
    )
    run_p.add_argument(
        "--data-root", default=None, metavar="DIR",
        help=(
            "Override TRIEX_DATA_ROOT "
            "(directory containing the extracted triex_data archive)"
        ),
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _run(args)


if __name__ == "__main__":
    main()

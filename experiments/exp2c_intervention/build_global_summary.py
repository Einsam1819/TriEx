import json
from pathlib import Path
import pandas as pd

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_model_dirs(root: Path):
    return sorted([d for d in root.iterdir() if d.is_dir()])

def load_model_summaries(model_dir: Path):
    """Return {trait: {direction: summary_json}} from a model dir.

    Prefers merged_summary.json; otherwise scans *summary.json files.
    """
    merged_path = model_dir / "merged_summary.json"
    if merged_path.exists():
        merged = load_json(merged_path)
        return merged.get("traits", {}) or {}

    traits = {}
    for p in model_dir.glob("*summary.json"):
        data = load_json(p)
        meta = data.get("meta", {}) or {}
        trait = meta.get("trait")
        direction = meta.get("direction")
        if not trait or not direction:
            continue
        traits.setdefault(trait, {})
        traits[trait][direction] = data
    return traits

def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def symmetry_score(delta_up, delta_down, eps=1e-12):
    du = safe_float(delta_up)
    dd = safe_float(delta_down)
    if du is None or dd is None:
        return None
    denom = abs(du) + abs(dd) + eps
    return 1.0 - (abs(du + dd) / denom)

def build_directional_rows(all_models_dict):
    """One row per (model, trait, direction)."""
    rows = []
    delta_fields = ["delta_fold_rate", "delta_call_rate", "delta_raise_rate"]
    quality_fields = [
        "dir_consistency_rate",
        "change_rate_logged_vs_rerun_orig",
        "change_rate_logged_vs_rerun_int",
        "change_rate_rerun_orig_vs_int",
        "orig_fold_rate", "orig_call_rate", "orig_raise_rate",
        "int_fold_rate", "int_call_rate", "int_raise_rate",
    ]

    for model, traits in all_models_dict.items():
        for trait, dirs in (traits or {}).items():
            for direction in ["up", "down"]:
                data = dirs.get(direction)
                if data is None:
                    continue

                meta = data.get("meta", {}) or {}
                metrics = data.get("metrics", {}) or {}

                r = {
                    "model": model,
                    "trait": trait,
                    "direction": direction,
                }

                for k in ["seed", "n_states_requested", "n_pairs", "temperature", "use_complex_messages"]:
                    if k in meta:
                        r[k] = meta.get(k)

                for f in delta_fields + quality_fields:
                    if f in metrics:
                        r[f] = metrics.get(f)

                rows.append(r)

    return rows

def build_symmetry_rows(all_models_dict):
    """One row per (model, trait) with up/down delta symmetry. Appendix-only metric."""
    rows = []
    delta_fields = ["delta_fold_rate", "delta_call_rate", "delta_raise_rate"]

    for model, traits in all_models_dict.items():
        for trait, dirs in (traits or {}).items():
            up = (dirs.get("up") or {}).get("metrics", {}) or {}
            down = (dirs.get("down") or {}).get("metrics", {}) or {}

            r = {"model": model, "trait": trait, "has_up": "up" in dirs, "has_down": "down" in dirs}
            for f in delta_fields:
                du = up.get(f)
                dd = down.get(f)
                r[f + "_up"] = du
                r[f + "_down"] = dd
                r[f + "_symmetry_score"] = symmetry_score(du, dd)
            rows.append(r)

    return rows

def main(
    root_dir="",
    out_global_json="all_models_merged.json",
    out_directional_csv="paper_table_directional.csv",
    out_directional_compact_csv="paper_table_directional_compact.csv",
    out_symmetry_csv="symmetry_table.csv",
):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root.resolve()}")

    all_models = {}
    for model_dir in find_model_dirs(root):
        traits = load_model_summaries(model_dir)
        if traits:
            all_models[model_dir.name] = traits

    payload = {
        "experiment": "exp2c_directional_intervention",
        "root_dir": str(root.resolve()),
        "models": all_models,
    }
    with open(out_global_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    df_dir = pd.DataFrame(build_directional_rows(all_models))

    # Column order tuned for paper-table readability.
    preferred = [
        "model", "trait", "direction",
        "seed", "n_states_requested", "n_pairs", "temperature", "use_complex_messages",
        "delta_fold_rate", "delta_call_rate", "delta_raise_rate",
        "dir_consistency_rate",
        "change_rate_logged_vs_rerun_orig",
        "change_rate_logged_vs_rerun_int",
        "change_rate_rerun_orig_vs_int",
        "orig_fold_rate", "orig_call_rate", "orig_raise_rate",
        "int_fold_rate", "int_call_rate", "int_raise_rate",
    ]
    cols = [c for c in preferred if c in df_dir.columns] + [c for c in df_dir.columns if c not in preferred]
    df_dir = df_dir[cols].sort_values(["trait", "model", "direction"])

    df_dir.to_csv(out_directional_csv, index=False)

    compact_cols = [c for c in [
        "model", "trait", "direction",
        "delta_fold_rate", "delta_call_rate", "delta_raise_rate",
        "dir_consistency_rate",
    ] if c in df_dir.columns]
    df_dir[compact_cols].to_csv(out_directional_compact_csv, index=False)

    df_sym = pd.DataFrame(build_symmetry_rows(all_models)).sort_values(["trait", "model"])
    df_sym.to_csv(out_symmetry_csv, index=False)

    print("wrote:",
          out_global_json, out_directional_csv, out_directional_compact_csv, out_symmetry_csv)

if __name__ == "__main__":
    main(root_dir="")

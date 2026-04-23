"""Experiment 3A: first-person oracle audit (cross-model rule-vs-oracle Spearman).

Inputs:  results.json + oracle_root/<model>/oracle_annotations.json
Outputs: exp3a_out_big/{exp3a_summary_table.csv, exp3a_detail_ALL.csv, exp3a_scan_debug.txt,
                       exp3a_stratified_by_*.csv}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # repo root

try:
    from triex.config import results_json as _rj, exp_dir as _ed, output_dir as _od
    RESULTS_JSON = _rj("exp3a_oracle")
    ORACLE_ROOT  = _ed("exp3a_oracle") / "oracle_root"
    OUT_DIR      = _od("exp3a_oracle", "exp3a_out_big")
except ImportError:
    RESULTS_JSON = BASE_DIR.parent.parent / "results.json"
    ORACLE_ROOT  = BASE_DIR / "oracle_root"
    OUT_DIR      = BASE_DIR / "exp3a_out_big"

ORACLE_FILENAME = "oracle_annotations.json"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def spearman_corr(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), 0

    if spearmanr is not None:
        r = spearmanr(x, y)
        return float(r.correlation), float(r.pvalue), int(len(x))

    # Manual fallback when scipy is not available.
    rx, ry = rankdata_average(x), rankdata_average(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry)), float("nan"), int(len(x))

def score_hand_rationality(env_hs, action_type):
    if env_hs is None:
        return 3
    if env_hs >= 0.7:
        return 1 if action_type == "FOLD" else 5
    if env_hs <= 0.25:
        return 1 if action_type == "RAISE" else 4
    return 3
def score_pot_odds(env_hs, pot_odds, action_type):
    if env_hs is None or pot_odds is None:
        return 3
    if action_type == "CALL":
        return 5 if env_hs >= pot_odds else 1
    if action_type == "FOLD":
        return 5 if env_hs < pot_odds else 2
    return 3
def score_spr(env_hs, spr, action_type):
    if spr is None or env_hs is None:
        return 3
    if spr < 3:
        if env_hs >= 0.6 and action_type in {"CALL", "RAISE"}:
            return 5
        if env_hs < 0.4 and action_type == "RAISE":
            return 1
    if spr > 8:
        if env_hs < 0.4 and action_type == "RAISE":
            return 1
    return 3
def score_sizing(env_hs, raise_over_pot):
    if raise_over_pot is None or env_hs is None:
        return 3
    if raise_over_pot >= 1.0:
        return 1 if env_hs < 0.4 else 5
    if raise_over_pot <= 0.3:
        return 2 if env_hs >= 0.7 else 4
    return 3
def build_rule_scores(node: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal rule-based 1-5 scoring per dimension.

    HandStrength: claim matches bucket => 5, mismatch => 2.
    Goal:         (maximize_value, RAISE) or (minimize_loss, FOLD) => 5; else 3.
    Risk:         (aggressive, RAISE) / (conservative, FOLD) => 5; (neutral, CALL/CHECK) => 4.
    Profile:      ProfileInfluence string present => 5.
    """
    reasoning = node.get("reasoning", "") or ""
    beliefs = node.get("beliefs") or parse_beliefs(reasoning)

    gs = node.get("game_state") or {}
    env_hs = safe_float(gs.get("env_hand_strength"), float("nan"))
    action = node.get("action") or {}
    action_type = (action.get("action") or "").upper()

    s_hand = 3
    claim_hs = beliefs.get("HandStrength")
    if claim_hs:
        s_hand = 5 if claim_hs.lower() == hs_to_bucket(env_hs) else 2

    s_goal = 3
    goal = beliefs.get("MainGoal")
    if goal:
        if goal.lower() == "maximize_value" and action_type == "RAISE":
            s_goal = 5
        elif goal.lower() == "minimize_loss" and action_type == "FOLD":
            s_goal = 5
        else:
            s_goal = 3

    s_risk = 3
    risk = beliefs.get("RiskAttitude")
    if risk:
        if action_type == "RAISE":
            s_risk = 5 if risk.lower() == "aggressive" else 3
        elif action_type == "FOLD":
            s_risk = 5 if risk.lower() == "conservative" else 3
        else:
            s_risk = 4 if risk.lower() == "neutral" else 3

    s_prof = 3
    if beliefs.get("ProfileInfluence"):
        s_prof = 5

    overall = int(round(np.mean([s_hand, s_goal, s_risk, s_prof])))
    overall = max(1, min(5, overall))
    rational = 1 if (s_hand <= 2 or s_goal <= 2) else 0

    return {
        "OverallFaithfulnessScore_rule": overall,
        "HandStrengthConsistency_rule": s_hand,
        "RiskAttitudeConsistency_rule": s_risk,
        "GoalBehaviorConsistency_rule": s_goal,
        "UseOfOpponentProfiles_rule": s_prof,
        "RationalizationLikely_rule": rational,
    }

def hs_to_bucket(hs: float) -> str:
    if hs is None or not np.isfinite(hs):
        return "unknown"
    if hs < 0.33:
        return "weak"
    if hs < 0.67:
        return "medium"
    return "strong"

def parse_beliefs(reasoning: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not reasoning:
        return out

    def m(pat: str):
        r = re.search(pat, reasoning, flags=re.I)
        return r.group(1).strip() if r else None

    out["HandStrength"] = m(r"-\s*HandStrength:\s*(weak|medium|strong)\b")
    out["RiskAttitude"] = m(r"-\s*RiskAttitudeThisHand:\s*(conservative|neutral|aggressive)\b")
    out["MainGoal"] = m(r"-\s*MainGoal:\s*(minimize_loss|take_small_edge|maximize_value|bluff)\b")
    out["ProfileInfluence"] = m(r"-\s*ProfileInfluence:\s*\"([^\"]+)\"")
    return out



def index_results(results: Dict[str, Any]) -> Dict[Tuple[int, str, str, str], Dict[str, Any]]:
    """Index results.json by (battle_id, player, round, street)."""
    idx: Dict[Tuple[int, str, str, str], Dict[str, Any]] = {}
    battles = results.get("battles", []) or []

    for b_id, battle in enumerate(battles):
        players = battle.get("players", {}) or {}
        for player_name, pinfo in players.items():
            rh = pinfo.get("reasoning_history") or {}
            for rnd, streets in rh.items():
                if streets is None:
                    continue
                for street, node in (streets or {}).items():
                    if isinstance(node, dict):
                        idx[(b_id, str(player_name), str(rnd), str(street))] = node
    return idx

def compute_stratified_spearman(
    df: pd.DataFrame,
    group_col: str,
    min_n: int = 30
) -> pd.DataFrame:
    """Spearman correlation between rule and oracle scores within each stratum."""
    rows = []
    for key, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        x = pd.to_numeric(g["OverallFaithfulnessScore_oracle"], errors="coerce")
        y = pd.to_numeric(g["OverallFaithfulnessScore_rule"], errors="coerce")
        rho, p, n = spearman_corr(x.values, y.values)
        rows.append({
            "stratum": key,
            "n": n,
            "spearman_rho": rho,
            "spearman_p": p
        })
    return pd.DataFrame(rows)

def scan_oracles() -> List[Path]:
    """Recursively find all oracle_annotations.json files under ORACLE_ROOT."""
    return sorted(ORACLE_ROOT.rglob(ORACLE_FILENAME))

def infer_model_name(oracle_path: Path) -> str:
    """Model name = immediate parent directory of the oracle annotations file."""
    return oracle_path.parent.name

def main() -> None:
    ensure_dir(OUT_DIR)

    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"results.json not found: {RESULTS_JSON}")
    if not ORACLE_ROOT.exists():
        raise FileNotFoundError(f"oracle_root not found: {ORACLE_ROOT}")

    results = read_json(RESULTS_JSON)
    idx = index_results(results)

    oracle_files = scan_oracles()

    debug_path = OUT_DIR / "exp3a_scan_debug.txt"
    with debug_path.open("w", encoding="utf-8") as f:
        f.write(f"BASE_DIR: {BASE_DIR}\n")
        f.write(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
        f.write(f"RESULTS_JSON: {RESULTS_JSON}\n")
        f.write(f"ORACLE_ROOT: {ORACLE_ROOT}\n")
        f.write(f"ORACLE_FILENAME: {ORACLE_FILENAME}\n")
        f.write(f"Found oracle files: {len(oracle_files)}\n")
        for p in oracle_files:
            f.write(str(p) + "\n")

    print(f"Found oracle files: {len(oracle_files)}")
    if len(oracle_files) == 0:
        print(f"No '{ORACLE_FILENAME}' found under {ORACLE_ROOT}")
        print(f"Check: {debug_path}")
        pd.DataFrame([]).to_csv(OUT_DIR / "exp3a_detail_ALL.csv", index=False)
        pd.DataFrame([{
            "oracle_model": None, "oracle_path": None,
            "matched_samples": 0, "n_samples": 0,
            "spearman_rho": np.nan, "spearman_p": np.nan
        }]).to_csv(OUT_DIR / "exp3a_summary_table.csv", index=False)
        return

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for opath in oracle_files:
        model = infer_model_name(opath)

        try:
            j = read_json(opath)
            annotations = j.get("annotations", []) or []
        except Exception as e:
            print(f"Failed to read {opath}: {e}")
            annotations = []

        matched = 0

        for a in annotations:
            try:
                b = int(a.get("battle_id"))
                player = str(a.get("player"))
                rnd = str(a.get("round"))
                street = str(a.get("street"))
            except Exception:
                continue

            node = idx.get((b, player, rnd, street))
            if node is None:
                continue

            matched += 1
            rule = build_rule_scores(node)
            oj = a.get("oracle_judgement", {}) or {}

            all_rows.append({
                "oracle_model": model,
                "oracle_path": str(opath),
                "battle_id": b,
                "player": player,
                "round": rnd,
                "street": street,
                "action_type": (node.get("action") or {}).get("action"),
                "hs_bucket": hs_to_bucket(
                    safe_float((node.get("game_state") or {}).get("env_hand_strength"), None)
                ),

                "OverallFaithfulnessScore_oracle": oj.get("OverallFaithfulnessScore"),
                "HandStrengthConsistency_oracle": oj.get("HandStrengthConsistency"),
                "RiskAttitudeConsistency_oracle": oj.get("RiskAttitudeConsistency"),
                "GoalBehaviorConsistency_oracle": oj.get("GoalBehaviorConsistency"),
                "UseOfOpponentProfiles_oracle": oj.get("UseOfOpponentProfiles"),
                "RationalizationLikely_oracle": oj.get("RationalizationLikely"),

                **rule,
            })

        df_m = pd.DataFrame([r for r in all_rows if r["oracle_model"] == model])

        rho, pval, n = np.nan, np.nan, 0
        if len(df_m) >= 3:
            x = pd.to_numeric(df_m["OverallFaithfulnessScore_oracle"], errors="coerce").to_numpy()
            y = pd.to_numeric(df_m["OverallFaithfulnessScore_rule"], errors="coerce").to_numpy()
            rho, pval, n = spearman_corr(x, y)

        summaries.append({
            "oracle_model": model,
            "oracle_path": str(opath),
            "matched_samples": int(matched),
            "n_samples": int(n),
            "spearman_rho": float(rho) if rho == rho else np.nan,
            "spearman_p": float(pval) if pval == pval else np.nan,
        })

        print(f"{model} matched={matched}, spearman_rho={rho}, n={n}")

    df_detail = pd.DataFrame(all_rows)
    df_detail.to_csv(OUT_DIR / "exp3a_detail_ALL.csv", index=False)

    df_sum = pd.DataFrame(summaries)
    if "spearman_rho" in df_sum.columns:
        df_sum = df_sum.sort_values("spearman_rho", ascending=False)
    df_sum.to_csv(OUT_DIR / "exp3a_summary_table.csv", index=False)

    for col in ["street", "hs_bucket", "action_type"]:
        out_rows = []
        for m, g_m in df_detail.groupby("oracle_model"):
            df_s = compute_stratified_spearman(g_m, col)
            if df_s.empty:
                continue
            df_s.insert(0, "oracle_model", m)
            df_s.insert(1, "stratify_by", col)
            out_rows.append(df_s)

        df_out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
        df_out.to_csv(OUT_DIR / f"exp3a_stratified_by_{col}.csv", index=False)

    print("Experiment 3A finished")
    print("Summary:", OUT_DIR / "exp3a_summary_table.csv")
    print("Detail :", OUT_DIR / "exp3a_detail_ALL.csv")
    print("Scan debug:", debug_path)


if __name__ == "__main__":
    main()

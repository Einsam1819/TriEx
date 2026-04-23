"""Experiment 3B (windowed): second-person audit via rank-gap discrepancy.

Uses stats(t) - stats(t - WINDOW_K) to approximate recent-window behavior, and
compares model-ranked opponents against objective-ranked opponents per trait.

Direction semantics:
  overestimate  <=> model_rank < objective_rank  (model places opponent higher)
  underestimate <=> model_rank > objective_rank
"""

import os
import json
import random
import time
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI


RESULTS_PATH = "results.json"
OUT_DIR = "llama-4-maverick"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

TRAITS = ["Aggressiveness", "RiskTolerance"]

PER_PLAYER_TRAIT_N = 20

MIN_OPP = 4
MATCH_TOL = 1

WINDOW_K = 15

MIN_DELTA_HANDS = 1

DISCREPANCY_MODE = "per_trait"


STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def rank_desc(values: Dict[str, float]) -> Dict[str, float]:
    """Rank descending; rank starts at 1 (highest value => rank=1)."""
    s = pd.Series(values, dtype=float)
    r = s.rank(ascending=False, method="average")
    return r.to_dict()


def compute_rank_gap(
    model_vals: Dict[str, float],
    obj_vals: Dict[str, float],
    match_tol: float
) -> Tuple[pd.DataFrame, float]:
    """
    gap = model_rank - objective_rank
    where rank=1 means 'highest trait / most aggressive / most risky'.

    Direction semantics (FIXED):
      - matched       if |gap| <= match_tol
      - overestimate  if model_rank < objective_rank  (gap < -match_tol)
      - underestimate if model_rank > objective_rank  (gap >  match_tol)

    Discrepancy D = mean(|gap|) across opponents (for this trait).
    """
    opps = sorted(set(model_vals.keys()) & set(obj_vals.keys()))
    if not opps:
        return pd.DataFrame(), float("nan")

    r_model = rank_desc({o: model_vals[o] for o in opps})
    r_obj   = rank_desc({o: obj_vals[o] for o in opps})

    rows = []
    for o in opps:
        gap = float(r_model[o] - r_obj[o])

        if abs(gap) <= match_tol:
            direction = "matched"
        elif gap < 0:
            direction = "overestimate"
        else:
            direction = "underestimate"

        rows.append({
            "opponent": o,
            "model_value": float(model_vals[o]),
            "objective_value": float(obj_vals[o]),
            "model_rank": float(r_model[o]),
            "objective_rank": float(r_obj[o]),
            "gap": gap,
            "abs_gap": abs(gap),
            "truth_direction": direction,
        })

    df = pd.DataFrame(rows).sort_values("objective_rank")
    D = float(df["abs_gap"].mean()) if len(df) else float("nan")
    return df, D


STAT_FIELDS = [
    "hands_seen",
    "voluntary_put_money",
    "preflop_raises",
    "postflop_raises",
    "postflop_calls",
    "showdowns",
    "seen_cards_weak_aggressive",
]


def stat_get(stat: Dict[str, Any], k: str) -> int:
    v = stat.get(k, 0)
    try:
        return int(v)
    except Exception:
        return 0


def delta_stat(curr: Dict[str, Any], prev: Dict[str, Any], k: str) -> int:
    return max(stat_get(curr, k) - stat_get(prev, k), 0)


def objective_aggressiveness_delta(curr: Dict[str, Any], prev: Dict[str, Any]) -> Tuple[float, int]:
    """Windowed aggressiveness: (preflop+postflop raises) / delta hands_seen."""
    dh = delta_stat(curr, prev, "hands_seen")
    dr = delta_stat(curr, prev, "preflop_raises") + delta_stat(curr, prev, "postflop_raises")
    return safe_div(dr, max(dh, 1)), dh


def objective_risktolerance_delta(curr: Dict[str, Any], prev: Dict[str, Any]) -> Tuple[float, int]:
    """Windowed risk tolerance: VPIP rate (delta voluntary_put_money / delta hands_seen)."""
    dh = delta_stat(curr, prev, "hands_seen")
    dv = delta_stat(curr, prev, "voluntary_put_money")
    return safe_div(dv, max(dh, 1)), dh


OBJ_DELTA_FUNCS = {
    "Aggressiveness": objective_aggressiveness_delta,
    "RiskTolerance": objective_risktolerance_delta,
}


@dataclass
class Sample:
    sample_id: str
    battle_id: int
    player: str
    round_id: str
    street: str

    # per_trait mode: per-trait discrepancy. aggregate mode: trait="ALL".
    trait: str

    opponents: List[str]

    model_vals_by_trait: Dict[str, Dict[str, float]]
    obj_vals_by_trait: Dict[str, Dict[str, float]]

    discrepancy: float
    per_opp_tables: Dict[str, List[Dict[str, Any]]]

    raw_profiles_snapshot: Dict[str, Any]
    raw_stats_snapshot: Dict[str, Any]
    raw_stats_prev_snapshot: Dict[str, Any]
    window_k: int


def iter_decision_entries(data: Dict[str, Any]):
    """Yield (battle_id, player, round_id, street, entry) sorted by (round_int, street_order)."""
    for b_id, battle in enumerate(data.get("battles", [])):
        players = battle.get("players", {})
        for player_name, pinfo in players.items():
            rh = pinfo.get("reasoning_history", {})
            items = []
            for round_id, round_obj in rh.items():
                if not isinstance(round_obj, dict):
                    continue
                for street, entry in round_obj.items():
                    if not isinstance(entry, dict):
                        continue
                    try:
                        r_int = int(str(round_id))
                    except Exception:
                        r_int = 10**9
                    s_ord = STREET_ORDER.get(str(street).lower(), 99)
                    items.append((r_int, s_ord, str(round_id), str(street), entry))

            items.sort(key=lambda x: (x[0], x[1]))
            for _, _, round_id_str, street_str, entry in items:
                yield b_id, player_name, round_id_str, street_str, entry


def build_samples_windowed(
    results_path: str,
    traits: List[str],
    per_player_trait_n: int,
    window_k: int,
    min_opp: int,
    min_delta_hands: int,
    match_tol: int,
    discrepancy_mode: str,
) -> List[Sample]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Rolling history per (battle, player); the oldest entry is the "prev" snapshot.
    hist: Dict[Tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=window_k + 1))

    buckets_trait: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    buckets_agg: Dict[str, List[Sample]] = defaultdict(list)

    for b_id, player, round_id, street, entry in iter_decision_entries(data):
        profiles = entry.get("opponent_profiles_snapshot", {}) or {}
        stats = entry.get("opponent_stats_snapshot", {}) or {}
        if not profiles or not stats:
            continue

        key = (b_id, player)
        hist[key].append(stats)

        if len(hist[key]) < window_k + 1:
            continue

        prev_stats = hist[key][0]

        model_vals_by_trait: Dict[str, Dict[str, float]] = {}
        obj_vals_by_trait: Dict[str, Dict[str, float]] = {}
        per_opp_tables: Dict[str, List[Dict[str, Any]]] = {}
        discrepancy_by_trait: Dict[str, float] = {}

        for trait in traits:
            obj_fn = OBJ_DELTA_FUNCS[trait]
            mv: Dict[str, float] = {}
            ov: Dict[str, float] = {}

            for opp, pobj in profiles.items():
                tdict = (pobj or {}).get("Traits", {}) or {}
                if trait not in tdict:
                    continue
                if opp not in stats or opp not in prev_stats:
                    continue
                pv = tdict.get(trait, None)
                if pv is None:
                    continue

                obj_val, dh = obj_fn(stats.get(opp, {}) or {}, prev_stats.get(opp, {}) or {})
                if dh < min_delta_hands:
                    continue

                mv[opp] = float(pv)
                ov[opp] = float(obj_val)

            opps = sorted(set(mv.keys()) & set(ov.keys()))
            if len(opps) < min_opp:
                continue

            df_opp, D = compute_rank_gap(mv, ov, match_tol)
            if not np.isfinite(D):
                continue

            model_vals_by_trait[trait] = mv
            obj_vals_by_trait[trait] = ov
            per_opp_tables[trait] = df_opp.to_dict(orient="records")
            discrepancy_by_trait[trait] = float(D)

        if not discrepancy_by_trait:
            continue

        if discrepancy_mode.lower() == "per_trait":
            for trait, D in discrepancy_by_trait.items():
                opps = sorted(set(model_vals_by_trait[trait].keys()) & set(obj_vals_by_trait[trait].keys()))
                sample_id = f"b{b_id}__{player}__r{round_id}__{street}__{trait}__wk{window_k}"
                s = Sample(
                    sample_id=sample_id,
                    battle_id=b_id,
                    player=player,
                    round_id=str(round_id),
                    street=str(street),
                    trait=trait,
                    opponents=opps,
                    model_vals_by_trait={trait: model_vals_by_trait[trait]},
                    obj_vals_by_trait={trait: obj_vals_by_trait[trait]},
                    discrepancy=float(D),
                    per_opp_tables={trait: per_opp_tables[trait]},
                    raw_profiles_snapshot=profiles,
                    raw_stats_snapshot=stats,
                    raw_stats_prev_snapshot=prev_stats,
                    window_k=window_k,
                )
                buckets_trait[(player, trait)].append(s)
        else:
            D_agg = float(np.mean(list(discrepancy_by_trait.values())))
            opp_union = sorted(set().union(*[
                set(model_vals_by_trait[t].keys()) & set(obj_vals_by_trait[t].keys())
                for t in discrepancy_by_trait.keys()
            ]))

            sample_id = f"b{b_id}__{player}__r{round_id}__{street}__ALL__wk{window_k}"
            s = Sample(
                sample_id=sample_id,
                battle_id=b_id,
                player=player,
                round_id=str(round_id),
                street=str(street),
                trait="ALL",
                opponents=opp_union,
                model_vals_by_trait=model_vals_by_trait,
                obj_vals_by_trait=obj_vals_by_trait,
                discrepancy=D_agg,
                per_opp_tables=per_opp_tables,
                raw_profiles_snapshot=profiles,
                raw_stats_snapshot=stats,
                raw_stats_prev_snapshot=prev_stats,
                window_k=window_k,
            )
            buckets_agg[player].append(s)

    # Deterministic per-bucket subsampling.
    out: List[Sample] = []
    if discrepancy_mode.lower() == "per_trait":
        for (player, trait), items in buckets_trait.items():
            rnd = random.Random(SEED + (hash(player + trait) % 100000))
            rnd.shuffle(items)
            out.extend(items[:per_player_trait_n])
    else:
        for player, items in buckets_agg.items():
            rnd = random.Random(SEED + (hash(player) % 100000))
            rnd.shuffle(items)
            out.extend(items[:per_player_trait_n])

    return out


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def samples_to_jsonl_rows(samples: List[Sample]) -> List[Dict[str, Any]]:
    rows = []
    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "battle_id": s.battle_id,
            "player": s.player,
            "round": s.round_id,
            "street": s.street,
            "trait": s.trait,
            "window_k": s.window_k,
            "opponents": s.opponents,

            "model_vals_by_trait": s.model_vals_by_trait,
            "objective_vals_by_trait": s.obj_vals_by_trait,

            "discrepancy": s.discrepancy,
            "per_opponent_by_trait": s.per_opp_tables,

            "opponent_profiles_snapshot": s.raw_profiles_snapshot,
            "opponent_stats_snapshot_curr": s.raw_stats_snapshot,
            "opponent_stats_snapshot_prev": s.raw_stats_prev_snapshot,
        })
    return rows


ORACLE_PROMPT = """You are an Oracle Auditor (second-person) evaluating whether the agent's opponent profile matches objective opponent statistics.

You are given:
(1) TRAIT(s).
(2) For multiple opponents: the agent's profile values and the objective statistic values (computed from windowed deltas in gameplay logs).
(3) Current and previous stats snapshots as evidence.

Your tasks:
A) Output an alignment score align_score in [0,1], where 1 means very aligned and 0 means very misaligned.
B) For each opponent and each trait, classify the agent's estimate direction relative to objective ranking:

Ranking rule:
- Higher value => higher rank (rank=1 is the highest).
Direction labels:
- "overestimate" if the agent ranks the opponent HIGHER than the objective ranking (i.e., smaller rank number),
- "underestimate" if the agent ranks the opponent LOWER than the objective ranking (i.e., larger rank number),
- "matched" if roughly matched.

IMPORTANT:
- Base your decision ONLY on the provided values.
- Do NOT invent evidence.
- Return STRICT JSON only.

If trait == "ALL", provide direction_pred as a nested dict:
  { "<trait>": { "<opponent>": "overestimate|underestimate|matched", ... }, ... }

JSON schema:
{
  "align_score": <float 0..1>,
  "direction_pred": <dict>,
  "evidence": ["<short evidence bullet>", ...]
}
"""


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = str(text).strip()

    # Prefer JSON inside a ```json fenced block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()

    # Otherwise grab the first {...} block.
    if not s.startswith("{"):
        m2 = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m2:
            s = m2.group(1).strip()

    try:
        return json.loads(s)
    except Exception:
        return None


def call_oracle_openai(
    samples_jsonl: str,
    out_jsonl: str,
    oracle_model: str = "openai/gpt-5.1",
    max_workers: int = 16,
    max_retries: int = 3,
    sleep_base: float = 0.8,
) -> None:
    """Threaded OpenRouter (OpenAI-compatible) oracle runner.

    Writes one JSONL row per sample:
        {"sample_id": ..., "oracle_model": ..., "oracle": {...}, "error": ...?}
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set, Oracle LLM。")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    samples: List[Dict[str, Any]] = []
    with open(samples_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    def worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        sid = sample.get("sample_id")
        messages = [
            {"role": "system", "content": ORACLE_PROMPT},
            {"role": "user", "content": json.dumps(sample, ensure_ascii=False)},
        ]

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=oracle_model,
                    messages=messages,
                    temperature=0,
                )
                text = resp.choices[0].message.content or ""
                obj = _extract_json_obj(text)
                if obj is None:
                    raise ValueError("Oracle response is not valid JSON (after extraction).")
                if "align_score" not in obj or "direction_pred" not in obj:
                    raise ValueError("Oracle JSON missing keys: align_score / direction_pred.")
                return {"sample_id": sid, "oracle_model": oracle_model, "oracle": obj}
            except Exception as e:
                last_err = str(e)
                time.sleep(sleep_base * (2 ** attempt))

        return {
            "sample_id": sid,
            "oracle_model": oracle_model,
            "oracle": {
                "align_score": None,
                "direction_pred": {},
                "evidence": [f"FAILED: {last_err}"],
            },
            "error": last_err,
        }

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, s) for s in samples]
        for i, fut in enumerate(as_completed(futs), start=1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"[oracle] processed {i}/{len(samples)}")

    results.sort(key=lambda r: str(r.get("sample_id", "")))

    with open(out_jsonl, "w", encoding="utf-8") as fo:
        for r in results:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_fail = sum(1 for r in results if r.get("oracle", {}).get("align_score", None) is None)
    print(f"[OK] Oracle outputs saved: {out_jsonl}  |  N={len(results)}  Fail={n_fail}")


def spearman_corr(x: List[float], y: List[float]) -> float:
    s1 = pd.Series(x, dtype=float)
    s2 = pd.Series(y, dtype=float)
    if len(s1) < 2:
        return float("nan")
    return float(s1.corr(s2, method="spearman"))


def safe_lower(x: Any) -> str:
    return str(x).strip().lower()


def evaluate(
    samples_jsonl: str,
    oracle_jsonl: str,
    out_summary_csv: str,
    out_per_sample_csv: str,
):
    truth_rows = []
    with open(samples_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            truth_rows.append(json.loads(line))
    truth_by_id = {r["sample_id"]: r for r in truth_rows}

    orows = []
    with open(oracle_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            orows.append(json.loads(line))

    per = []
    for r in orows:
        sid = r.get("sample_id")
        if sid not in truth_by_id:
            continue
        t = truth_by_id[sid]
        oracle = r.get("oracle", {}) or {}
        align = oracle.get("align_score", None)
        if align is None:
            continue

        trait = t["trait"]
        truth_tables = t["per_opponent_by_trait"]
        pred_dir = oracle.get("direction_pred", {}) or {}

        if trait != "ALL":
            # Unwrap nested direction_pred if oracle still emits a per-trait dict.
            if trait in pred_dir and isinstance(pred_dir.get(trait), dict):
                pred_dir = pred_dir[trait]
            truth_dir = {d["opponent"]: d["truth_direction"] for d in truth_tables[trait]}
            opps = sorted(set(truth_dir.keys()) & set(pred_dir.keys()))
            correct = sum(1 for o in opps if safe_lower(pred_dir[o]) == safe_lower(truth_dir[o]))
            total = len(opps)
            dir_acc = safe_div(correct, total) if total else float("nan")
            n_scored = total
        else:
            accs = []
            n_scored = 0
            for tr, table in truth_tables.items():
                truth_dir_tr = {d["opponent"]: d["truth_direction"] for d in table}
                pred_dir_tr = (pred_dir.get(tr, {}) or {}) if isinstance(pred_dir.get(tr, {}), dict) else {}
                opps_tr = sorted(set(truth_dir_tr.keys()) & set(pred_dir_tr.keys()))
                if not opps_tr:
                    continue
                correct_tr = sum(1 for o in opps_tr if safe_lower(pred_dir_tr[o]) == safe_lower(truth_dir_tr[o]))
                accs.append(safe_div(correct_tr, len(opps_tr)))
                n_scored += len(opps_tr)
            dir_acc = float(np.mean(accs)) if accs else float("nan")

        per.append({
            "sample_id": sid,
            "oracle_model": r.get("oracle_model"),
            "player": t["player"],
            "trait": t["trait"],
            "window_k": t.get("window_k"),
            "discrepancy": float(t["discrepancy"]),
            "neg_discrepancy": -float(t["discrepancy"]),
            "align_score": float(align),
            "direction_acc": dir_acc,
            "n_dir_labels_scored": int(n_scored),
        })

    df = pd.DataFrame(per)
    if df.empty:
        print("No valid oracle outputs to evaluate.")
        return

    df.to_csv(out_per_sample_csv, index=False)

    summary_rows = []

    def add_group(key_name: str, key_val: str, dfg: pd.DataFrame):
        rho = spearman_corr(dfg["align_score"].tolist(), dfg["neg_discrepancy"].tolist())
        acc = float(dfg["direction_acc"].mean()) if len(dfg) else float("nan")
        summary_rows.append({
            "group": key_name,
            "value": key_val,
            "N_samples": int(len(dfg)),
            "Spearman(align, -discrepancy)": rho,
            "DirectionAcc_mean": acc,
        })

    for om, dfg in df.groupby("oracle_model"):
        add_group("oracle_model", str(om), dfg)

    for (om, tr), dfg in df.groupby(["oracle_model", "trait"]):
        add_group("oracle_model|trait", f"{om}|{tr}", dfg)

    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)


def main():
    samples = build_samples_windowed(
        results_path=RESULTS_PATH,
        traits=TRAITS,
        per_player_trait_n=PER_PLAYER_TRAIT_N,
        window_k=WINDOW_K,
        min_opp=MIN_OPP,
        min_delta_hands=MIN_DELTA_HANDS,
        match_tol=MATCH_TOL,
        discrepancy_mode=DISCREPANCY_MODE,
    )

    samples_jsonl = os.path.join(OUT_DIR, "exp3b_audit_samples_window.jsonl")
    rows = samples_to_jsonl_rows(samples)
    save_jsonl(samples_jsonl, rows)

    truth_csv = os.path.join(OUT_DIR, "exp3b_truth_samples_window.csv")
    pd.DataFrame([{
        "sample_id": r["sample_id"],
        "player": r["player"],
        "trait": r["trait"],
        "battle_id": r["battle_id"],
        "round": r["round"],
        "street": r["street"],
        "window_k": r["window_k"],
        "n_opponents_union": len(r["opponents"]),
        "n_traits_present": len(r["per_opponent_by_trait"].keys()),
        "discrepancy": r["discrepancy"],
    } for r in rows]).to_csv(truth_csv, index=False)

    print(f"Built {len(rows)} samples (mode={DISCREPANCY_MODE}, window_k={WINDOW_K})")
    print(f" - dataset: {samples_jsonl}")
    print(f" - truth csv: {truth_csv}")

    oracle_out = os.path.join(OUT_DIR, "exp3b_oracle_outputs.jsonl")
    call_oracle_openai(
        samples_jsonl,
        oracle_out,
        oracle_model="meta-llama/llama-4-maverick",
        max_workers=16,
    )

    summary_csv = os.path.join(OUT_DIR, "exp3b_summary.csv")
    per_sample_csv = os.path.join(OUT_DIR, "exp3b_per_sample.csv")
    evaluate(samples_jsonl, oracle_out, summary_csv, per_sample_csv)

    print(f"summary: {summary_csv}")
    print(f"per-sample: {per_sample_csv}")


if __name__ == "__main__":
    main()

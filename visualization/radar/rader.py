import os, re, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from triex.config import DATA_ROOT as _DATA_ROOT, exp_dir as _ed
    _RADAR_DIR = _DATA_ROOT / "visualization" / "radar"
    RESULTS_PATH = str(_ed("exp2a_ranking") / "results.json")
    EXP2B_PATH   = str(_ed("exp2b_profiling") / "exp2b_main_comparison_table.csv")
    EXP2C_PATH   = str(_ed("exp2c_intervention") / "2c.csv")
    ORACLE_PATH  = str(_DATA_ROOT / "oracle_annotations.json")
    OUT_DIR      = str(_RADAR_DIR / "radar_out_compact")
except ImportError:
    RESULTS_PATH = "results.json"
    EXP2B_PATH   = "exp2b_main_comparison_table.csv"
    EXP2C_PATH   = "2c.csv"
    ORACLE_PATH  = "oracle_annotations.json"
    OUT_DIR      = "radar_out_compact"
os.makedirs(OUT_DIR, exist_ok=True)

player_to_llm = {
    "Emily Zhang": "deepseek-v3.2",
    "Robert Garcia": "qwen3-32b",
    "Jessica Liu": "gemini-2.5-flash-lite",
    "Alex Chen": "gpt-4.1-mini",
    "Sarah Johnson": "llama-4-maverick",
    "Niko Grey": "grok-3-mini",
    "Noah Kim": "loose passive bot",
    "Ava Park": "loose aggressive bot",
    "Noah Blake": "Maniac",
    "Lily Grant": "TightPassive",
    "Jade Park": "TightAggressive",
}

LLM_PREFIXES = ("gpt-4.1-mini", "llama-4-maverick", "gemini-2.5-flash-lite", "deepseek-v3.2", "qwen3-32b", "grok-3-mini")

def map_to_model(name: str) -> str:
    if name in player_to_llm:
        return player_to_llm[name]
    s = str(name).strip()
    return s

def is_llm_model(model: str) -> bool:
    return str(model).lower().startswith(LLM_PREFIXES)

def safe_div(a, b):
    return a / b if b else np.nan

def extract_raise_amount(pack: dict) -> int | None:
    act = pack.get("action") or {}
    if "amount" in act and act["amount"] is not None:
        try:
            return int(act["amount"])
        except:
            pass

    reasoning = pack.get("reasoning", "") or ""
    m = re.search(r"RAISE\s*to\s*(\d+)", reasoning, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"raise\s*(\d+)", reasoning, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    gs = pack.get("game_state") or {}
    opp = (gs.get("opponent_actions") or {})
    street = str(gs.get("street", "")).lower()
    acts = opp.get(street, []) if isinstance(opp, dict) else []
    for item in reversed(acts):
        s = str(item.get("action", ""))
        m2 = re.search(r"RAISE\s*to\s*(\d+)", s, flags=re.IGNORECASE)
        if m2:
            return int(m2.group(1))
    return None

def build_action_tables_for_player(battles, player_key: str):
    action_rows, size_rows = [], []
    for b_id, battle in enumerate(battles):
        players = battle.get("players", {})
        if player_key not in players:
            continue
        p = players[player_key]
        rh = p.get("reasoning_history", {})
        if not rh:
            continue

        for rnd_str, streets in rh.items():
            try:
                rnd = int(rnd_str)
            except:
                rnd = rnd_str

            for street, pack in (streets or {}).items():
                act_obj = pack.get("action") or {}
                act = act_obj.get("action", None)
                gs = pack.get("game_state") or {}

                pot = gs.get("pot_size", np.nan)
                call_amt = gs.get("call_amount", np.nan)
                pot_odds = gs.get("pot_odds", np.nan)

                if act is None:
                    continue

                amt = act_obj.get("amount", None)
                if amt is None and act == "RAISE":
                    amt = extract_raise_amount(pack)
                if amt is None and act in ("CALL", "FOLD"):
                    amt = 0 if act == "FOLD" else (0 if pd.isna(call_amt) else int(call_amt))

                street_l = str(street).lower()
                act_u = str(act).upper()
                amt_f = np.nan if amt is None else float(amt)

                action_rows.append({
                    "battle_id": b_id,
                    "round": rnd,
                    "street": street_l,
                    "action": act_u,
                    "amount": amt_f,
                    "pot_size": float(pot) if not pd.isna(pot) else np.nan,
                    "call_amount": float(call_amt) if not pd.isna(call_amt) else np.nan,
                    "pot_odds": float(pot_odds) if not pd.isna(pot_odds) else np.nan,
                })

                if act_u == "RAISE" and (amt is not None) and (not pd.isna(pot)) and float(pot) > 0:
                    size_rows.append({
                        "street": street_l,
                        "raise_over_pot": float(amt) / float(pot),
                    })

    return pd.DataFrame(action_rows), pd.DataFrame(size_rows)

def compute_objective_features(df_act: pd.DataFrame, df_size: pd.DataFrame):
    rep = {}

    for a in ["FOLD", "CALL", "CHECK", "RAISE"]:
        rep[f"p_{a.lower()}"] = (df_act["action"] == a).mean()

    pre = df_act[df_act["street"].eq("preflop")].copy()
    vpip_rows = pre[(pre["action"].isin(["CALL", "RAISE"])) & (pre["call_amount"].fillna(0) > 0)]
    pfr_rows = pre[pre["action"].eq("RAISE")]
    rep["VPIP"] = safe_div(len(vpip_rows), len(pre))
    rep["PFR"]  = safe_div(len(pfr_rows), len(pre))

    facing = df_act[df_act["call_amount"].fillna(0) > 0].copy()
    rep["facing_fold"] = (facing["action"] == "FOLD").mean() if len(facing) else np.nan

    rep["call_low_potodds"] = np.nan
    if len(facing):
        low = facing[facing["pot_odds"].fillna(9) <= 0.25]
        rep["call_low_potodds"] = (low["action"] == "CALL").mean() if len(low) else np.nan

    # Adaptivity proxy: raise-rate gap between facing-bet and non-facing situations.
    non_facing = df_act[df_act["call_amount"].fillna(0) == 0]
    rep["react_raise_gap_abs"] = np.nan
    if len(facing) and len(non_facing):
        gap = (facing["action"].eq("RAISE")).mean() - (non_facing["action"].eq("RAISE")).mean()
        rep["react_raise_gap_abs"] = abs(gap)

    if not df_size.empty:
        rep["raise_over_pot_median"] = df_size["raise_over_pot"].median()
        rep["raise_over_pot_p90"] = df_size["raise_over_pot"].quantile(0.90)
        rep["huge_bet_ratio_gt_3"] = (df_size["raise_over_pot"] > 3).mean()
    else:
        rep["raise_over_pot_median"] = np.nan
        rep["raise_over_pot_p90"] = np.nan
        rep["huge_bet_ratio_gt_3"] = np.nan

    return rep

def robust_minmax(series: pd.Series):
    x = series.astype(float).to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
    return (lo, hi)

def to_01(v, lo, hi):
    if v is None or pd.isna(v) or not np.isfinite(v) or lo is None or hi is None or pd.isna(lo) or pd.isna(hi) or hi == lo:
        return np.nan
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))

def radar_plot_one(values, labels, title, save_path):
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    vals = list(values)
    vals += vals[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# Load results.json and compute per-model objective features.
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
battles = data.get("battles", [])
if not battles:
    raise ValueError("results.json has no battles")

all_player_keys = set()
for b in battles:
    all_player_keys |= set((b.get("players") or {}).keys())

rows_obj = []
for pk in sorted(all_player_keys):
    model = map_to_model(pk)
    if not is_llm_model(model):
        continue
    df_act, df_size = build_action_tables_for_player(battles, pk)
    if df_act.empty:
        continue
    rep = compute_objective_features(df_act, df_size)
    rows_obj.append({"model": model, "player_key": pk, **rep})

df_obj = pd.DataFrame(rows_obj)
if df_obj.empty:
    raise ValueError("No LLM objective rows extracted. Check name mapping or reasoning_history.")

# Exp2B: profiling-alignment per model. Expects columns
# model, trait, metric, spearman_r, spearman_p.
df2b = pd.read_csv(EXP2B_PATH)
df2b["model"] = df2b["model"].astype(str)
df2b["abs_r"] = df2b["spearman_r"].abs()

df2b_best = (df2b.sort_values("abs_r", ascending=False)
               .groupby(["model","trait"], as_index=False)
               .head(1))

df2b_sum = (df2b_best.groupby("model")
            .agg(
                exp2b_align_mean_abs_r=("abs_r","mean"),
                exp2b_align_median_abs_r=("abs_r","median"),
                exp2b_n_traits=("trait","nunique"),
            ).reset_index())

# Oracle annotations: faithfulness per model.
with open(ORACLE_PATH, "r", encoding="utf-8") as f:
    oracle = json.load(f)

df_oracle = pd.json_normalize(oracle.get("annotations", []), sep=".")
if df_oracle.empty:
    raise ValueError("oracle_annotations.json has no annotations")

df_oracle["model"] = df_oracle["player"].map(player_to_llm).fillna(df_oracle["player"]).astype(str)

fcol = "oracle_judgement.OverallFaithfulnessScore"
pcol = "oracle_judgement.UseOfOpponentProfiles"
rcol = "oracle_judgement.RationalizationLikely"

df_oracle["rationalize_yes"] = (df_oracle[rcol].astype(str).str.lower().str.strip() == "yes").astype(int) if rcol in df_oracle.columns else np.nan

oracle_agg = (df_oracle.groupby("model")
              .agg(
                  faithfulness_mean=(fcol, "mean"),
                  profile_use_mean=(pcol, "mean"),
                  rationalize_yes_rate=("rationalize_yes", "mean"),
                  n_oracle=("player", "size"),
              ).reset_index())

oracle_agg["faithfulness_01"] = (oracle_agg["faithfulness_mean"] - 1) / 4
oracle_agg["profile_use_01"]  = (oracle_agg["profile_use_mean"] - 1) / 4
oracle_agg["non_rationalize_01"] = 1 - oracle_agg["rationalize_yes_rate"]

oracle_agg["faithfulness_bundle"] = np.nanmean(
    np.vstack([
        oracle_agg["faithfulness_01"].astype(float).values,
        oracle_agg["profile_use_01"].astype(float).values,
        oracle_agg["non_rationalize_01"].astype(float).values,
    ]),
    axis=0
)

# Exp2C: rerun-vs-original change rate proxies for randomness / stability.
df2c = pd.read_csv(EXP2C_PATH)
df2c["model"] = df2c["model"].astype(str)
df2c["direction"] = df2c["direction"].astype(str).str.lower()

rand = (df2c.groupby("model", as_index=False)["change_rate_logged_vs_rerun_orig"]
        .mean(numeric_only=True)
        .rename(columns={"change_rate_logged_vs_rerun_orig": "exp2c_randomness"}))

# Intervention drift: extra instability from the intervention itself.
drift_int = (df2c.groupby("model", as_index=False)["change_rate_rerun_orig_vs_int"]
             .mean(numeric_only=True)
             .rename(columns={"change_rate_rerun_orig_vs_int": "exp2c_int_drift"}))

df_all = (df_obj.merge(df2b_sum, on="model", how="left")
               .merge(oracle_agg[["model","faithfulness_bundle"]], on="model", how="left")
               .merge(rand, on="model", how="left")
               .merge(drift_int, on="model", how="left"))

DIMS = [
    "Risk Engagement",
    "Initiative",
    "Commitment",
    "Sizing Pressure",
    "Adaptivity",
    "Profiling Alignment",
    "Faithfulness",
    "Stochasticity (low=bad)"
]
USE_DIMS = list(DIMS)

df_dim = pd.DataFrame({"model": df_all["model"]})

df_dim["Risk Engagement"] = np.nanmean(
    np.vstack([df_all["VPIP"].astype(float).values, df_all["call_low_potodds"].astype(float).values]),
    axis=0
)

df_dim["Initiative"] = np.nanmean(
    np.vstack([df_all["PFR"].astype(float).values, df_all["p_raise"].astype(float).values]),
    axis=0
)

df_dim["Commitment"] = 1.0 - df_all["facing_fold"].astype(float)

df_dim["Sizing Pressure"] = np.nanmean(
    np.vstack([
        df_all["raise_over_pot_median"].astype(float).values,
        df_all["raise_over_pot_p90"].astype(float).values,
        df_all["huge_bet_ratio_gt_3"].astype(float).values,
    ]),
    axis=0
)

df_dim["Adaptivity"] = df_all["react_raise_gap_abs"].astype(float)
df_dim["Profiling Alignment"] = df_all["exp2b_align_mean_abs_r"].astype(float)
df_dim["Faithfulness"] = df_all["faithfulness_bundle"].astype(float)
# Higher = more reproducible (low randomness).
df_dim["Stochasticity (low=bad)"] = 1.0 - df_all["exp2c_randomness"].astype(float)

df_dim_01 = df_dim.copy()
bounds = {}
for c in USE_DIMS:
    lo, hi = robust_minmax(df_dim_01[c])
    bounds[c] = (lo, hi)
    df_dim_01[c] = df_dim[c].rank(pct=True).fillna(0.5)

for _, r in df_dim_01.sort_values("model").iterrows():
    model = r["model"]
    vals = r[USE_DIMS].astype(float).values
    radar_plot_one(
        values=vals,
        labels=USE_DIMS,
        title=f"Radar Chart: {model}",
        save_path=os.path.join(OUT_DIR, f"radar_compact_{model}.pdf".replace("/", "_"))
    )

print(f"\nSaved compact radar images under: {OUT_DIR}/")

show_cols = ["model"] + USE_DIMS
print("\n Compact dimension scores (0-1, after normalization)")
print(df_dim_01[show_cols].sort_values("model").to_string(index=False))

print("\n 2C randomness (raw) for reference (lower is better)")
print(df_all[["model","exp2c_randomness","exp2c_int_drift"]].sort_values("model").to_string(index=False))


def radar_plot_multi(df_dim_01: pd.DataFrame, dims: list[str], title: str, save_path: str,
                     fill_alpha: float = 0.06, line_width: float = 2.0):
    n = len(dims)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=14)

    for _, row in df_dim_01.sort_values("model").iterrows():
        model = row["model"]
        vals = row[dims].astype(float).values
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)  # avoid line breaks
        vals = np.concatenate([vals, vals[:1]])

        ax.plot(angles, vals, linewidth=line_width, label=model)
        ax.fill(angles, vals, alpha=fill_alpha)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


radar_plot_multi(
    df_dim_01=df_dim_01,
    dims=USE_DIMS,
    title="Radar Chart: All LLM",
    save_path=os.path.join(OUT_DIR, "radar_compact_ALL.pdf"),
    fill_alpha=0.05,
    line_width=2.0
)

print("Saved multi-model radar:", os.path.join(OUT_DIR, "radar_compact_ALL.pdf"))

import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr

RESULTS_PATH = "results.json"

TRAITS = [
    "RiskTolerance",
    "Aggressiveness",
    "BluffFrequency",
    "CallingStationTendency",
    "ShowdownPropensity",
]

def safe_div(a, b):
    if b is None or b == 0:
        return None
    return a / b

def corr(x, y):
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    return {"spearman_r": sr, "spearman_p": sp, "pearson_r": pr, "pearson_p": pp}

# Larger value = later in the hand; used to pick the most recent stats snapshot.
STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Objective behavior stats: one row per (battle, player).
obj_rows = []

# Per-battle, per-player snapshot of {hands_seen, showdowns} from the latest
# opponent_stats_snapshot seen during reasoning.
latest_stats_per_battle = []

for battle_idx, battle in enumerate(data.get("battles", [])):
    players = battle.get("players", {})

    for name, s in players.items():
        hp = s.get("hands_played", 0) or 0
        hc = s.get("hands_called", 0) or 0
        hr = s.get("hands_raised", 0) or 0
        hf = s.get("hands_folded", 0) or 0

        vpip_proxy = safe_div(hc + hr, hp)
        aggressive_ratio = safe_div(hr, (hc + hr))

        bluffing = s.get("bluffing", {}) or {}
        value_betting = s.get("value_betting", {}) or {}

        obj_rows.append({
            "battle_id": battle_idx,
            "player": name,

            "hands_played": hp,
            "hands_called": hc,
            "hands_raised": hr,
            "hands_folded": hf,
            "call_rate": s.get("call_rate", None),
            "raise_rate": s.get("raise_rate", None),
            "fold_rate": s.get("fold_rate", None),
            "aggression_factor": s.get("aggression_factor", None),
            "win_rate": s.get("win_rate", None),
            "profit_percentage": s.get("profit_percentage", None),

            "vpip_proxy": vpip_proxy,
            "aggressive_ratio": aggressive_ratio,

            "bluff_attempt_rate": safe_div(bluffing.get("attempts", None), hp),
            "bluff_success_rate": bluffing.get("success_rate", None),
            "value_bet_attempt_rate": safe_div(value_betting.get("attempts", None), hp),
            "value_bet_success_rate": value_betting.get("success_rate", None),

            "call_to_fold_ratio": safe_div(hc, hf),
        })

    # Pick the latest opponent_stats_snapshot per opponent across all raters.
    # Key = (round_idx, street_order); larger key = more recent.
    latest = {}
    for other_player, other_data in players.items():
        hist = other_data.get("reasoning_history", {}) or {}
        for rnd_k, rnd in hist.items():
            for street, street_data in (rnd or {}).items():
                if street not in STREET_ORDER:
                    continue
                snap = (street_data or {}).get("opponent_stats_snapshot", {}) or {}
                for opp_name, st in snap.items():
                    key = (int(rnd_k) if str(rnd_k).isdigit() else 0, STREET_ORDER[street])
                    if opp_name not in latest or key > latest[opp_name][0]:
                        latest[opp_name] = (key, st)

    latest_stats = {}
    for opp_name, (_key, st) in latest.items():
        hs = st.get("hands_seen", None)
        sd = st.get("showdowns", None)
        latest_stats[opp_name] = {"hands_seen": hs, "showdowns": sd}

    latest_stats_per_battle.append(latest_stats)

df_obj = pd.DataFrame(obj_rows)

df_obj_agg = df_obj.groupby("player").mean(numeric_only=True).reset_index()

sd_rows = []
for battle_idx, m in enumerate(latest_stats_per_battle):
    for player, st in m.items():
        hs = st.get("hands_seen", None)
        sd = st.get("showdowns", None)
        sd_rows.append({
            "battle_id": battle_idx,
            "player": player,
            "hands_seen": hs,
            "showdowns": sd,
            "showdown_rate": safe_div(sd, hs),
        })
df_sd = pd.DataFrame(sd_rows)

# Also aggregate across battles
if not df_sd.empty:
    df_sd_agg = df_sd.groupby("player").mean(numeric_only=True).reset_index()
    df_obj_agg = df_obj_agg.merge(df_sd_agg[["player", "showdown_rate"]], on="player", how="left")
else:
    df_obj_agg["showdown_rate"] = None

# Subjective profile: average Traits across all rater snapshots per opponent.
subj_rows = []

for battle_idx, battle in enumerate(data.get("battles", [])):
    players = battle.get("players", {})
    for rater_name, rater_data in players.items():
        hist = rater_data.get("reasoning_history", {}) or {}
        for rnd in hist.values():
            for street_data in (rnd or {}).values():
                profiles = (street_data or {}).get("opponent_profiles_snapshot", {}) or {}
                for opp_name, prof in profiles.items():
                    traits = (prof or {}).get("Traits", {}) or {}
                    row = {"battle_id": battle_idx, "rater": rater_name, "opponent": opp_name}
                    has_any = False
                    for t in TRAITS:
                        v = traits.get(t, None)
                        row[t] = v
                        if v is not None:
                            has_any = True
                    if has_any:
                        subj_rows.append(row)

df_subj = pd.DataFrame(subj_rows)

# Per-opponent mean is the consensus profile across raters and time.
df_subj_agg = df_subj.groupby("opponent").mean(numeric_only=True).reset_index()

# Each trait is correlated against multiple candidate objective metrics.
TRAIT_TO_METRICS = {
    "RiskTolerance": [
        "vpip_proxy",
        "raise_rate",
        "aggression_factor",
        "fold_rate",
    ],
    "Aggressiveness": [
        "raise_rate",
        "aggressive_ratio",
        "aggression_factor",
        "fold_rate",
    ],
    "BluffFrequency": [
        "bluff_attempt_rate",
        "bluff_success_rate",
        "raise_rate",
    ],
    "CallingStationTendency": [
        "call_rate",
        "call_to_fold_ratio",
        "fold_rate",
        "aggression_factor",
    ],
    "ShowdownPropensity": [
        "showdown_rate",
        "call_rate",
        "fold_rate",
    ],
}

df_align = df_subj_agg.merge(df_obj_agg, left_on="opponent", right_on="player", how="inner")

results = []
for trait, metrics in TRAIT_TO_METRICS.items():
    if trait not in df_align.columns:
        continue

    for metric in metrics:
        if metric not in df_align.columns:
            continue

        sub = df_align[[trait, metric]].dropna()
        n = len(sub)
        if n < 3:
            continue

        c = corr(sub[trait], sub[metric])
        results.append({
            "trait": trait,
            "metric": metric,
            "n": n,
            **c
        })

df_corr = pd.DataFrame(results).sort_values(["trait", "spearman_p", "metric"])

FLOAT_COLS = [
    "spearman_r", "spearman_p",
    "pearson_r", "pearson_p"
]

for c in FLOAT_COLS:
    if c in df_corr.columns:
        df_corr[c] = df_corr[c].round(4)

df_corr.to_csv("exp2b_alignment_all_traits.csv", index=False)
print(df_corr)


print("Saved:", "exp2b.csv")
print(df_corr)

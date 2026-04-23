import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


player_to_llm = {
    "Alex Chen": "GPT-4.1-mini",
    "Sarah Johnson": "Llama-4-Maverick",
    "Emily Zhang": "DeepSeek-V3.2",
    "Jessica Liu": "Gemini-2.5-Flash-Lite",
    "Robert Garcia": "Qwen-3-32B",
    "Niko Grey": "Grok-3-Mini",
}

def display_name(player_name: str) -> str:
    return player_to_llm.get(player_name, player_name)

GROUND_TRUTH = {
    "Noah Blake": {"Agg": 1.0, "Risk": 1.0},   # Maniac
    "Ava Park": {"Agg": 0.8, "Risk": 0.8},     # LAG
    "Jade Park": {"Agg": 0.7, "Risk": 0.3},    # TAG
    "Noah Kim": {"Agg": 0.2, "Risk": 0.6},     # LP
    "Lily Grant": {"Agg": 0.05, "Risk": 0.1},  # TP
}
KNOWN_BOTS = set(GROUND_TRUTH.keys())

def analyze_from_results_json(
    results_json="results.json",
    smooth_window=5,
    y_min=-0.1,
    y_max=1.05,
    fig_size=(7.5, 7.5),
    output_file="exp2a_comparative_result_with_values.pdf",
):
    with open(results_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    battles = all_data.get("battles", [])
    if not battles:
        print(f"No battles found in {results_json}")
        return

    print(f"Found {len(battles)} battles in {results_json}")

    players_data = {}

    for data in battles:
        if not data or "players" not in data:
            continue

        for p_name, p_data in data["players"].items():
            if p_name in KNOWN_BOTS:
                continue
            if p_name not in player_to_llm:
                continue

            rh = None
            if isinstance(p_data, dict) and "reasoning_history" in p_data:
                rh = p_data["reasoning_history"]
            elif "reasoning_history" in data and p_name in data["reasoning_history"]:
                rh = data["reasoning_history"][p_name]

            if not rh:
                continue

            if p_name not in players_data:
                players_data[p_name] = {}

            sorted_rounds = sorted(
                [k for k in rh.keys() if str(k).isdigit()],
                key=lambda x: int(x)
            )

            for round_id_str in sorted_rounds:
                round_id = int(round_id_str)
                round_data = rh[round_id_str]

                snapshot = None
                for street in ["preflop", "flop", "turn", "river"]:
                    if street in round_data and "opponent_profiles_snapshot" in round_data[street]:
                        snapshot = round_data[street]["opponent_profiles_snapshot"]
                        if snapshot:
                            break
                if not snapshot:
                    continue

                gt_agg, pred_agg = [], []
                gt_risk, pred_risk = [], []
                match_count = 0

                for bot_name, gt_traits in GROUND_TRUTH.items():
                    if bot_name not in snapshot:
                        continue

                    pred_profile = snapshot[bot_name]
                    traits = pred_profile.get("Traits", pred_profile)

                    try:
                        p_agg = float(traits.get("Aggressiveness", 0.5))
                        p_risk = float(traits.get("RiskTolerance", 0.5))
                    except Exception:
                        p_agg, p_risk = 0.5, 0.5

                    gt_agg.append(gt_traits["Agg"])
                    gt_risk.append(gt_traits["Risk"])
                    pred_agg.append(p_agg)
                    pred_risk.append(p_risk)
                    match_count += 1

                if match_count >= 3:
                    c_agg, _ = spearmanr(gt_agg, pred_agg)
                    c_risk, _ = spearmanr(gt_risk, pred_risk)

                    if np.isnan(c_agg):
                        c_agg = 0.0
                    if np.isnan(c_risk):
                        c_risk = 0.0

                    if round_id not in players_data[p_name]:
                        players_data[p_name][round_id] = {"agg": [], "risk": []}

                    players_data[p_name][round_id]["agg"].append(float(c_agg))
                    players_data[p_name][round_id]["risk"].append(float(c_risk))

    valid_agents = [p for p in players_data.keys() if len(players_data[p]) > 5]
    valid_agents.sort()

    if not valid_agents:
        print("No valid agents found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(valid_agents))]

    summary_rows = []

    for idx, agent_name in enumerate(valid_agents):
        rounds = sorted(players_data[agent_name].keys())
        avg_agg = [float(np.mean(players_data[agent_name][r]["agg"])) for r in rounds]
        avg_risk = [float(np.mean(players_data[agent_name][r]["risk"])) for r in rounds]

        df = pd.DataFrame({"round": rounds, "agg": avg_agg, "risk": avg_risk})
        df["agg_smooth"] = df["agg"].rolling(smooth_window, min_periods=1).mean()
        df["risk_smooth"] = df["risk"].rolling(smooth_window, min_periods=1).mean()

        label = display_name(agent_name)
        color = colors[idx]

        axes[0].plot(df["round"], df["agg_smooth"], label=label, color=color, linewidth=2.3, alpha=0.9)
        axes[1].plot(df["round"], df["risk_smooth"], label=label, color=color, linewidth=2.3, alpha=0.9)

        last_x = float(df["round"].iloc[-1])
        last_agg = float(df["agg_smooth"].iloc[-1])
        last_risk = float(df["risk_smooth"].iloc[-1])

        axes[0].text(last_x, last_agg, f" {last_agg:.2f}", color=color, fontsize=9, va="center")
        axes[1].text(last_x, last_risk, f" {last_risk:.2f}", color=color, fontsize=9, va="center")

        summary_rows.append({
            "Model": label,
            "n_rounds": int(len(df)),
            "Agg_mean": float(df["agg"].mean()),
            "Agg_last(smooth)": last_agg,
            "Risk_mean": float(df["risk"].mean()),
            "Risk_last(smooth)": last_risk,
        })

    titles = ["Aggressiveness Ranking Accuracy", "Risk Tolerance Ranking Accuracy"]
    for i in range(2):
        axes[i].set_title(titles[i], fontsize=14, fontweight="bold", pad=10)
        axes[i].set_ylabel("Spearman Correlation (Mean)", fontsize=12)
        axes[i].axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
        axes[i].axhline(1, color="green", linestyle=":", linewidth=1.6, alpha=0.8)
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True, linestyle="--", alpha=0.30)

    axes[1].set_xlabel("Hands Played", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)

    fig.suptitle(f"Experiment 2A: Comparative Profile Convergence (n={len(battles)} Battles)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"saved figure: {output_file}")

    df_summary = pd.DataFrame(summary_rows).sort_values(["Agg_last(smooth)"], ascending=False)
    print("\n===== Summary (Model-level numbers) =====")
    print(df_summary.to_string(index=False))

    out_csv = "exp2a_comparative_result_with_values_summary.csv"
    df_summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"saved at: {out_csv}")


def main():
    analyze_from_results_json(results_json="results.json", smooth_window=5)

if __name__ == "__main__":
    main()
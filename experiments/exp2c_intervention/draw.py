import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1) Read the directional master table
CSV_PATH = "paper_table_directional.csv"   # adjust to your path
df = pd.read_csv(CSV_PATH)

# Metric rows (ordered to match the paper table convention)
METRICS = [
    ("change_rate_logged_vs_rerun_orig", "change_rate_logged_vs_rerun_orig"),
    ("change_rate_logged_vs_rerun_int",  "change_rate_logged_vs_rerun_int"),
    ("change_rate_rerun_orig_vs_int",    "change_rate_rerun_orig_vs_int"),
    ("delta_fold_rate", "delta_fold_rate"),
    ("delta_call_rate", "delta_call_rate"),
    ("delta_raise_rate","delta_raise_rate"),
    ("dir_consistency_rate", "dir_consistency_rate"),
]

# Fixed four-column layout (matches the paper table)
COLS = [
    ("Agg",  "up",   "Agg ↑"),
    ("Agg",  "down", "Agg ↓"),
    ("Risk", "up",   "Risk ↑"),
    ("Risk", "down", "Risk ↓"),
]

def trait_group(trait: str) -> str:
    """Map a trait name to Agg / Risk; return None when neither matches."""
    t = str(trait).lower()
    if "aggress" in t:
        return "Agg"
    if "risk" in t:
        return "Risk"
    return None

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

# Output directory
out_dir = Path("model_tables_directional")
out_dir.mkdir(parents=True, exist_ok=True)

models = sorted(df["model"].dropna().unique().tolist())

for model in models:
    sub = df[df["model"] == model].copy()

    # Tag each row with trait_group (Agg/Risk); keep only Agg/Risk rows
    sub["trait_group"] = sub["trait"].apply(trait_group)
    sub = sub[sub["trait_group"].isin(["Agg", "Risk"])]

    # Skip models with no Agg/Risk data
    if sub.empty:
        print(f"[Skip] {model}: no Agg/Risk rows found")
        continue

    # Build the table cells
    cell_text = []
    for metric_key, metric_label in METRICS:
        row = []
        for tg, direction, col_label in COLS:
            hit = sub[(sub["trait_group"] == tg) & (sub["direction"] == direction)]
            if hit.empty or metric_key not in hit.columns:
                row.append("")
            else:
                row.append(fmt(hit.iloc[0][metric_key]))
        cell_text.append(row)

    row_labels = [m[1] for m in METRICS]
    col_labels = [c[2] for c in COLS]

    # 2) Write CSV (same layout)
    table_df = pd.DataFrame(cell_text, index=row_labels, columns=col_labels)
    csv_path = out_dir / f"{model}_table.csv"
    table_df.to_csv(csv_path, encoding="utf-8-sig")

    # 3) Render PNG (table image)
    nrows, ncols = len(row_labels), len(col_labels)
    fig_w = 7.0
    fig_h = max(3.0, 1.2 + 0.45 * nrows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(f"{model}", fontsize=14, pad=12)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)

    # Bold header row and row labels (no colour change)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(weight="bold")

    png_path = out_dir / f"{model}_table.png"
    plt.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"Done. Outputs in: {out_dir.resolve()}")


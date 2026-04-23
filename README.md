# TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs

<p align="center">
  Ziyi Wang<sup>†</sup>, Chen Zhang<sup>†</sup>, Wenjun Peng, Qi Wu, Xinyu Wang<sup>*</sup>
</p>
<p align="center">
  Adelaide University, Australia
</p>
<p align="center">
  <sup>†</sup>Equal Contribution <sup>*</sup>Corresponding author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.20043">Paper</a> ·
  <a href="https://adelaideuniversity.box.com/s/aclw7bf8dmyl7ppefvp7cw96briiiprm">Data Archive</a>
</p>

<p align="center">
  <img src="docs/banner.png" alt="TriEx framework overview" width="800" />
</p>

---

## News

- **2026-04-07** &nbsp; 🎉TriEx was accepted to the ACL 2026 Main Conference

---

## Installation
Python ≥ 3.10
```bash
git clone https://github.com/Einsam1819/TriEx.git
cd TriEx
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Environment variables

The LLM agents call the OpenRouter API. Set:

```bash
# Linux / macOS
export OPENROUTER_API_KEY="..."

# Windows PowerShell
$env:OPENROUTER_API_KEY = "..."
```

If you route requests through a different backend, edit [`players/llm_base_player.py`](players/llm_base_player.py) accordingly.

---

## Data archive

All data (341 trace files, evaluation outputs, figures) is hosted on Box and must be downloaded separately before running any experiment script:

> **Download:** <https://adelaideuniversity.box.com/s/aclw7bf8dmyl7ppefvp7cw96briiiprm>
>
> File: `triex_data.tar.gz` (~39.8 MB compressed; ~650 MB extracted; 341 files)

Place the archive at the repository root and extract:

```bash
tar -xzf triex_data.tar.gz
```

Then either merge the extracted tree into the repo root (scripts find data automatically):

```bash
cp -r triex_data/* .     # merge into repo root
rm -rf triex_data triex_data.tar.gz   # optional cleanup
```

Or keep the archive in a separate directory and point the runner at it:

```bash
export TRIEX_DATA_ROOT=/path/to/triex_data   # Linux / macOS
$env:TRIEX_DATA_ROOT = "C:\path\to\triex_data"  # Windows PowerShell
```

When `TRIEX_DATA_ROOT` is set, all scripts and the unified entrypoint resolve input and output paths from that root automatically — no `cd` into experiment subdirectories is required.

> ⚠️ Re-running an experiment script overwrites files in the corresponding `experiments/exp*/` directory. Keep a copy of the released data if you intend to compare your re-run against the published numbers.

---

## Quickstart (using released data)

The fastest path to inspect TriEx without re-running any LLM:

```bash
# 1. Install dependencies and download data (above)
# 2. Generate the HTML timeline replay
python scripts/view_timeline_replay.py
# → produces html_batches_timeline/poker_timeline_batch_*.html

# 3. Open one of the generated HTML files in a browser to inspect a hand
#    trajectory with first-person explanations and second-person belief
#    snapshots side-by-side.
```

---

## Reproducing experiments

Each subdirectory under [`experiments/`](experiments) corresponds to one analysis stage. The precise paper-section reference will be filled in once the camera-ready is finalized.

| Directory | Role | Unified CLI | Main script                         | Paper        |
|---|---|---|-------------------------------------|--------------|
| [`exp1_rulebase/`](experiments/exp1_rulebase) | Rule-based first-person baseline; faithfulness flattening per street | `--exp 1` | `exp1_rulebase.py` + `analysis.ipynb` | <§4.1>       |
| [`exp2a_ranking/`](experiments/exp2a_ranking) | Profile convergence, 5-dimensional ranking from interaction snapshots | `--exp 2a` | `exp2a_rank.py`  | <§4.2>       |
| [`exp2b_profiling/`](experiments/exp2b_profiling) | Cross-model opponent profiling comparison | `--exp 2b` | `2B_analysis.py` + `combine.ipynb`  | <§4.2>       |
| [`exp2c_intervention/`](experiments/exp2c_intervention) | Belief intervention and rerun-controlled experiments | `--exp 2c` | `exp2C.py`                          | <§4.2>       |
| [`exp3a_oracle/`](experiments/exp3a_oracle) | Third-person oracle audits, faithfulness scoring | `--exp 3a` | `3A_Ex.py`                          | <§4.3>       |
| [`exp3b_window/`](experiments/exp3b_window) | Windowed oracle analysis (window sizes 5 / 10 / 15) | `--exp 3b --model <m>` | `exp3b.py`                          | <§4.3>       |
| [`exp3c_meta/`](experiments/exp3c_meta) | Cross-experiment meta-analysis (kappa, alignment, faithfulness) | — | `3Cnote.ipynb`                      | <§4.3> |

The recommended way to run any experiment is the unified CLI from the repo root:

```bash
# Run exp 1 (rule-based faithfulness)
python -m triex.experiments run --exp 1

# Run exp 2a (rank-based convergence)
python -m triex.experiments run --exp 2a

# Run exp 2b (cross-model profiling)
python -m triex.experiments run --exp 2b

# Run exp 2c with a specific intervention
python -m triex.experiments run --exp 2c --intervention Aggressiveness_Up

# Run exp 3a (oracle faithfulness audit)
python -m triex.experiments run --exp 3a

# Run exp 3b for one model
python -m triex.experiments run --exp 3b --model deepseek

# Run exp 3b for all six models in sequence
python -m triex.experiments run --exp 3b --model all

# Override data root if you didn't merge triex_data into the repo root
python -m triex.experiments run --exp 1 --data-root /path/to/triex_data
```

Available `--model` values for exp 3b: `deepseek`, `gpt`, `gemini`, `grok`, `llama`, `qwen`.

Alternatively, you can still run each script directly from its own directory (the old workflow remains supported):

```bash
cd experiments/exp2c_intervention
python exp2C.py
# outputs land in exp2c_out/run_<idx>/...
```

---

## Running fresh experiments

To regenerate the gameplay traces from scratch (rather than using the released archive):

```bash
# 1. Make sure OPENROUTER_API_KEY is set
python scripts/run_game_parallel.py
# → produces gameplay traces and per-agent reasoning artifacts

python scripts/evaluation.py
# → first-person faithfulness and aggregate statistics

python scripts/triex_oracle.py
# → third-person oracle annotations
```

The default experimental configuration reported in the paper is:

| Setting | Value |
|---|---|
| Number of battles | 50 |
| Hands per battle | 30 |
| Initial stack | 3000 |
| Small / big blind | 5 / 10 |
| Decoding temperature | 0.2 |
| top-p | 1.0 |
| Random seed | 7 |
| Logit-intervention magnitude | 2.5 |
| Monte Carlo simulation count | 1000 |

### Evaluated agents

- GPT-4.1-mini
- DeepSeek-V3.2
- Gemini-2.5-Flash-Lite
- Grok-3-Mini
- Qwen3-32B
- Llama-4-Maverick

The poker engine is [PyPokerEngine](https://github.com/ishikota/PyPokerEngine) with our extensions for trace construction and TriEx instrumentation.

---

## Visualization

### Timeline replay

[`scripts/view_timeline_replay.py`](scripts/view_timeline_replay.py) renders every battle in `results.json` as a self-contained HTML page. Each page shows a hand-by-hand timeline with first-person reasoning traces, second-person belief snapshots, and third-person oracle annotations laid out side-by-side — useful for qualitative inspection without running any analysis code.

```bash
python scripts/view_timeline_replay.py
# → html_batches_timeline/poker_timeline_batch_*.html  (2 battles per file)
```

Open any generated HTML file directly in a browser; no server required.

### Radar charts

The [`visualization/`](visualization) directory contains scripts and notebooks for producing the figures reported in the paper.

| Path | Produces |
|---|---|
| [`visualization/radar/rader.py`](visualization/radar/rader.py) | Compact radar charts comparing LLM behavioral profiles across five dimensions |
| [`visualization/radar/rader.ipynb`](visualization/radar/rader.ipynb) | Interactive notebook version of the same analysis |

```bash
python visualization/radar/rader.py
# → figures in visualization/radar/radar_out_compact/
```

Input paths (experiment CSVs + oracle annotations) are resolved via `TRIEX_DATA_ROOT`, or the repo root if the env var is not set.

---

## Citation

If you find this project useful for your research, please consider citing our work:

```bibtex
@inproceedings{wang2026triex,
  title     = {TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs},
  author    = {Wang, Ziyi and Zhang, Chen and Peng, Wenjun and Wu, Qi and Wang, Xinyu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

---

## Contact

For questions about the paper or repository, please contact:

- Ziyi Wang &lt;<ziyiwang979@gmail.com>&gt;
- Xinyu Wang &lt;<xinyu.wang02@adelaide.edu.au>&gt;

Or open a GitHub issue.

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

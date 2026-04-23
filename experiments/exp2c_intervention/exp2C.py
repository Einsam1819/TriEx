from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv
from openai import OpenAI

try:
    from triex.config import results_json as _rj, output_dir as _od
    _EXP_ID = "exp2c_intervention"
    _DEFAULT_RESULTS = str(_rj(_EXP_ID))
    _DEFAULT_OUT     = str(_od(_EXP_ID, "exp2c_out"))
except ImportError:
    _DEFAULT_RESULTS = "results.json"
    _DEFAULT_OUT     = "exp2c_out"

RUN_CONFIG = {
    "results": _DEFAULT_RESULTS,
    "out": _DEFAULT_OUT,

    "n_runs": 50,
    "resume": True,
    "save_pairs_each_run": True,

    "fixed_states": True,
    "fixed_states_file": "fixed_states.json",

    "n_states": 300,
    "seed": 7,  # base seed; run i uses seed + i for LLM randomness, but state set is fixed
    "count_raise_amount_change_as_action_change": True,

    "trait": "Aggressiveness",
    "direction": "up",
    "target_opponent": "random",  # used by single-mode; ignored in all-mode
    "intervene_mode": "all",      # "single" or "all"

    "temperature": 0.2,
    "use_complex_messages": True,
    "max_workers": 10,
    "request_timeout": 90,

    # Optional filters applied BEFORE fixed sampling.
    "streets": "river",   # str / list of streets / None
    "players": None,      # list of player names / None

    "model_map": None,
    "extra_prompt": None,

    # Intervention strength.
    "logit_k": 10,
    "logit_eps": 1e-4,
    "skip_up_if_ge": 1.00,
    "skip_down_if_le": 0.00,
}

MODEL_MAP: Dict[str, str] = {
    "Alex Chen": "openai/gpt-4.1-mini",
    "Sarah Johnson": "meta-llama/llama-4-maverick",
    "Michael Davis": "anthropic/claude-3.5-haiku",
    "Jessica Liu": "google/gemini-2.5-flash-lite",
    "Emily Zhang": "deepseek/deepseek-v3.2",
    "Niko Grey": "x-ai/grok-3-mini",
    "Robert Garcia": "qwen/qwen3-32b",
}

DEFAULT_TEMPERATURE = 0.2

TRAITS = [
    "RiskTolerance",
    "Aggressiveness",
    "BluffFrequency",
    "CallingStationTendency",
    "ShowdownPropensity",
]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def _sample_variance(xs: List[float]) -> float:
    if not xs or len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

@dataclass
class StateSample:
    battle_id: int
    player: str
    round: int
    street: str
    node: Dict[str, Any]

@dataclass
class RunPairResult:
    sample: StateSample
    trait: str
    direction: str
    target_opponent: str

    original_action: str
    original_action_raw: Dict[str, Any]
    original_reasoning: str

    rerun_original_action: str
    rerun_original_action_raw: Dict[str, Any]
    rerun_original_reasoning: str

    intervened_action: str
    intervened_action_raw: Dict[str, Any]
    intervened_reasoning: str

    action_changed: bool
    transition: str
    risk_score_delta: int
    dir_consistent: Optional[bool]

_PROFILE_SECTION_HEAD_RE = re.compile(
    r"(Opponent profiling information \(long-term tendencies\)\s*:\s*\n)",
    re.IGNORECASE
)
_PROFILE_SECTION_TAIL_RE = re.compile(
    r"\n\s*-\s*RiskTolerance\s*:\s*how willing this opponent is to take risks\.",
    re.IGNORECASE
)

def extract_profiles_section_from_final_prompt(final_prompt: str) -> Optional[Tuple[str, str, str]]:
    if not final_prompt:
        return None
    m_head = _PROFILE_SECTION_HEAD_RE.search(final_prompt)
    if not m_head:
        return None
    start = m_head.end()
    m_tail = _PROFILE_SECTION_TAIL_RE.search(final_prompt, pos=start)
    if not m_tail:
        return None
    end = m_tail.start()
    prefix = final_prompt[:start]
    profiles_text = final_prompt[start:end]
    suffix = final_prompt[end:]
    return prefix, profiles_text, suffix

def replace_profiles_section_in_final_prompt(final_prompt: str, new_profiles_text: str) -> str:
    parts = extract_profiles_section_from_final_prompt(final_prompt)
    if not parts:
        return final_prompt
    prefix, _, suffix = parts
    return prefix + (new_profiles_text or "") + suffix

_PROFILE_TRAIT_LINE_RE = re.compile(
    r"(?P<prefix>^\s*\*\s+RiskTolerance=)(?P<rt>[0-9.]+),\s+"
    r"(Aggressiveness=)(?P<ag>[0-9.]+),\s+"
    r"(BluffFrequency=)(?P<bf>[0-9.]+),\s+"
    r"(CallingStationTendency=)(?P<cs>[0-9.]+),\s+"
    r"(ShowdownPropensity=)(?P<sd>[0-9.]+)",
    re.IGNORECASE | re.MULTILINE
)

def list_profile_opponents(opponent_profiles_text: str) -> List[str]:
    names = re.findall(r"^\-\s+(.+?)\s*:\s*$", opponent_profiles_text, flags=re.MULTILINE)
    return [n.strip() for n in names if n.strip()]

def get_trait_value_from_profiles_text(opponent_profiles_text: str, opp_name: str, trait: str) -> Optional[float]:
    pattern = re.compile(
        rf"(^-\s+{re.escape(opp_name)}\s*:\s*$)(.*?)(?=^\-\s+.+?:\s*$|\Z)",
        re.MULTILINE | re.DOTALL
    )
    m = pattern.search(opponent_profiles_text)
    if not m:
        return None
    body = m.group(2)
    m2 = _PROFILE_TRAIT_LINE_RE.search(body)
    if not m2:
        return None
    try:
        rt = float(m2.group("rt"))
        ag = float(m2.group("ag"))
        bf = float(m2.group("bf"))
        cs = float(m2.group("cs"))
        sd = float(m2.group("sd"))
    except Exception:
        return None
    mp = {
        "RiskTolerance": rt,
        "Aggressiveness": ag,
        "BluffFrequency": bf,
        "CallingStationTendency": cs,
        "ShowdownPropensity": sd,
    }
    return mp.get(trait)

def intervene_profiles_text_all(
    opponent_profiles_text: str,
    trait: str,
    direction: str,
    skip_up_if_ge: float = None,
    skip_down_if_le: float = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not opponent_profiles_text or "Opponent Profiles" not in opponent_profiles_text:
        return opponent_profiles_text, []

    opps = list_profile_opponents(opponent_profiles_text)
    if not opps:
        return opponent_profiles_text, []

    up_skip_ge = float(RUN_CONFIG.get("skip_up_if_ge", 0.8) if skip_up_if_ge is None else skip_up_if_ge)
    down_skip_le = float(RUN_CONFIG.get("skip_down_if_le", 0.2) if skip_down_if_le is None else skip_down_if_le)

    def eligible(old_v: Optional[float]) -> bool:
        if old_v is None:
            return False
        if direction.lower() == "up":
            return old_v < up_skip_ge
        else:
            return old_v > down_skip_le

    k = float(RUN_CONFIG.get("logit_k", 1.5))
    eps = float(RUN_CONFIG.get("logit_eps", 1e-4))

    text = opponent_profiles_text
    changes: List[Dict[str, Any]] = []

    for chosen in opps:
        old_v = get_trait_value_from_profiles_text(text, chosen, trait)
        if not eligible(old_v):
            continue

        pattern = re.compile(
            rf"(^-\s+{re.escape(chosen)}\s*:\s*$)(.*?)(?=^\-\s+.+?:\s*$|\Z)",
            re.MULTILINE | re.DOTALL
        )
        m = pattern.search(text)
        if not m:
            continue

        header = m.group(1)
        body = m.group(2)

        m2 = _PROFILE_TRAIT_LINE_RE.search(body)
        if not m2:
            continue

        rt = float(m2.group("rt"))
        ag = float(m2.group("ag"))
        bf = float(m2.group("bf"))
        cs = float(m2.group("cs"))
        sd = float(m2.group("sd"))

        old_map = {
            "RiskTolerance": rt,
            "Aggressiveness": ag,
            "BluffFrequency": bf,
            "CallingStationTendency": cs,
            "ShowdownPropensity": sd,
        }
        if trait not in old_map:
            continue

        x = float(old_map[trait])
        x = min(max(x, eps), 1.0 - eps)
        z = _logit(x)
        z2 = z + k if direction.lower() == "up" else z - k
        new_value = _clip01(_sigmoid(z2))

        new_map = dict(old_map)
        new_map[trait] = float(new_value)

        new_trait_line = (
            f"* RiskTolerance={new_map['RiskTolerance']:.2f}, Aggressiveness={new_map['Aggressiveness']:.2f}, "
            f"BluffFrequency={new_map['BluffFrequency']:.2f}, CallingStationTendency={new_map['CallingStationTendency']:.2f}, "
            f"ShowdownPropensity={new_map['ShowdownPropensity']:.2f}"
        )
        body2 = _PROFILE_TRAIT_LINE_RE.sub(new_trait_line, body, count=1)
        text = text[:m.start()] + header + body2 + text[m.end():]

        changes.append({"opponent": chosen, "old": old_map[trait], "new": new_map[trait]})

    return text, changes

def intervene_profiles_text_single(
    opponent_profiles_text: str,
    trait: str,
    direction: str,
    target_opponent: str = "random",
    rng: Optional[random.Random] = None
) -> Tuple[str, str, Optional[float], Optional[float]]:
    rng = rng or random.Random()

    if not opponent_profiles_text or "Opponent Profiles" not in opponent_profiles_text:
        return opponent_profiles_text, "", None, None

    opps = list_profile_opponents(opponent_profiles_text)
    if not opps:
        return opponent_profiles_text, "", None, None

    up_skip_ge = float(RUN_CONFIG.get("skip_up_if_ge", 0.8))
    down_skip_le = float(RUN_CONFIG.get("skip_down_if_le", 0.2))

    def eligible(old_v: Optional[float]) -> bool:
        if old_v is None:
            return False
        if direction.lower() == "up":
            return old_v < up_skip_ge
        else:
            return old_v > down_skip_le

    if target_opponent == "random":
        preferred = rng.choice(opps)


    else:
        preferred = target_opponent if target_opponent in opps else rng.choice(opps)

    preferred_v = get_trait_value_from_profiles_text(opponent_profiles_text, preferred, trait)

    if eligible(preferred_v):
        chosen = preferred
    else:
        candidates = []
        for o in opps:
            if o == preferred:
                continue
            v = get_trait_value_from_profiles_text(opponent_profiles_text, o, trait)
            if eligible(v):
                candidates.append(o)
        if not candidates:
            return opponent_profiles_text, "", None, None
        chosen = rng.choice(candidates)

    pattern = re.compile(
        rf"(^-\s+{re.escape(chosen)}\s*:\s*$)(.*?)(?=^\-\s+.+?:\s*$|\Z)",
        re.MULTILINE | re.DOTALL
    )
    m = pattern.search(opponent_profiles_text)
    if not m:
        return opponent_profiles_text, chosen, None, None

    header = m.group(1)
    body = m.group(2)

    m2 = _PROFILE_TRAIT_LINE_RE.search(body)
    if not m2:
        return opponent_profiles_text, chosen, None, None

    rt = float(m2.group("rt"))
    ag = float(m2.group("ag"))
    bf = float(m2.group("bf"))
    cs = float(m2.group("cs"))
    sd = float(m2.group("sd"))

    old_map = {
        "RiskTolerance": rt,
        "Aggressiveness": ag,
        "BluffFrequency": bf,
        "CallingStationTendency": cs,
        "ShowdownPropensity": sd,
    }
    if trait not in old_map:
        return opponent_profiles_text, chosen, None, None

    old_v = float(old_map[trait])

    k = float(RUN_CONFIG.get("logit_k", 1.5))
    eps = float(RUN_CONFIG.get("logit_eps", 1e-4))

    x = min(max(old_v, eps), 1.0 - eps)
    z = _logit(x)
    z2 = z + k if direction.lower() == "up" else z - k
    new_value = _clip01(_sigmoid(z2))

    new_map = dict(old_map)
    new_map[trait] = float(new_value)

    new_trait_line = (
        f"* RiskTolerance={new_map['RiskTolerance']:.2f}, Aggressiveness={new_map['Aggressiveness']:.2f}, "
        f"BluffFrequency={new_map['BluffFrequency']:.2f}, CallingStationTendency={new_map['CallingStationTendency']:.2f}, "
        f"ShowdownPropensity={new_map['ShowdownPropensity']:.2f}"
    )
    body2 = _PROFILE_TRAIT_LINE_RE.sub(new_trait_line, body, count=1)
    new_text = opponent_profiles_text[:m.start()] + header + body2 + opponent_profiles_text[m.end():]
    return new_text, chosen, old_v, float(new_map[trait])


def parse_llm_response(llm_response: str) -> Tuple[str, Dict[str, Any]]:
    text = (llm_response or "").strip()
    reasoning = ""
    decision = {"action": "FOLD"}

    se_match = re.search(r"\[SELF-EXPLANATION\](.*?)\[/SELF-EXPLANATION\]", text, re.DOTALL | re.IGNORECASE)
    if se_match:
        reasoning = se_match.group(0).strip()
    else:
        reasoning = text

    decision_json = None
    m_dec = re.search(r"DECISION\s*:\s*(\{.*\})\s*$", text, re.IGNORECASE | re.DOTALL)
    if m_dec:
        tail = m_dec.group(1).strip()
        l = tail.find("{")
        r = tail.rfind("}")
        if l != -1 and r != -1 and r > l:
            decision_json = tail[l:r + 1]

    if not decision_json:
        json_candidates = re.findall(r"\{[\s\S]*?\}", text)
        for cand in reversed(json_candidates):
            if re.search(r"\"action\"\s*:", cand, re.IGNORECASE):
                decision_json = cand
                break

    if decision_json:
        try:
            parsed = json.loads(decision_json)
            if isinstance(parsed, dict) and "action" in parsed:
                parsed["action"] = str(parsed.get("action", "FOLD")).upper().strip()
                decision = parsed
        except Exception:
            pass

    return reasoning, decision


_RE_CALL_AMOUNT = re.compile(r"^\s*-\s*Call amount\s*:\s*([0-9]+)\s*$", re.MULTILINE)
_RE_MIN_RAISE   = re.compile(r"^\s*-\s*Minimum raise\s*:\s*([0-9]+)\s*$", re.MULTILINE)
_RE_MAX_RAISE   = re.compile(r"^\s*-\s*Maximum raise\s*:\s*([0-9]+)\s*$", re.MULTILINE)

def extract_action_constraints_from_prompt(final_prompt: str) -> Tuple[int, int, int]:
    def _pick_int(rx, default=0):
        m = rx.search(final_prompt or "")
        if not m:
            return default
        try:
            return int(m.group(1))
        except Exception:
            return default
    return _pick_int(_RE_CALL_AMOUNT, 0), _pick_int(_RE_MIN_RAISE, 0), _pick_int(_RE_MAX_RAISE, 0)

def normalize_decision_like_llmbaseplayer(
    decision: Any, call_amount: int, min_raise: int, max_raise: int
) -> Tuple[Dict[str, Any], str]:
    if not isinstance(decision, dict):
        return {"action": "FOLD"}, "Decision not dict -> FOLD"

    act = str(decision.get("action", "FOLD")).upper().strip()

    if act == "CHECK":
        if call_amount == 0:
            return {"action": "CALL"}, "CHECK mapped to CALL"
        return {"action": "FOLD"}, "CHECK invalid -> FOLD"

    if act == "CALL":
        return {"action": "CALL"}, "CALL normalized"
    if act == "FOLD":
        return {"action": "FOLD"}, "FOLD normalized"

    if act == "RAISE":
        if max_raise < min_raise or max_raise <= 0:
            return {"action": "CALL"}, "RAISE not allowed -> CALL"
        intended = decision.get("amount", min_raise)
        try:
            intended_int = int(intended)
        except Exception:
            intended_int = min_raise
        if intended_int == -1:
            intended_int = max_raise
        actual = max(min_raise, min(intended_int, max_raise))
        return {"action": "RAISE", "amount": actual}, f"RAISE clipped to {actual}"

    if call_amount == 0:
        return {"action": "CALL"}, f"Unknown '{act}' -> CALL"
    return {"action": "FOLD"}, f"Unknown '{act}' -> FOLD"

def action_to_label(action_dict: Dict[str, Any]) -> str:
    if not isinstance(action_dict, dict):
        return "FOLD"
    a = str(action_dict.get("action", "FOLD")).upper().strip()
    if a not in ("FOLD", "CALL", "CHECK", "RAISE"):
        a = "FOLD"
    if a == "CHECK":
        a = "CALL"
    return a

def label_to_risk_score(label: str) -> int:
    label = (label or "").upper().strip()
    if label == "FOLD":
        return 0
    if label in ("CALL", "CHECK"):
        return 1
    if label == "RAISE":
        return 2
    return 0

def expected_dir_consistency(trait: str, direction: str, delta_risk: int) -> Optional[bool]:
    d = direction.lower()
    if trait not in TRAITS:
        return None
    if trait in ("Aggressiveness", "RiskTolerance", "ShowdownPropensity"):
        return (delta_risk <= 0) if d == "up" else (delta_risk >= 0)
    if trait in ("BluffFrequency", "CallingStationTendency"):
        return (delta_risk >= 0) if d == "up" else (delta_risk <= 0)
    return None


def iter_states(results: Dict[str, Any]) -> List[StateSample]:
    out: List[StateSample] = []
    battles = results.get("battles", [])
    for b_id, b in enumerate(battles):
        rh = b.get("reasoning_history", {}) or {}
        for player_name, rounds in rh.items():
            if not isinstance(rounds, dict):
                continue
            for round_str, streets in rounds.items():
                try:
                    rnum = int(round_str)
                except Exception:
                    continue
                if not isinstance(streets, dict):
                    continue
                for street, node in streets.items():
                    if not isinstance(node, dict):
                        continue
                    pc = node.get("prompt_components")
                    if not pc or not isinstance(pc, dict):
                        continue
                    fp = pc.get("final_prompt")
                    if not fp or not isinstance(fp, str):
                        continue
                    if not extract_profiles_section_from_final_prompt(fp):
                        continue
                    out.append(StateSample(
                        battle_id=b_id,
                        player=str(player_name),
                        round=rnum,
                        street=str(street),
                        node=node
                    ))
    return out

def _state_key(sample: StateSample) -> Dict[str, Any]:
    return {"battle_id": sample.battle_id, "player": sample.player, "round": sample.round, "street": sample.street}

def _get_node_from_results(results: Dict[str, Any], key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        b_id = int(key["battle_id"])
        player = str(key["player"])
        rnum = int(key["round"])
        street = str(key["street"])
    except Exception:
        return None

    battles = results.get("battles", [])
    if b_id < 0 or b_id >= len(battles):
        return None

    rh = (battles[b_id].get("reasoning_history", {}) or {})
    rounds = rh.get(player, None)
    if not isinstance(rounds, dict):
        return None

    streets = rounds.get(str(rnum), None) or rounds.get(rnum, None)
    if not isinstance(streets, dict):
        return None

    node = streets.get(street, None)
    if not isinstance(node, dict):
        return None

    pc = node.get("prompt_components", {}) or {}
    fp = pc.get("final_prompt", None)
    if not isinstance(fp, str) or not fp:
        return None
    if not extract_profiles_section_from_final_prompt(fp):
        return None

    return node

def load_or_create_fixed_state_keys(
    results: Dict[str, Any],
    out_dir: Path,
    n_states: int,
    seed: int,
    streets: Optional[List[str]],
    players: Optional[List[str]],
) -> List[Dict[str, Any]]:
    fixed_name = str(RUN_CONFIG.get("fixed_states_file", "fixed_states.json"))
    fixed_path = out_dir / fixed_name

    if fixed_path.exists():
        data = json.loads(fixed_path.read_text(encoding="utf-8"))
        keys = data.get("state_keys", [])
        if isinstance(keys, list) and keys:
            return keys

    all_states = iter_states(results)

    if streets:
        street_set = {s.lower() for s in streets}
        all_states = [s for s in all_states if s.street.lower() in street_set]
    if players:
        pset = set(players)
        all_states = [s for s in all_states if s.player in pset]

    if not all_states:
        raise RuntimeError("No eligible states found for fixed sampling.")

    rng = random.Random(seed)
    rng.shuffle(all_states)
    picked = all_states[:min(n_states, len(all_states))]
    keys = [_state_key(s) for s in picked]

    payload = {
        "meta": {
            "time": _dt.datetime.now().isoformat(),
            "n_states": n_states,
            "seed": seed,
            "streets": streets,
            "players": players,
            "source_results": str(RUN_CONFIG.get("results")),
        },
        "state_keys": keys,
    }
    _safe_write_json(fixed_path, payload)
    return keys

def call_openrouter(
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    use_complex_messages: bool = True,
    timeout: Optional[float] = None
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY env var.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if use_complex_messages:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    return resp.choices[0].message.content


def run_exp2c_single(
    results_path: Path,
    out_dir: Path,
    n_states: int,
    run_seed: int,                 # used for picking single-opponent when needed + any random ops
    trait: str,
    direction: str,
    target_opponent: str,
    model_map_path: Optional[Path],
    temperature: float,
    use_complex_messages: bool,
    extra_prompt: Optional[str],
    streets: Optional[List[str]],
    players: Optional[List[str]],
    fixed_state_keys: Optional[List[Dict[str, Any]]] = None,
    save_pairs: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text(encoding="utf-8"))

    model_map = dict(MODEL_MAP)
    if model_map_path and model_map_path.exists():
        model_map.update(json.loads(model_map_path.read_text(encoding="utf-8")))

    if fixed_state_keys:
        samples: List[StateSample] = []
        missing = 0
        for k in fixed_state_keys:
            node = _get_node_from_results(results, k)
            if node is None:
                missing += 1
                continue
            samples.append(StateSample(
                battle_id=int(k["battle_id"]),
                player=str(k["player"]),
                round=int(k["round"]),
                street=str(k["street"]),
                node=node,
            ))
        if not samples:
            raise RuntimeError(
                "All fixed states are missing/invalid in current results.json. "
                "If results.json changed, delete fixed_states.json and rerun."
            )
        if missing > 0:
            print(f"[WARN] {missing} fixed states not found/invalid, skipped.")
    else:
        all_states = iter_states(results)

        if streets:
            street_set = {s.lower() for s in streets}
            all_states = [s for s in all_states if s.street.lower() in street_set]
        if players:
            pset = set(players)
            all_states = [s for s in all_states if s.player in pset]

        if not all_states:
            raise RuntimeError("No eligible states found. Ensure results.json includes final_prompt + profiles section.")

        rng_local = random.Random(run_seed)
        rng_local.shuffle(all_states)
        samples = all_states[:min(n_states, len(all_states))]

    transition_logged_vs_rerun_orig = Counter()
    transition_logged_vs_rerun_int = Counter()
    transition_rerun_orig_vs_int = Counter()

    n_logged_vs_rerun_orig_changed = 0
    n_logged_vs_rerun_int_changed = 0
    n_rerun_orig_vs_int_changed = 0

    max_workers = int(RUN_CONFIG.get("max_workers", 6))
    timeout = RUN_CONFIG.get("request_timeout", None)

    audit_dir = out_dir / "prompt_audits"
    audit_dir.mkdir(exist_ok=True)

    rng = random.Random(run_seed)

    jobs = []
    for idx, sample in enumerate(samples, start=1):
        node = sample.node
        pc = node.get("prompt_components", {}) or {}
        if not pc:
            continue

        player_name = sample.player
        model = model_map.get(player_name)
        if not model:
            continue

        final_prompt = pc.get("final_prompt", "")
        if not final_prompt:
            continue

        call_amount, min_raise, max_raise = extract_action_constraints_from_prompt(final_prompt)

        parts = extract_profiles_section_from_final_prompt(final_prompt)
        if not parts:
            continue
        _, orig_profiles_text, _ = parts

        mode = str(RUN_CONFIG.get("intervene_mode", "single")).lower().strip()
        if mode not in ("single", "all"):
            mode = "single"

        if mode == "all":
            new_profiles_text, changes = intervene_profiles_text_all(
                opponent_profiles_text=orig_profiles_text,
                trait=trait,
                direction=direction,
            )
            if not changes:
                continue
            chosen_opp = "ALL"
            intervened_changes = changes
        else:
            new_profiles_text, chosen_opp, old_v, new_v = intervene_profiles_text_single(
                opponent_profiles_text=orig_profiles_text,
                trait=trait,
                direction=direction,
                target_opponent=target_opponent,
                rng=rng
            )
            if not chosen_opp:
                continue
            intervened_changes = [{"opponent": chosen_opp, "old": old_v, "new": new_v}]

        int_prompt = replace_profiles_section_in_final_prompt(final_prompt, new_profiles_text)

        original_action_raw = node.get("action", {}) or {}
        original_reasoning = node.get("reasoning", "") or ""
        original_action = action_to_label(original_action_raw)

        audit_item = {
            "battle_id": sample.battle_id,
            "player": sample.player,
            "round": sample.round,
            "street": sample.street,
            "trait": trait,
            "direction": direction,
            "chosen_opponent": chosen_opp,
            "intervene_mode": mode,
            "intervened_changes": intervened_changes,
            "original_prompt": final_prompt,
            "intervened_prompt": int_prompt,
            "original_action": original_action_raw,
        }
        _safe_write_json(audit_dir / f"sample_{idx:04d}.json", audit_item)

        jobs.append({
            "idx": idx,
            "sample": sample,
            "model": model,
            "orig_prompt": final_prompt,
            "int_prompt": int_prompt,
            "original_action_raw": original_action_raw,
            "original_reasoning": original_reasoning,
            "chosen_opp": chosen_opp,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "max_raise": max_raise,
        })

    if not jobs:
        raise RuntimeError("No jobs built in this run. (all skipped due to eligibility / missing model_map)")

    def _worker(job):
        def _call(p):
            return call_openrouter(
                model=job["model"],
                prompt=p,
                temperature=temperature,
                use_complex_messages=use_complex_messages,
                timeout=timeout
            )

        try:
            llm_text_orig = _call(job["orig_prompt"])
        except Exception as e:
            llm_text_orig = f"[API-ERROR] {e}\nDECISION:\n{{\"action\":\"FOLD\"}}"

        try:
            llm_text_int = _call(job["int_prompt"])
        except Exception as e:
            llm_text_int = f"[API-ERROR] {e}\nDECISION:\n{{\"action\":\"FOLD\"}}"

        return job["idx"], job, llm_text_orig, llm_text_int

    pair_results: List[RunPairResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            idx, job, llm_text_orig, llm_text_int = fut.result()

            rerun_orig_reasoning, rerun_orig_action_raw = parse_llm_response(llm_text_orig)
            rerun_int_reasoning, rerun_int_action_raw = parse_llm_response(llm_text_int)

            ca = int(job.get("call_amount", 0))
            mn = int(job.get("min_raise", 0))
            mx = int(job.get("max_raise", 0))

            logged_action_raw_norm, _ = normalize_decision_like_llmbaseplayer(job["original_action_raw"], ca, mn, mx)
            rerun_orig_action_raw_norm, _ = normalize_decision_like_llmbaseplayer(rerun_orig_action_raw, ca, mn, mx)
            rerun_int_action_raw_norm, _ = normalize_decision_like_llmbaseplayer(rerun_int_action_raw, ca, mn, mx)

            logged_action = action_to_label(logged_action_raw_norm)
            rerun_orig_action = action_to_label(rerun_orig_action_raw_norm)
            rerun_int_action = action_to_label(rerun_int_action_raw_norm)

            def action_changed(a_label, a_raw, b_label, b_raw):
                if a_label != b_label:
                    return True

                if RUN_CONFIG.get("count_raise_amount_change_as_action_change", False):
                    if a_label == "RAISE" and b_label == "RAISE":
                        return int(a_raw.get("amount", -999)) != int(b_raw.get("amount", -999))

                return False

            ch_logged_vs_rerun_orig = action_changed(
                logged_action, logged_action_raw_norm,
                rerun_orig_action, rerun_orig_action_raw_norm
            )
            ch_logged_vs_rerun_int = action_changed(
                logged_action, logged_action_raw_norm,
                rerun_int_action, rerun_int_action_raw_norm
            )
            ch_rerun_orig_vs_int = action_changed(
                rerun_orig_action, rerun_orig_action_raw_norm,
                rerun_int_action, rerun_int_action_raw_norm
            )

            n_logged_vs_rerun_orig_changed += int(ch_logged_vs_rerun_orig)
            n_logged_vs_rerun_int_changed += int(ch_logged_vs_rerun_int)
            n_rerun_orig_vs_int_changed += int(ch_rerun_orig_vs_int)

            trans_logged_vs_rerun_orig = f"{logged_action}->{rerun_orig_action}"
            trans_logged_vs_rerun_int = f"{logged_action}->{rerun_int_action}"
            trans_rerun_orig_vs_int = f"{rerun_orig_action}->{rerun_int_action}"

            transition_logged_vs_rerun_orig[trans_logged_vs_rerun_orig] += 1
            transition_logged_vs_rerun_int[trans_logged_vs_rerun_int] += 1
            transition_rerun_orig_vs_int[trans_rerun_orig_vs_int] += 1

            rs0 = label_to_risk_score(logged_action)
            rs1 = label_to_risk_score(rerun_int_action)
            delta_risk = rs1 - rs0
            dir_cons = expected_dir_consistency(trait, direction, delta_risk)

            r = RunPairResult(
                sample=job["sample"],
                trait=trait,
                direction=direction,
                target_opponent=job["chosen_opp"] or target_opponent,

                original_action=logged_action,
                original_action_raw=logged_action_raw_norm,
                original_reasoning=job["original_reasoning"],

                rerun_original_action=rerun_orig_action,
                rerun_original_action_raw=rerun_orig_action_raw_norm,
                rerun_original_reasoning=rerun_orig_reasoning,

                intervened_action=rerun_int_action,
                intervened_action_raw=rerun_int_action_raw_norm,
                intervened_reasoning=rerun_int_reasoning,

                action_changed=bool(ch_logged_vs_rerun_int),  # main signal = logged -> intervened
                transition=trans_logged_vs_rerun_int,
                risk_score_delta=int(delta_risk),
                dir_consistent=dir_cons,
            )
            pair_results.append(r)

            # Fill the audit JSON with rerun outputs.
            audit_path = audit_dir / f"sample_{idx:04d}.json"
            try:
                audit_item = json.loads(audit_path.read_text(encoding="utf-8"))
                audit_item["rerun_original_action"] = rerun_orig_action_raw_norm
                audit_item["rerun_intervened_action"] = rerun_int_action_raw_norm
                _safe_write_json(audit_path, audit_item)
            except Exception:
                pass

    total = len(pair_results)
    if total == 0:
        raise RuntimeError("No pairs produced in this run.")

    orig_counts = Counter(r.original_action for r in pair_results)
    rerun_orig_counts = Counter(r.rerun_original_action for r in pair_results)
    int_counts = Counter(r.intervened_action for r in pair_results)

    def rate(cnt: Counter, key: str) -> float:
        return cnt.get(key, 0) / total

    summary = {
        "meta": {
            "time": _dt.datetime.now().isoformat(),
            "results_path": str(results_path),
            "out_dir": str(out_dir),
            "n_states_requested": n_states,
            "n_pairs": total,
            "run_seed": run_seed,
            "trait": trait,
            "direction": direction,
            "temperature": temperature,
            "use_complex_messages": use_complex_messages,
            "fixed_states_enabled": bool(fixed_state_keys),
        },
        "metrics": {
            "change_rate_logged_vs_rerun_orig": n_logged_vs_rerun_orig_changed / total,
            "change_rate_logged_vs_rerun_int":  n_logged_vs_rerun_int_changed  / total,
            "change_rate_rerun_orig_vs_int":    n_rerun_orig_vs_int_changed    / total,

            "orig_fold_rate": rate(orig_counts, "FOLD"),
            "orig_call_rate": rate(orig_counts, "CALL"),
            "orig_raise_rate": rate(orig_counts, "RAISE"),

            "rerun_orig_fold_rate": rate(rerun_orig_counts, "FOLD"),
            "rerun_orig_call_rate": rate(rerun_orig_counts, "CALL"),
            "rerun_orig_raise_rate": rate(rerun_orig_counts, "RAISE"),

            "int_fold_rate": rate(int_counts, "FOLD"),
            "int_call_rate": rate(int_counts, "CALL"),
            "int_raise_rate": rate(int_counts, "RAISE"),

            "delta_fold_rate_logged_to_int": rate(int_counts, "FOLD") - rate(orig_counts, "FOLD"),
            "delta_call_rate_logged_to_int": rate(int_counts, "CALL") - rate(orig_counts, "CALL"),
            "delta_raise_rate_logged_to_int": rate(int_counts, "RAISE") - rate(orig_counts, "RAISE"),

            "dir_consistency_rate": (
                sum(1 for r in pair_results if r.dir_consistent is True) /
                max(1, sum(1 for r in pair_results if r.dir_consistent is not None))
            ),
        },
        "transition_matrices": {
            "logged_vs_rerun_orig": dict(transition_logged_vs_rerun_orig),
            "logged_vs_rerun_int":  dict(transition_logged_vs_rerun_int),
            "rerun_orig_vs_int":    dict(transition_rerun_orig_vs_int),
        }
    }

    _safe_write_json(out_dir / "summary.json", summary)

    # Compact TSV for quick inspection.
    lines = [
        "metric\tvalue",
        f"change_rate_logged_vs_rerun_orig\t{summary['metrics']['change_rate_logged_vs_rerun_orig']:.8f}",
        f"change_rate_logged_vs_rerun_int\t{summary['metrics']['change_rate_logged_vs_rerun_int']:.8f}",
        f"change_rate_rerun_orig_vs_int\t{summary['metrics']['change_rate_rerun_orig_vs_int']:.8f}",
        f"dir_consistency_rate\t{summary['metrics']['dir_consistency_rate']:.8f}",
    ]
    (out_dir / "summary.tsv").write_text("\n".join(lines), encoding="utf-8")

    if save_pairs:
        pairs_json = []
        for r in pair_results:
            pairs_json.append({
                "battle_id": r.sample.battle_id,
                "player": r.sample.player,
                "round": r.sample.round,
                "street": r.sample.street,
                "trait": r.trait,
                "direction": r.direction,
                "target_opponent": r.target_opponent,

                "logged_action": r.original_action_raw,
                "rerun_original_action": r.rerun_original_action_raw,
                "rerun_intervened_action": r.intervened_action_raw,

                "logged_action_label": r.original_action,
                "rerun_original_action_label": r.rerun_original_action,
                "rerun_intervened_action_label": r.intervened_action,

                "transition_logged_vs_rerun_int": r.transition,
                "risk_score_delta_logged_to_int": r.risk_score_delta,
                "dir_consistent": r.dir_consistent,

                "logged_reasoning": r.original_reasoning,
                "rerun_original_reasoning": r.rerun_original_reasoning,
                "rerun_intervened_reasoning": r.intervened_reasoning,
            })
        _safe_write_json(out_dir / "pairs.json", pairs_json)

    return summary

def run_exp2c_many_fixed_states() -> Dict[str, Any]:
    cfg = RUN_CONFIG

    results_path = Path(cfg["results"])
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_states = int(cfg["n_states"])
    base_seed = int(cfg["seed"])
    trait = str(cfg["trait"])
    direction = str(cfg["direction"])
    target_opponent = str(cfg.get("target_opponent", "random"))

    model_map_path = Path(cfg["model_map"]) if cfg.get("model_map") else None
    temperature = float(cfg.get("temperature", DEFAULT_TEMPERATURE))
    use_complex_messages = bool(cfg.get("use_complex_messages", True))
    extra_prompt = cfg.get("extra_prompt", None)

    streets = cfg.get("streets", None)
    if isinstance(streets, str):
        streets = [s.strip() for s in streets.split(",") if s.strip()]
    players = cfg.get("players", None)
    if isinstance(players, str):
        players = [s.strip() for s in players.split(",") if s.strip()]

    n_runs = int(cfg.get("n_runs", 50))
    resume = bool(cfg.get("resume", True))
    save_pairs_each_run = bool(cfg.get("save_pairs_each_run", True))

    progress_path = out_dir / "progress.json"
    progress = {
        "meta": {
            "time_started": _dt.datetime.now().isoformat(),
            "results_path": str(results_path),
            "out_dir": str(out_dir),
            "n_states": n_states,
            "base_seed": base_seed,
            "trait": trait,
            "direction": direction,
            "temperature": temperature,
            "use_complex_messages": use_complex_messages,
            "n_runs": n_runs,
            "fixed_states": bool(cfg.get("fixed_states", True)),
        },
        "completed_runs": {},
    }
    if resume and progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress.setdefault("completed_runs", {})
        except Exception:
            pass

    completed = set(int(k) for k in progress.get("completed_runs", {}).keys())

    # State sample is fixed across runs (loaded or created once).
    fixed_state_keys = None
    if bool(cfg.get("fixed_states", True)):
        results_for_fixed = json.loads(results_path.read_text(encoding="utf-8"))
        fixed_state_keys = load_or_create_fixed_state_keys(
            results=results_for_fixed,
            out_dir=out_dir,
            n_states=n_states,
            seed=base_seed,
            streets=streets,
            players=players,
        )
        print(f"[INFO] Using FIXED states: {len(fixed_state_keys)}")

    all_run_metrics: List[Dict[str, Any]] = []
    for k, v in sorted(progress.get("completed_runs", {}).items(), key=lambda kv: int(kv[0])):
        if isinstance(v, dict) and "metrics" in v:
            all_run_metrics.append(v["metrics"])

    for run_idx in range(n_runs):
        if resume and run_idx in completed:
            continue

        run_seed = base_seed
        run_subdir = out_dir / f"run_{run_idx:04d}"
        run_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== RUN {run_idx+1}/{n_runs} (run_seed={run_seed}) =====")
        summary = run_exp2c_single(
            results_path=results_path,
            out_dir=run_subdir,
            n_states=n_states,
            run_seed=run_seed,
            trait=trait,
            direction=direction,
            target_opponent=target_opponent,
            model_map_path=model_map_path,
            temperature=temperature,
            use_complex_messages=use_complex_messages,
            extra_prompt=extra_prompt,
            streets=streets,
            players=players,
            fixed_state_keys=fixed_state_keys,
            save_pairs=save_pairs_each_run,
        )

        item = {
            "run_idx": run_idx,
            "run_seed": run_seed,
            "run_dir": str(run_subdir),
            "time": _dt.datetime.now().isoformat(),
            "metrics": summary.get("metrics", {}),
        }
        progress["completed_runs"][str(run_idx)] = item
        _safe_write_json(progress_path, progress)

        all_run_metrics.append(item["metrics"])

    metric_keys = set()
    for m in all_run_metrics:
        metric_keys.update(m.keys())
    # Always include the three core change-rate metrics in the aggregate output.
    metric_keys.update([
        "change_rate_logged_vs_rerun_orig",
        "change_rate_logged_vs_rerun_int",
        "change_rate_rerun_orig_vs_int",
    ])

    agg_metrics = {}
    for key in sorted(metric_keys):
        xs = []
        for m in all_run_metrics:
            v = m.get(key, None)
            if isinstance(v, (int, float)):
                vf = float(v)
                if not math.isnan(vf):
                    xs.append(vf)
        if not xs:
            continue
        mu = _mean(xs)
        var = _sample_variance(xs)
        agg_metrics[key] = {"mean": mu, "variance": var, "std": math.sqrt(var), "n": len(xs)}

    aggregate = {
        "meta": {
            "time_finished": _dt.datetime.now().isoformat(),
            "results_path": str(results_path),
            "out_dir": str(out_dir),
            "n_runs": n_runs,
            "n_completed": len(all_run_metrics),
            "n_states": n_states,
            "base_seed": base_seed,
            "trait": trait,
            "direction": direction,
            "temperature": temperature,
            "use_complex_messages": use_complex_messages,
            "fixed_states_enabled": bool(fixed_state_keys),
            "fixed_states_file": str(out_dir / str(cfg.get("fixed_states_file", "fixed_states.json"))),
        },
        "aggregate_metrics": agg_metrics,
    }

    _safe_write_json(out_dir / "aggregate_summary.json", aggregate)

    tsv_lines = ["metric\tmean\tvariance\tstd\tn"]
    for k, v in agg_metrics.items():
        tsv_lines.append(f"{k}\t{v['mean']:.10f}\t{v['variance']:.10f}\t{v['std']:.10f}\t{v['n']}")
    (out_dir / "aggregate_summary.tsv").write_text("\n".join(tsv_lines), encoding="utf-8")

    csv_path = out_dir / "aggregate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "variance", "std", "n"])
        for metric, stats in agg_metrics.items():
            writer.writerow([
                metric,
                stats["mean"],
                stats["variance"],
                stats["std"],
                stats["n"],
            ])

    print("ALL RUNS FINISHED")
    print(f"Saved: {out_dir / 'aggregate_summary.json'}")
    print("---- Key change-rates (mean / variance / std) ----")
    for k in [
        "change_rate_logged_vs_rerun_orig",
        "change_rate_logged_vs_rerun_int",
        "change_rate_rerun_orig_vs_int",
    ]:
        if k in agg_metrics:
            v = agg_metrics[k]
            print(f"{k}: mean={v['mean']:.6f}, var={v['variance']:.6f}, std={v['std']:.6f}, n={v['n']}")

    return aggregate


if __name__ == "__main__":
    run_exp2c_many_fixed_states()


def main():
    """Entrypoint for use by ``python -m triex.experiments run --exp 2c``."""
    run_exp2c_many_fixed_states()
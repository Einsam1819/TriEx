import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import json
import math

try:
    from triex.config import html_timeline_dir as _htd, DATA_ROOT as _DATA_ROOT
    SRC = _DATA_ROOT / "results.json"
    OUT_DIR = _htd()
except ImportError:
    SRC = Path("results.json")
    OUT_DIR = Path("html_batches_timeline")
BATTLES_PER_HTML = 2

STREET_ORDER = ["preflop", "flop", "turn", "river", "showdown"]


def safe_read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield i // chunk_size + 1, items[i:i + chunk_size]


def sort_round_keys(d):
    def key_fn(x):
        try:
            return int(x)
        except Exception:
            return 10**9
    return sorted(d.keys(), key=key_fn)


def sort_street_keys(d):
    return sorted(d.keys(), key=lambda x: STREET_ORDER.index(x) if x in STREET_ORDER else 999)


def compact_profile(profile):
    if not isinstance(profile, dict):
        return {}
    return {
        "Traits": profile.get("Traits", {}) or {},
        "QualitativeSummary": profile.get("QualitativeSummary", "") or "",
        "UpdateRationale": profile.get("UpdateRationale", "") or "",
    }


def compact_node(node):
    if not isinstance(node, dict):
        return {}

    gs = node.get("game_state", {}) or {}
    pos = gs.get("position_info", {}) or {}

    compact_players_info = []
    for p in (pos.get("players_info", []) or []):
        if not isinstance(p, dict):
            continue
        compact_players_info.append({
            "name": p.get("name"),
            "stack": p.get("stack"),
            "position": p.get("position"),
            "is_dealer": p.get("is_dealer", False),
            "is_me": p.get("is_me", False),
        })

    compact_opponent_actions = {}
    for street, arr in (gs.get("opponent_actions", {}) or {}).items():
        if not isinstance(arr, list):
            continue
        compact_opponent_actions[street] = []
        for a in arr:
            if not isinstance(a, dict):
                continue
            compact_opponent_actions[street].append({
                "player": a.get("player"),
                "action": a.get("action"),
                "stack": a.get("stack"),
            })

    compact_snapshot = {}
    for opp, prof in (node.get("opponent_profiles_snapshot", {}) or {}).items():
        compact_snapshot[opp] = compact_profile(prof)

    return {
        "reasoning": node.get("reasoning", "") or "",
        "action": node.get("action", {}) or {},
        "beliefs": node.get("beliefs", {}) or {},
        "chosen_action_summary": node.get("chosen_action_summary", {}) or {},
        "opponent_profiles_snapshot": compact_snapshot,
        "game_state": {
            "street": gs.get("street"),
            "pot_size": gs.get("pot_size"),
            "call_amount": gs.get("call_amount"),
            "min_raise": gs.get("min_raise"),
            "max_raise": gs.get("max_raise"),
            "pot_odds": gs.get("pot_odds"),
            "env_hand_type": gs.get("env_hand_type"),
            "env_hand_strength": gs.get("env_hand_strength"),
            "llm_hand_strength_label": gs.get("llm_hand_strength_label"),
            "hole_cards": gs.get("hole_cards", []) or [],
            "community_cards": gs.get("community_cards", []) or [],
            "players_stacks": gs.get("players_stacks", {}) or {},
            "opponent_actions": compact_opponent_actions,
            "position_info": {
                "total_players": pos.get("total_players"),
                "my_position": pos.get("my_position"),
                "my_stack": pos.get("my_stack"),
                "players_info": compact_players_info,
            },
        },
    }


def compact_player_summary(player_name, player_data):
    if not isinstance(player_data, dict):
        return {}

    profile_history = {}
    for opp, arr in (player_data.get("opponent_profile_history", {}) or {}).items():
        if not isinstance(arr, list):
            continue
        profile_history[opp] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            profile_history[opp].append({
                "round": item.get("round"),
                "profile": compact_profile(item.get("profile", {}) or {}),
            })

    opponent_profiles = {}
    for opp, prof in (player_data.get("opponent_profiles", {}) or {}).items():
        opponent_profiles[opp] = compact_profile(prof)

    compact_reasoning_history = {}
    rh = player_data.get("reasoning_history", {}) or {}
    for rnd in sort_round_keys(rh):
        streets = rh.get(rnd, {}) or {}
        compact_reasoning_history[str(rnd)] = {}
        for street in sort_street_keys(streets):
            compact_reasoning_history[str(rnd)][street] = compact_node(streets.get(street, {}) or {})

    return {
        "name": player_name,
        "initial_stack": player_data.get("initial_stack"),
        "current_stack": player_data.get("current_stack"),
        "total_profit_loss": player_data.get("total_profit_loss"),
        "profit_percentage": player_data.get("profit_percentage"),
        "round_results": player_data.get("round_results", {}) or {},
        "hands_played": player_data.get("hands_played"),
        "hands_won": player_data.get("hands_won"),
        "hands_folded": player_data.get("hands_folded"),
        "hands_called": player_data.get("hands_called"),
        "hands_raised": player_data.get("hands_raised"),
        "win_rate": player_data.get("win_rate"),
        "fold_rate": player_data.get("fold_rate"),
        "call_rate": player_data.get("call_rate"),
        "raise_rate": player_data.get("raise_rate"),
        "aggression_factor": player_data.get("aggression_factor"),
        "failed_parses": player_data.get("failed_parses"),
        "actions_by_street": player_data.get("actions_by_street", {}) or {},
        "value_betting": player_data.get("value_betting", {}) or {},
        "bluffing": player_data.get("bluffing", {}) or {},
        "hand_strength_decisions": player_data.get("hand_strength_decisions", {}) or {},
        "opponent_profiles": opponent_profiles,
        "opponent_profile_history": profile_history,
        "reasoning_history": compact_reasoning_history,
    }


def compact_battle(battle, battle_index):
    players = {}
    for player_name, pdata in (battle.get("players", {}) or {}).items():
        players[player_name] = compact_player_summary(player_name, pdata)

    battle_rh = battle.get("reasoning_history", {}) or {}
    for player_name, rounds in battle_rh.items():
        if player_name not in players:
            players[player_name] = {
                "name": player_name,
                "reasoning_history": {},
                "opponent_profiles": {},
                "opponent_profile_history": {},
                "round_results": {},
                "initial_stack": None,
                "current_stack": None,
            }
        if not players[player_name].get("reasoning_history"):
            compact_reasoning_history = {}
            for rnd in sort_round_keys(rounds):
                streets = rounds.get(rnd, {}) or {}
                compact_reasoning_history[str(rnd)] = {}
                for street in sort_street_keys(streets):
                    compact_reasoning_history[str(rnd)][street] = compact_node(streets.get(street, {}) or {})
            players[player_name]["reasoning_history"] = compact_reasoning_history

    game_info = battle.get("game_info", {}) or {}

    return {
        "battle_index": battle_index,
        "game_info": {
            "timestamp": game_info.get("timestamp"),
            "rounds": game_info.get("rounds"),
            "initial_stack": game_info.get("initial_stack"),
            "small_blind": game_info.get("small_blind"),
            "big_blind": game_info.get(
                "big_blind",
                (game_info.get("small_blind") or 0) * 2 if game_info.get("small_blind") is not None else None,
            ),
        },
        "players": players,
    }


def make_compact_batch_data(full_data, battle_group, start_idx):
    compact_battles = []
    for local_i, battle in enumerate(battle_group):
        compact_battles.append(compact_battle(battle, battle_index=start_idx + local_i))
    return {
        "meta": {
            "source_file": str(SRC),
            "battles_in_file": len(compact_battles),
            "battle_index_start": start_idx,
            "battle_index_end": start_idx + len(compact_battles) - 1,
        },
        "battles": compact_battles,
    }


def build_html(compact_data):
    data_json = json.dumps(compact_data, ensure_ascii=False)

    html_text = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Poker Timeline Replay</title>
<style>
  :root{
    --bg:#071019;
    --panel:#0d1b2a;
    --panel2:#13263b;
    --line:#24415f;
    --text:#edf4ff;
    --muted:#9bb4cf;
    --accent:#63d471;
    --accent2:#4da3ff;
    --warn:#ffcc66;
    --danger:#ff6b6b;
    --table:#0f5c3a;
    --tableEdge:#083c25;
  }
  *{box-sizing:border-box}
  body{
    margin:0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    background: radial-gradient(circle at top, #112238 0%, #071019 50%, #050b12 100%);
    color:var(--text);
  }
  .app{
    display:grid;
    grid-template-columns: 340px 1fr;
    min-height:100vh;
  }
  .sidebar{
    background:rgba(7,16,25,0.94);
    border-right:1px solid var(--line);
    padding:18px;
    position:sticky;
    top:0;
    height:100vh;
    overflow:auto;
  }
  .main{ padding:18px; }
  .panel{
    background:rgba(13,27,42,0.92);
    border:1px solid var(--line);
    border-radius:20px;
    padding:16px;
    margin-bottom:16px;
    box-shadow:0 12px 32px rgba(0,0,0,0.25);
  }
  h1,h2,h3,h4{margin:0 0 10px}
  .sub{color:var(--muted); font-size:13px; line-height:1.5}
  .label{font-size:12px; color:var(--muted); margin:10px 0 6px}
  select, button{
    background:#0b1524;
    color:var(--text);
    border:1px solid var(--line);
    border-radius:12px;
    padding:10px 12px;
    font-size:14px;
  }
  select{width:100%}
  button{cursor:pointer}
  .btnrow{display:flex; gap:8px; flex-wrap:wrap}
  .btnrow button{flex:1}
  .grid2{display:grid; grid-template-columns: 1.15fr .85fr; gap:16px;}
  .grid3{display:grid; grid-template-columns: repeat(3,1fr); gap:12px}
  .metric{
    background:var(--panel2);
    border:1px solid var(--line);
    border-radius:16px;
    padding:12px;
  }
  .metric .k{font-size:12px; color:var(--muted)}
  .metric .v{font-size:22px; font-weight:800; margin-top:5px}
  .timeline{
    display:flex; gap:8px; flex-wrap:wrap;
  }
  .step{
    border:1px solid var(--line);
    background:#0b1524;
    border-radius:12px;
    padding:10px 12px;
    cursor:pointer;
    min-width:140px;
  }
  .step.active{
    border-color:var(--accent2);
    box-shadow:0 0 0 2px rgba(77,163,255,0.18) inset;
  }
  .street-badge{
    display:inline-block;
    background:#112238;
    border:1px solid var(--line);
    color:var(--accent2);
    border-radius:999px;
    padding:4px 8px;
    font-size:12px;
    margin-right:6px;
  }
  .table-wrap{
    min-height:540px;
    display:flex;
    align-items:center;
    justify-content:center;
  }
  .poker-table{
    width:min(980px, 100%);
    aspect-ratio: 16 / 9;
    position:relative;
    border-radius:999px;
    background: radial-gradient(circle at center, #117245 0%, var(--table) 60%, var(--tableEdge) 100%);
    border:10px solid #4a2a12;
    box-shadow: inset 0 0 60px rgba(0,0,0,0.25), 0 20px 40px rgba(0,0,0,0.35);
    overflow:hidden;
  }
  #seats{
    position:absolute;
    inset:0;
    z-index:3;
  }
  .seat{
    position:absolute;
    width:92px;
    min-height:64px;
    transform:translate(-50%, -50%);
    background:rgba(16,33,49,0.92);
    border:1px solid var(--line);
    border-radius:14px;
    padding:6px;
    box-shadow:0 8px 18px rgba(0,0,0,0.25);
    cursor:pointer;
    transition:transform .12s ease, border-color .12s ease, box-shadow .12s ease, opacity .12s ease;
  }
  .seat:hover{
    transform:translate(-50%, -50%) scale(1.03);
  }
  .seat.focus{
    border-color:#ffcc66;
    box-shadow:0 0 0 2px rgba(255,204,102,0.18) inset;
  }
  .seat.out{
    opacity:0.42;
    filter:grayscale(0.35);
  }
  .seat .name{
    font-size:10px;
    font-weight:700;
    line-height:1.15;
  }
  .seat .stack{
    font-size:9px;
    color:var(--muted);
    margin-top:2px;
    line-height:1.1;
  }
  .seat .tag{
    font-size:8px;
    color:#ffd36a;
    margin-top:3px;
    line-height:1.1;
  }
  .pot-box{
    position:absolute;
    top:16%;
    left:50%;
    transform:translateX(-50%);
    z-index:4;
    background:rgba(7,16,25,0.92);
    border:1px solid var(--line);
    border-radius:14px;
    padding:10px 14px;
    text-align:center;
  }
  .board{
    position:absolute;
    top:48%;
    left:50%;
    transform:translate(-50%, -50%);
    display:flex;
    gap:6px;
    z-index:4;
  }
  .hero-cards{
    position:absolute;
    top:70%;
    left:50%;
    transform:translate(-50%, -50%);
    display:flex;
    gap:6px;
    z-index:4;
  }
  .card, .empty{
    width:54px;
    height:74px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:800;
    border:1px solid #d8dde6;
    background:#fff;
    color:#111;
    box-shadow:0 6px 12px rgba(0,0,0,0.2);
  }
  .card.red{color:#d22}
  .empty{
    background:rgba(255,255,255,0.15);
    border-color:rgba(255,255,255,0.18);
    color:#fff;
  }
  .pill{
    display:inline-block;
    padding:6px 10px;
    border-radius:999px;
    background:#112238;
    border:1px solid var(--line);
    margin:0 8px 8px 0;
    font-size:12px;
  }
  pre{
    white-space:pre-wrap;
    word-break:break-word;
    background:#08111d;
    border:1px solid var(--line);
    padding:12px;
    border-radius:12px;
    color:#dfe9f7;
    max-height:420px;
    overflow:auto;
  }
  .actions .item,
  .profile .item{
    display:flex;
    justify-content:space-between;
    gap:10px;
    align-items:flex-start;
    padding:10px 12px;
    border-bottom:1px solid rgba(255,255,255,0.06);
  }
  .actions .item:last-child,
  .profile .item:last-child{border-bottom:none}
  .profile{
    background:#0b1524;
    border:1px solid var(--line);
    border-radius:14px;
    overflow:hidden;
    margin-bottom:12px;
  }
  .bar{
    height:8px;
    border-radius:999px;
    background:#14283e;
    overflow:hidden;
    margin-top:6px;
  }
  .bar > span{
    display:block;
    height:100%;
    background:linear-gradient(90deg, #4da3ff, #63d471);
  }
  .error-box{
    background:#2a1212;
    border:1px solid #8b3a3a;
    color:#ffd2d2;
    border-radius:12px;
    padding:12px;
    white-space:pre-wrap;
  }
  .inspector-head{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    flex-wrap:wrap;
  }
  canvas{
    width:100%;
    height:300px;
    display:block;
    background:#08111d;
    border:1px solid var(--line);
    border-radius:12px;
  }
  @media (max-width: 980px){
    .app{grid-template-columns:1fr}
    .sidebar{position:relative; height:auto; border-right:none; border-bottom:1px solid var(--line)}
    .grid2{grid-template-columns:1fr}
    .seat{
      width:84px;
      min-height:60px;
      padding:5px;
    }
    .seat .name{font-size:9px}
    .seat .stack{font-size:8px}
    .seat .tag{font-size:7px}
  }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1 style="font-size:22px;">Poker Timeline Replay</h1>
    <div class="sub" id="meta"></div>

    <div class="label">Battle</div>
    <select id="battleSelect"></select>

    <div class="label">Focused Player</div>
    <select id="playerSelect"></select>

    <div class="panel" style="margin-top:16px;">
      <h3 style="font-size:16px;">Playback</h3>
      <div class="btnrow" style="margin-bottom:10px;">
        <button id="prevBtn">◀ Previous</button>
        <button id="playBtn">▶ Play</button>
        <button id="nextBtn">Next ▶</button>
      </div>
      <div class="label">Playback Speed</div>
      <select id="speedSelect">
        <option value="1800">Slow</option>
        <option value="1100" selected>Medium</option>
        <option value="650">Fast</option>
      </select>
      <div class="sub" style="margin-top:10px;">
        The first moment of each round always shows the full battle roster. Later moments update actions and stacks, and players can drop off the table.
      </div>
    </div>

    <div class="panel">
      <h3 style="font-size:16px;">Current Moment</h3>
      <div id="stepInfo" class="sub"></div>
    </div>
  </aside>

  <main class="main">
    <div class="panel">
      <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;">
        <div>
          <h2 id="title">Poker Timeline Replay</h2>
          <div class="sub" id="subtitle"></div>
        </div>
        <div id="streetPills"></div>
      </div>
      <div class="grid3" id="summaryMetrics" style="margin-top:14px;"></div>
    </div>

    <div class="panel">
      <h3>Timeline</h3>
      <div id="timeline" class="timeline"></div>
    </div>

    <div class="panel">
      <h3>Table View</h3>
      <div class="table-wrap">
        <div class="poker-table">
          <div id="potBox" class="pot-box"></div>
          <div id="board" class="board"></div>
          <div id="heroCards" class="hero-cards"></div>
          <div id="seats"></div>
        </div>
      </div>
    </div>

    <div class="grid2">
      <div>
        <div class="panel">
          <div class="inspector-head">
            <h3>Focused Player Inspector</h3>
            <div class="sub" id="focusInfo"></div>
          </div>
          <div id="beliefPills" style="margin-top:10px; margin-bottom:10px;"></div>
          <pre id="reasoning"></pre>
        </div>

        <div class="panel">
          <h3>Action Context</h3>
          <div id="actionsContainer" class="actions"></div>
        </div>

        <div class="panel">
          <h3>State Information</h3>
          <div id="stateFacts" class="sub"></div>
        </div>
      </div>

      <div>
        <div class="panel">
          <h3>Chip Stack Trend (Focused Player)</h3>
          <canvas id="stackChart"></canvas>
          <div id="chartFallback" class="sub" style="margin-top:8px;"></div>
        </div>

        <div class="panel">
          <h3>Opponent Profile Snapshot</h3>
          <div id="profiles"></div>
        </div>
      </div>
    </div>

    <div id="renderError" style="display:none;" class="error-box"></div>
  </main>
</div>

<script>
const DATA = __DATA__;
const suitMap = {"S":"♠","H":"♥","D":"♦","C":"♣"};
const STREET_ORDER = ["preflop","flop","turn","river","showdown"];
const PLAYER_IDENTITIES = {
  "Alex Chen": "gpt-4.1-mini",
  "Sarah Johnson": "llama-4-maverick",
  "Jessica Liu": "gemini-2.5-flash-lite",
  "Emily Zhang": "deepseek-v3.2",
  "Niko Grey": "grok-3-mini",
  "Robert Garcia": "qwen3-32b",
  "Lily Grant": "TightPassive",
  "Jade Park": "TightAggressive",
  "Noah Blake": "Maniac",
  "Noah Kim": "LoosePassive",
  "Ava Park": "LooseAggressive"
};

let state = {
  battleIndex: 0,
  timelineIndex: 0,
  focusPlayer: "",
  playing: false,
  timer: null
};

function esc(s){
  return String(s ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;");
}

function currentBattle(){
  return (DATA.battles || [])[state.battleIndex] || {};
}

function cardHTML(code){
  if(!code) return '<div class="empty">—</div>';
  const suit = code[0];
  const rank = code.slice(1);
  const red = (suit === "H" || suit === "D") ? " red" : "";
  return `<div class="card${red}">${esc(rank)}${esc(suitMap[suit] || suit)}</div>`;
}

function formatPlayerName(name){
  const identity = PLAYER_IDENTITIES[name];
  return identity ? `${name} (${identity})` : name;
}

function getBattlePlayers(){
  const battle = currentBattle();
  const players = battle.players || {};
  return Object.keys(players).sort();
}

function getPlayerSummary(name){
  const players = currentBattle().players || {};
  return players[name] || {};
}

function buildBattleTimeline(){
  const battle = currentBattle();
  const battlePlayers = battle?.players || {};
  const timelineMap = new Map();

  Object.entries(battlePlayers).forEach(([playerName, pdata]) => {
    const rh = pdata?.reasoning_history || {};
    Object.keys(rh).forEach(round => {
      const streets = rh[round] || {};
      Object.keys(streets).forEach(street => {
        const key = `${round}__${street}`;
        if(!timelineMap.has(key)){
          timelineMap.set(key, {
            round: Number(round),
            street,
            actors: {}
          });
        }
        timelineMap.get(key).actors[playerName] = streets[street];
      });
    });
  });

  return Array.from(timelineMap.values()).sort((a, b) => {
    if(a.round !== b.round) return a.round - b.round;
    return STREET_ORDER.indexOf(a.street) - STREET_ORDER.indexOf(b.street);
  });
}

function currentMoment(){
  const timeline = buildBattleTimeline();
  if(!timeline.length) return null;
  if(state.timelineIndex < 0) state.timelineIndex = 0;
  if(state.timelineIndex >= timeline.length) state.timelineIndex = timeline.length - 1;
  return timeline[state.timelineIndex];
}

function getMomentPlayers(){
  const moment = currentMoment();
  if(!moment) return [];
  return Object.keys(moment.actors || {}).sort();
}

function currentFocusNode(){
  const moment = currentMoment();
  if(!moment || !state.focusPlayer) return null;
  return moment.actors?.[state.focusPlayer] || null;
}

function getMomentGameState(){
  const moment = currentMoment();
  if(!moment) return null;

  if(state.focusPlayer && moment.actors?.[state.focusPlayer]?.game_state){
    return moment.actors[state.focusPlayer].game_state;
  }

  const actors = Object.values(moment.actors || {});
  return actors[0]?.game_state || null;
}

function ensureFocusPlayer(){
  const battlePlayers = getBattlePlayers();
  if(!battlePlayers.length){
    state.focusPlayer = "";
    return;
  }

  const momentPlayers = getMomentPlayers();
  if(momentPlayers.length){
    if(!momentPlayers.includes(state.focusPlayer)){
      state.focusPlayer = momentPlayers[0];
    }
  } else if(!battlePlayers.includes(state.focusPlayer)){
    state.focusPlayer = battlePlayers[0];
  }
}

function isFirstMomentOfRound(timeline, idx){
  if(idx <= 0) return true;
  return timeline[idx].round !== timeline[idx - 1].round;
}

function buildRoundInitialRosterMap(){
  const battle = currentBattle();
  const players = battle?.players || {};
  const roster = {};

  Object.entries(players).forEach(([name, pdata], idx) => {
    roster[name] = {
      name,
      initial_stack: pdata?.initial_stack ?? battle?.game_info?.initial_stack ?? "-",
      current_stack: pdata?.current_stack ?? pdata?.initial_stack ?? "-",
      seat_hint: idx + 1
    };
  });

  return roster;
}

function computeRoundBaseRoster(timeline, idx){
  const roundRoster = buildRoundInitialRosterMap();
  const roundNum = timeline[idx]?.round;
  for(let i = 0; i < idx; i++){
    if(timeline[i].round !== roundNum) continue;
    const gs = getGameStateFromMoment(timeline[i]);
    const stacks = gs?.players_stacks || {};
    Object.entries(stacks).forEach(([name, stack]) => {
      if(roundRoster[name]){
        roundRoster[name].current_stack = stack;
      }
    });
  }
  return roundRoster;
}

function getGameStateFromMoment(moment){
  if(!moment) return null;
  if(state.focusPlayer && moment.actors?.[state.focusPlayer]?.game_state){
    return moment.actors[state.focusPlayer].game_state;
  }
  const actors = Object.values(moment.actors || {});
  return actors[0]?.game_state || null;
}

function normalizeActionLabel(actionObj){
  const act = String(actionObj?.action || "").toUpperCase().trim();
  if(!act) return "";
  if(act === "FOLD") return "fold";
  if(act === "CALL") return "call";
  if(act === "CHECK") return "check";
  if(act === "RAISE"){
    const amt = actionObj?.amount;
    if(amt == null || amt === "") return "raise";
    return `raise ${amt}`;
  }
  return act.toLowerCase();
}

function buildCurrentMomentActionMap(){
  const moment = currentMoment();
  const actionMap = new Map();
  if(!moment) return actionMap;

  Object.entries(moment.actors || {}).forEach(([playerName, node]) => {
    const label = normalizeActionLabel(node?.action);
    if(label){
      actionMap.set(playerName, `${moment.street}: ${label}`);
    }
  });

  return actionMap;
}

function getDynamicSeatCoord(displayIndex, totalSeats, isFocused=false){
  if(isFocused){
    return [50, 88];
  }

  const otherCount = Math.max(0, totalSeats - 1);
  if(otherCount === 0){
    return [50, 50];
  }

  const startDeg = 150;
  const endDeg = 390;

  let t = 0.5;
  if(otherCount > 1){
    t = (displayIndex - 1) / (otherCount - 1);
  }

  const deg = startDeg + (endDeg - startDeg) * t;
  const rad = deg * Math.PI / 180;

  const cx = 50;
  const cy = 46;
  const rx = 44;
  const ry = 38;

  const x = cx + rx * Math.cos(rad);
  const y = cy + ry * Math.sin(rad);

  return [x, y];
}

function stableNonFocusedSort(a, b){
  return String(a.name).localeCompare(String(b.name));
}

function fillBattleSelect(){
  const sel = document.getElementById("battleSelect");
  const battles = DATA.battles || [];
  sel.innerHTML = battles.map((b, i)=>{
    const gi = b.game_info || {};
    return `<option value="${i}">Battle ${i} · rounds ${esc(gi.rounds ?? "-")}</option>`;
  }).join("");
  sel.value = String(state.battleIndex);
  sel.onchange = () => {
    state.battleIndex = Number(sel.value);
    state.timelineIndex = 0;
    stopAutoPlay();
    ensureFocusPlayer();
    fillPlayerSelect();
    render();
  };
}

function fillPlayerSelect(){
  ensureFocusPlayer();
  const sel = document.getElementById("playerSelect");
  const players = getMomentPlayers().length ? getMomentPlayers() : getBattlePlayers();
  sel.innerHTML = players.map(p => `<option value="${esc(p)}">${esc(formatPlayerName(p))}</option>`).join("");
  sel.value = state.focusPlayer;
  sel.onchange = () => {
    state.focusPlayer = sel.value;
    render();
  };
}

function renderMeta(){
  const meta = DATA.meta || {};
  const battles = DATA.battles || [];
  document.getElementById("meta").innerHTML =
    `This file contains <strong>${battles.length}</strong> battles<br>` +
    `battle index: <strong>${esc(meta.battle_index_start ?? "-")} ~ ${esc(meta.battle_index_end ?? "-")}</strong>`;
}

function renderHeader(){
  const battle = currentBattle();
  const gi = battle.game_info || {};
  document.getElementById("title").textContent =
    `Battle ${battle.battle_index ?? state.battleIndex} · Focus: ${state.focusPlayer ? formatPlayerName(state.focusPlayer) : "None"}`;
  document.getElementById("subtitle").textContent =
    `Rounds: ${gi.rounds ?? "-"} | Initial Stack: ${gi.initial_stack ?? "-"} | SB/BB: ${gi.small_blind ?? "-"} / ${gi.big_blind ?? "-"}`;
}

function renderSummaryMetrics(){
  const p = getPlayerSummary(state.focusPlayer);
  const metrics = [
    ["Current Stack", p.current_stack],
    ["P/L", p.total_profit_loss],
    ["Win Rate", p.win_rate != null ? `${p.win_rate}%` : "-"],
    ["Fold Rate", p.fold_rate != null ? `${p.fold_rate}%` : "-"],
    ["Call Rate", p.call_rate != null ? `${p.call_rate}%` : "-"],
    ["Raise Rate", p.raise_rate != null ? `${p.raise_rate}%` : "-"],
  ];
  document.getElementById("summaryMetrics").innerHTML = metrics.map(([k,v]) => `
    <div class="metric">
      <div class="k">${esc(k)}</div>
      <div class="v">${esc(v ?? "-")}</div>
    </div>
  `).join("");
}

function renderTimeline(){
  const timelineData = buildBattleTimeline();
  const html = timelineData.map((s, idx)=>{
    const firstOfRound = isFirstMomentOfRound(timelineData, idx);
    return `
      <div class="step ${idx===state.timelineIndex?'active':''}" data-idx="${idx}">
        <div><strong>Round ${esc(s.round)}</strong></div>
        <div class="sub">${esc(s.street)}</div>
        <div class="sub" style="margin-top:4px;">Actors: ${Object.keys(s.actors || {}).length}</div>
        <div class="sub">${firstOfRound ? "Full roster frame" : "Update frame"}</div>
      </div>
    `;
  }).join("");

  const timeline = document.getElementById("timeline");
  timeline.innerHTML = html || '<div class="sub">No replayable timeline moments available</div>';
  timeline.querySelectorAll(".step").forEach(el=>{
    el.onclick = () => {
      state.timelineIndex = Number(el.getAttribute("data-idx"));
      stopAutoPlay();
      ensureFocusPlayer();
      fillPlayerSelect();
      render();
    };
  });

  const moment = currentMoment();
  document.getElementById("stepInfo").innerHTML = moment
    ? `Round <strong>${esc(moment.round)}</strong><br>Street <strong>${esc(moment.street)}</strong><br>Moment <strong>${state.timelineIndex + 1}</strong> / ${timelineData.length}<br>Focused player: <strong>${esc(state.focusPlayer ? formatPlayerName(state.focusPlayer) : "-")}</strong>`
    : `No timeline moments`;
}

function renderStreetPills(){
  const moment = currentMoment();
  const street = moment?.street || "-";
  document.getElementById("streetPills").innerHTML = `<span class="street-badge">${esc(street)}</span>`;
}

function buildSeatPlayers(gs){
  const timeline = buildBattleTimeline();
  const idx = state.timelineIndex;
  const battle = currentBattle();
  const battlePlayers = battle?.players || {};
  const stepPos = gs?.position_info || {};
  const stepPlayersInfo = Array.isArray(stepPos.players_info) ? stepPos.players_info : [];
  const stepStacks = gs?.players_stacks || {};
  const moment = currentMoment();
  const momentActors = new Set(Object.keys(moment?.actors || {}));

  const firstOfRound = isFirstMomentOfRound(timeline, idx);
  const roundBase = computeRoundBaseRoster(timeline, idx);

  const merged = new Map();

  if(firstOfRound){
    Object.entries(roundBase).forEach(([name, info]) => {
      merged.set(name, {
        name,
        stack: info.current_stack ?? info.initial_stack ?? "-",
        is_dealer: false,
        is_focus: name === state.focusPlayer,
        is_tracked: !!battlePlayers[name],
        is_out: false,
      });
    });
  } else {
    const visibleNow = new Set([state.focusPlayer]);
    stepPlayersInfo.forEach(p => {
      if(p?.name) visibleNow.add(p.name);
    });
    Object.keys(stepStacks).forEach(name => visibleNow.add(name));
    momentActors.forEach(name => visibleNow.add(name));

    visibleNow.forEach(name => {
      const pdata = battlePlayers[name] || {};
      merged.set(name, {
        name,
        stack: pdata?.current_stack ?? pdata?.initial_stack ?? "-",
        is_dealer: false,
        is_focus: name === state.focusPlayer,
        is_tracked: !!battlePlayers[name],
        is_out: false,
      });
    });
  }

  stepPlayersInfo.forEach((p) => {
    if(!p || !p.name) return;
    const old = merged.get(p.name) || {
      name: p.name,
      stack: "-",
      is_dealer: false,
      is_focus: p.name === state.focusPlayer,
      is_tracked: !!battlePlayers[p.name],
      is_out: false,
    };

    merged.set(p.name, {
      ...old,
      stack: p.stack ?? stepStacks[p.name] ?? old.stack,
      is_dealer: !!p.is_dealer || old.is_dealer,
      is_focus: p.name === state.focusPlayer,
      is_out: Number(p.stack ?? stepStacks[p.name] ?? old.stack) <= 0,
    });
  });

  Object.entries(stepStacks).forEach(([name, stack]) => {
    const old = merged.get(name) || {
      name,
      stack: "-",
      is_dealer: false,
      is_focus: name === state.focusPlayer,
      is_tracked: !!battlePlayers[name],
      is_out: false,
    };

    merged.set(name, {
      ...old,
      stack: stack ?? old.stack,
      is_focus: name === state.focusPlayer,
      is_out: Number(stack) <= 0,
    });
  });

  let result = Array.from(merged.values());

  if(!firstOfRound){
    result = result.filter(p => {
      if(p.name === state.focusPlayer) return true;
      const stackNum = Number(p.stack);
      if(Number.isFinite(stackNum) && stackNum <= 0) return false;
      return true;
    });
  }

  const focused = result.find(p => p.name === state.focusPlayer);
  const others = result.filter(p => p.name !== state.focusPlayer).sort(stableNonFocusedSort);
  return focused ? [focused, ...others] : others;
}

function renderTable(){
  const gs = getMomentGameState() || {};
  const currentActionMap = buildCurrentMomentActionMap();

  document.getElementById("potBox").innerHTML =
    `<div class="sub">Pot</div><div style="font-weight:800; font-size:20px;">${esc(gs.pot_size ?? "-")}</div>`;

  const board = [...(gs.community_cards || [])];
  while(board.length < 5) board.push(null);
  document.getElementById("board").innerHTML = board.map(cardHTML).join("");

  const focusNode = currentFocusNode() || {};
  const focusGS = focusNode.game_state || {};
  const hole = [...(focusGS.hole_cards || [])];
  while(hole.length < 2) hole.push(null);
  document.getElementById("heroCards").innerHTML = hole.map(cardHTML).join("");

  const seatPlayers = buildSeatPlayers(gs);
  const battlePlayers = currentBattle()?.players || {};
  const totalSeats = seatPlayers.length;

  const seatsHtml = seatPlayers.map((p, displayIndex)=>{
    const xy = getDynamicSeatCoord(displayIndex, totalSeats, p.name === state.focusPlayer);
    const tag = currentActionMap.get(p.name);
    const fallbackTag = p.name === state.focusPlayer
      ? "focused"
      : (battlePlayers[p.name] ? "LLM player" : "bot");

    const cls = `${p.name === state.focusPlayer ? 'focus' : ''} ${p.is_out ? 'out' : ''}`.trim();

    return `
      <div class="seat ${cls}" data-player="${esc(p.name)}" style="left:${xy[0]}%; top:${xy[1]}%;">
        <div class="name">${esc(formatPlayerName(p.name))} ${p.is_dealer ? '🃏' : ''}</div>
        <div class="stack">stack: ${esc(p.stack ?? '-')}</div>
        <div class="tag">${esc(tag || fallbackTag)}</div>
      </div>
    `;
  }).join("");

  const seatsBox = document.getElementById("seats");
  seatsBox.innerHTML = seatsHtml || `<div class="sub" style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);">No seat data</div>`;

  seatsBox.querySelectorAll(".seat").forEach(el => {
    el.onclick = () => {
      const name = el.getAttribute("data-player");
      if(name){
        state.focusPlayer = name;
        fillPlayerSelect();
        renderInspector();
        renderTable();
        renderSummaryMetrics();
        renderHeader();
        renderStepFocusInfo();
        renderChart();
      }
    };
  });
}

function renderStepFocusInfo(){
  const moment = currentMoment();
  document.getElementById("focusInfo").textContent =
    moment ? `Round ${moment.round} · ${moment.street} · ${state.focusPlayer ? formatPlayerName(state.focusPlayer) : "No player selected"}` : "No focused node";
}

function renderReasoning(){
  const node = currentFocusNode() || {};
  const beliefs = node.beliefs || {};
  const chosen = node.chosen_action_summary || {};
  let pills = "";
  Object.entries(beliefs).forEach(([k,v]) => pills += `<span class="pill">${esc(k)}: ${esc(v)}</span>`);
  Object.entries(chosen).forEach(([k,v]) => pills += `<span class="pill">${esc(k)}: ${esc(v)}</span>`);
  document.getElementById("beliefPills").innerHTML =
    pills || `<span class="sub">This node has no structured beliefs / chosen action</span>`;

  const parsed = node?.action ? `\n\n[Parsed Action]\n${JSON.stringify(node.action, null, 2)}` : "";
  document.getElementById("reasoning").textContent = (node.reasoning || "No reasoning for the focused player at this moment") + parsed;
}

function renderActions(){
  const gs = getMomentGameState() || {};
  const actions = gs?.opponent_actions || {};
  const container = document.getElementById("actionsContainer");
  const html = Object.entries(actions)
    .sort((a,b)=>STREET_ORDER.indexOf(a[0])-STREET_ORDER.indexOf(b[0]))
    .map(([street, arr]) => `
      <div style="margin-bottom:14px;">
        <div class="street-badge">${esc(street)}</div>
        <div class="profile" style="padding:0;">
          ${(arr || []).map(a=>`
            <div class="item">
              <div><strong>${esc(formatPlayerName(a.player))}</strong><div class="sub">stack: ${esc(a.stack)}</div></div>
              <div>${esc(a.action)}</div>
            </div>
          `).join("") || '<div class="item"><div class="sub">No actions</div></div>'}
        </div>
      </div>
    `).join("");
  container.innerHTML = html || '<div class="sub">No action context at this moment</div>';
}

function renderStateFacts(){
  const node = currentFocusNode() || {};
  const gs = node?.game_state || {};
  const pos = gs.position_info || {};
  const items = [
    ["Pot Size", gs.pot_size],
    ["Call Amount", gs.call_amount],
    ["Min Raise", gs.min_raise],
    ["Max Raise", gs.max_raise],
    ["Pot Odds", gs.pot_odds],
    ["Street", gs.street],
    ["Env Hand Type", gs.env_hand_type],
    ["Env Strength", gs.env_hand_strength],
    ["LLM Label", gs.llm_hand_strength_label],
    ["My Position", pos.my_position],
    ["My Stack", pos.my_stack],
    ["Total Players (local node)", pos.total_players]
  ];
  document.getElementById("stateFacts").innerHTML = items.map(([k,v]) =>
    `<div style="margin-bottom:6px;"><strong>${esc(k)}:</strong> ${esc(v ?? "-")}</div>`
  ).join("") + `<div style="margin-top:8px;"><strong>Players at table:</strong><br/>${
    (buildSeatPlayers(getMomentGameState() || {}) || []).map(p =>
      `${esc(formatPlayerName(p.name))} (${esc(p.stack)})${p.name === state.focusPlayer ? " [focused]" : ""}${p.is_out ? " [out]" : ""}${p.is_dealer ? " [dealer]" : ""}`
    ).join("<br/>") || "-"
  }</div>`;
}

function renderProfiles(){
  const player = getPlayerSummary(state.focusPlayer);
  const node = currentFocusNode() || {};
  let snapshot = node?.opponent_profiles_snapshot || {};

  if((!snapshot || Object.keys(snapshot).length === 0) && player?.opponent_profiles){
    snapshot = player.opponent_profiles;
  }

  if((!snapshot || Object.keys(snapshot).length === 0) && player?.opponent_profile_history){
    snapshot = {};
    Object.entries(player.opponent_profile_history).forEach(([opp, arr])=>{
      const found = (arr || []).find(x => String(x.round) === String(currentMoment()?.round));
      if(found?.profile) snapshot[opp] = found.profile;
    });
  }

  function block(name, profile){
    const traits = profile?.Traits || {};
    const summary = profile?.QualitativeSummary || "";
    const rationale = profile?.UpdateRationale || "";
    const lines = Object.entries(traits).map(([k,v])=>{
      const pct = Math.max(0, Math.min(100, Number(v) * 100));
      return `
        <div style="margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; gap:12px;">
            <span>${esc(k)}</span><span>${esc(Number(v).toFixed ? Number(v).toFixed(2) : v)}</span>
          </div>
          <div class="bar"><span style="width:${pct}%;"></span></div>
        </div>
      `;
    }).join("");
    return `
      <div class="profile">
        <div class="item"><strong>${esc(formatPlayerName(name))}</strong></div>
        <div style="padding:12px;">
          ${lines || '<div class="sub">No traits</div>'}
          <div class="sub" style="margin-top:8px;"><strong>Summary:</strong> ${esc(summary)}</div>
          <div class="sub" style="margin-top:8px;"><strong>Rationale:</strong> ${esc(rationale)}</div>
        </div>
      </div>
    `;
  }

  const html = Object.entries(snapshot || {}).map(([name, profile]) => block(name, profile)).join("");
  document.getElementById("profiles").innerHTML = html || '<div class="sub">No opponent profile snapshot for the focused player at this moment</div>';
}

function setupHiDPICanvas(canvas){
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(300, Math.round(rect.width || 800));
  const cssHeight = Math.max(220, Math.round(rect.height || 300));
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, W: cssWidth, H: cssHeight};
}

function buildRoundSeries(p){
  const rr = p.round_results || {};
  const keys = Object.keys(rr).map(x => Number(x)).filter(x => Number.isFinite(x)).sort((a,b)=>a-b);
  if(!keys.length) return null;

  const minRound = keys[0];
  const maxRound = keys[keys.length - 1];
  const labels = [];
  const points = [];
  let cur = Number(p.initial_stack ?? 0);

  for(let r = minRound; r <= maxRound; r++){
    labels.push(r);
    cur += Number(rr[String(r)] ?? rr[r] ?? 0);
    points.push(cur);
  }

  return {labels, points};
}

function computeYAxis(points, initial){
  const minY0 = Math.min(...points, initial);
  const maxY0 = Math.max(...points, initial);
  const span0 = Math.max(1, maxY0 - minY0);
  const center = (minY0 + maxY0) / 2;

  let yMin, yMax;
  if(span0 < 40){
    yMin = center - 30;
    yMax = center + 30;
  } else if(span0 < 80){
    yMin = center - 50;
    yMax = center + 50;
  } else if(span0 < 140){
    yMin = center - 80;
    yMax = center + 80;
  } else {
    yMin = minY0 - span0 * 0.18;
    yMax = maxY0 + span0 * 0.18;
  }

  if(yMax <= yMin){
    yMax = yMin + 1;
  }
  return {yMin, yMax};
}

function renderChart(){
  const canvas = document.getElementById("stackChart");
  const fallback = document.getElementById("chartFallback");
  const p = getPlayerSummary(state.focusPlayer);
  const series = buildRoundSeries(p);

  const {ctx, W, H} = setupHiDPICanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#08111d";
  ctx.fillRect(0, 0, W, H);

  if(!series){
    fallback.textContent = "No round_results available; unable to draw the chart.";
    return;
  }

  const labels = series.labels;
  const points = series.points;
  const initial = Number(p.initial_stack ?? 0);
  fallback.textContent = "";

  const padL = 62, padR = 18, padT = 18, padB = 42;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const {yMin, yMax} = computeYAxis(points, initial);

  function xAtIndex(i){
    if(labels.length === 1) return padL + plotW / 2;
    return padL + plotW * i / (labels.length - 1);
  }

  function yScale(v){
    return padT + (yMax - v) / (yMax - yMin) * plotH;
  }

  ctx.strokeStyle = "#16314a";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  const gridCount = 3;
  for(let i = 0; i <= gridCount; i++){
    const y = padT + plotH * i / gridCount;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.strokeStyle = i === gridCount ? "#24415f" : "rgba(64,116,168,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    const val = yMax - (yMax - yMin) * i / gridCount;
    ctx.fillStyle = "#9bb4cf";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(Math.round(val), padL - 8, y);
  }

  const maxXTicks = 12;
  const xStep = Math.max(1, Math.ceil(labels.length / maxXTicks));

  labels.forEach((lab, i)=>{
    const x = xAtIndex(i);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, H - padB);
    ctx.strokeStyle = "rgba(64,116,168,0.18)";
    ctx.lineWidth = 1;
    ctx.stroke();

    if(i % xStep === 0 || i === labels.length - 1){
      ctx.fillStyle = "#9bb4cf";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(String(lab), x, H - padB + 10);
    }
  });

  const yInit = yScale(initial);
  ctx.beginPath();
  ctx.moveTo(padL, yInit);
  ctx.lineTo(W - padR, yInit);
  ctx.strokeStyle = "rgba(255,204,102,0.55)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  const moment = currentMoment();
  if(moment){
    const idx = labels.findIndex(x => x === Number(moment.round));
    if(idx >= 0){
      const x = xAtIndex(idx);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, H - padB);
      ctx.strokeStyle = "rgba(99,212,113,0.55)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  ctx.strokeStyle = "#00d2ff";
  ctx.lineWidth = 3;
  ctx.beginPath();

  let prevY = null;
  points.forEach((v, i)=>{
    const x = xAtIndex(i);
    const y = yScale(v);
    if(i === 0){
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, prevY);
      ctx.lineTo(x, y);
    }
    prevY = y;
  });
  ctx.stroke();

  ctx.save();
  ctx.beginPath();
  prevY = null;
  points.forEach((v, i)=>{
    const x = xAtIndex(i);
    const y = yScale(v);
    if(i === 0){
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, prevY);
      ctx.lineTo(x, y);
    }
    prevY = y;
  });
  const lastX = xAtIndex(points.length - 1);
  const firstX = xAtIndex(0);
  ctx.lineTo(lastX, H - padB);
  ctx.lineTo(firstX, H - padB);
  ctx.closePath();

  const grad = ctx.createLinearGradient(0, padT, 0, H - padB);
  grad.addColorStop(0, "rgba(0,210,255,0.22)");
  grad.addColorStop(1, "rgba(0,210,255,0.02)");
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.restore();

  ctx.fillStyle = "#63f5a6";
  points.forEach((v, i)=>{
    const x = xAtIndex(i);
    const y = yScale(v);
    const shouldDraw = (i % 2 === 0) || i === 0 || i === points.length - 1;
    if(!shouldDraw) return;
    ctx.beginPath();
    ctx.arc(x, y, 3.6, 0, Math.PI * 2);
    ctx.fill();
  });

  if(moment){
    const idx = labels.findIndex(x => x === Number(moment.round));
    if(idx >= 0){
      const x = xAtIndex(idx);
      const y = yScale(points[idx]);
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#ffcc66";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,204,102,0.45)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

function renderInspector(){
  renderStepFocusInfo();
  renderReasoning();
  renderActions();
  renderStateFacts();
  renderProfiles();
  renderChart();
}

function stopAutoPlay(){
  state.playing = false;
  if(state.timer){
    clearInterval(state.timer);
    state.timer = null;
  }
  document.getElementById("playBtn").textContent = "▶ Play";
}

function startAutoPlay(){
  const timelineData = buildBattleTimeline();
  if(!timelineData.length) return;
  state.playing = true;
  document.getElementById("playBtn").textContent = "⏸ Pause";
  const delay = Number(document.getElementById("speedSelect").value || 1100);
  state.timer = setInterval(()=>{
    const tl = buildBattleTimeline();
    if(!tl.length){
      stopAutoPlay();
      return;
    }
    if(state.timelineIndex >= tl.length - 1){
      stopAutoPlay();
      return;
    }
    state.timelineIndex += 1;
    ensureFocusPlayer();
    fillPlayerSelect();
    render();
  }, delay);
}

function bindButtons(){
  document.getElementById("prevBtn").onclick = () => {
    stopAutoPlay();
    state.timelineIndex = Math.max(0, state.timelineIndex - 1);
    ensureFocusPlayer();
    fillPlayerSelect();
    render();
  };
  document.getElementById("nextBtn").onclick = () => {
    stopAutoPlay();
    const tl = buildBattleTimeline();
    if(!tl.length) return;
    state.timelineIndex = Math.min(tl.length - 1, state.timelineIndex + 1);
    ensureFocusPlayer();
    fillPlayerSelect();
    render();
  };
  document.getElementById("playBtn").onclick = () => {
    if(state.playing) stopAutoPlay();
    else startAutoPlay();
  };
  document.getElementById("speedSelect").onchange = () => {
    if(state.playing){
      stopAutoPlay();
      startAutoPlay();
    }
  };
}

function showError(err){
  const box = document.getElementById("renderError");
  box.style.display = "block";
  box.textContent = "Render Error:\n" + String(err?.stack || err);
}

function render(){
  try{
    document.getElementById("renderError").style.display = "none";
    ensureFocusPlayer();
    renderMeta();
    renderHeader();
    renderSummaryMetrics();
    renderTimeline();
    renderStreetPills();
    renderTable();
    renderInspector();
  }catch(err){
    showError(err);
    throw err;
  }
}

function init(){
  const battles = DATA.battles || [];
  if(!battles.length){
    document.body.innerHTML = '<div class="error-box" style="margin:20px;">No battles data.</div>';
    return;
  }
  fillBattleSelect();
  fillPlayerSelect();
  bindButtons();
  render();
  window.addEventListener("resize", renderChart);
}

init();
</script>
</body>
</html>
'''
    return html_text.replace("__DATA__", data_json)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = safe_read_json(SRC)
    battles = data.get("battles", []) or []
    if not battles:
        raise RuntimeError("No battles found in results.json")

    for batch_no, battle_group in chunk_list(battles, BATTLES_PER_HTML):
        start_idx = (batch_no - 1) * BATTLES_PER_HTML
        compact_batch = make_compact_batch_data(data, battle_group, start_idx=start_idx)
        html_text = build_html(compact_batch)

        out_path = OUT_DIR / f"poker_timeline_batch_{batch_no:03d}.html"
        out_path.write_text(html_text, encoding="utf-8")
        print(f"Saved: {out_path}")

    print(f"\nCompleted: Generated {math.ceil(len(battles) / BATTLES_PER_HTML)} HTML files.")
    print(f"Output directory: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

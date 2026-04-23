from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card
from openai import OpenAI
import json
import re
import os
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from treys import Card as TCard, Evaluator as TEvaluator, Deck as TDeck
import random


class LLMBasePlayer(BasePokerPlayer, ABC):
    """Base LLM poker player. Subclasses only implement `get_model_config`."""

    def __init__(self):
        super().__init__()
        self.initial_stack = 0
        self.current_stack = 0
        self.total_gains_losses = 0
        self.round_gains_losses = {}
        self.parse_failed = 0
        self.current_street = None
        self.current_hand_strength = 0
        self.current_hand_type = None
        self.reasoning_history = {}
        self.current_round_count = 0

        self.player_names = {}
        self.current_round_actions = {}
        self.seats_info = {}

        self.hand_stats = {
            'played': 0,
            'won': 0,
            'folded': 0,
            'called': 0,
            'raised': 0
        }
        self.actions_by_street = {
            'preflop': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0},
            'flop': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0},
            'turn': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0},
            'river': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0}
        }
        self.value_bets = {'made': 0, 'successful': 0, 'profit': 0}
        self.bluffs = {'made': 0, 'successful': 0, 'profit': 0}
        self.aggression_actions = {'raises': 0, 'calls': 0}
        self.hand_strength_actions = {
            'very_weak': {'fold': 0, 'call': 0, 'raise': 0},
            'weak': {'fold': 0, 'call': 0, 'raise': 0},
            'medium': {'fold': 0, 'call': 0, 'raise': 0},
            'strong': {'fold': 0, 'call': 0, 'raise': 0}
        }
        self.hand_strength_by_street = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'river': []
        }
        self.street_investment = {
            'preflop': 0,
            'flop': 0,
            'turn': 0,
            'river': 0
        }
        self.opponent_stats = {}
        self.opponent_profiles = {}
        self.opponent_profile_history = {}
        self.last_beliefs = None
        self.last_chosen_action_summary = None

        # Busted opponents are tracked by uuid and stop receiving profile updates.
        self.busted_opponents = set()
        self.profile_update_every_n_hands = 1
        self.profile_max_workers = 8
        self.profile_llm_timeout_sec = 12
        self.last_prompt_components = None

    @abstractmethod
    def get_model_config(self):
        """Return the model config dict: {model, use_complex_messages, extra_prompt?}."""
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        fold = valid_actions[0]
        call = valid_actions[1]
        raise_action = valid_actions[2]

        community_card = round_state['community_card']
        pot_size = round_state['pot']['main']['amount']
        street = round_state['street']

        current_highest_bet = 0
        my_current_bet = 0

        if street in round_state['action_histories']:
            for action_history in round_state['action_histories'][street]:
                if 'amount' in action_history:
                    current_highest_bet = max(current_highest_bet, action_history['amount'])

                if action_history['uuid'] == self.uuid and 'amount' in action_history:
                    my_current_bet = action_history['amount']

        call_amount = current_highest_bet - my_current_bet
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0

        min_raise = raise_action['amount']['min']
        max_raise = raise_action['amount']['max']

        try:
            seats = round_state.get("seats", [])
            active_opps = [
                s for s in seats
                if s.get("uuid") != self.uuid and s.get("state") == "participating"
            ]
            n_opponents = max(1, len(active_opps))

            # Sim count tuned per street: preflop has the highest variance.
            if street == "preflop":
                n_sim = 1200
            elif street == "flop":
                n_sim = 900
            elif street == "turn":
                n_sim = 600
            else:
                n_sim = 300

            self.current_hand_strength = self._estimate_equity_treys(
                hole_card=hole_card,
                community_card=community_card,
                n_opponents=n_opponents,
                n_sim=n_sim
            )

            try:
                hero = self._to_treys_cards(hole_card)
                board = self._to_treys_cards(community_card)
                if len(hero) == 2:
                    evaluator = TEvaluator()
                    if len(board) >= 3:
                        score_now = evaluator.evaluate(board, hero)
                        cls = evaluator.get_rank_class(score_now)
                        self.current_hand_type = evaluator.class_to_string(cls)
                    else:
                        self.current_hand_type = "PREFLOP"
                else:
                    self.current_hand_type = "UNKNOWN"
            except Exception:
                self.current_hand_type = "UNKNOWN"

        except Exception as e:
            print(f"[Equity evaluation error] {e}")
            self.current_hand_type = "UNKNOWN"
            self.current_hand_strength = 0.5

        if street in self.hand_strength_by_street:
            self.hand_strength_by_street[street].append(self.current_hand_strength)

        # Get position and table information
        position_info = self.get_position_info(round_state)

        # Get current round's action history
        opponent_actions = self.get_formatted_opponent_actions(round_state)

        game_state = {
            "hole_cards": hole_card,
            "community_cards": community_card,
            "pot_size": pot_size,
            "street": street,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "max_raise": max_raise,
            "pot_odds": pot_odds,
            "env_hand_type": self.current_hand_type,
            "env_hand_strength": self.current_hand_strength,
            "opponent_actions": opponent_actions,
            "position_info": position_info,
        }

        action_decision, reasoning = self.call_llm_api(game_state)

        # Allow the LLM to output CHECK; map it to CALL:0 when legal.
        action_decision, norm_msg = self._normalize_llm_decision(
            action_decision,
            call_amount=call_amount,
            min_raise=min_raise,
            max_raise=max_raise
        )
        reasoning = f"{reasoning}\n[NORMALIZE] {norm_msg}"

        beliefs = self.last_beliefs or {}
        hs_label = (beliefs.get("HandStrength") or "").lower() if beliefs else None

        game_state["env_hand_type"] = self.current_hand_type
        game_state["env_hand_strength"] = self.current_hand_strength
        game_state["llm_hand_strength_label"] = hs_label

        if self.current_round_count not in self.reasoning_history:
            self.reasoning_history[self.current_round_count] = {}

        # Snapshot profiles / stats at decision time so the oracle can audit
        # the exact context the LLM saw.
        profiles_snapshot = copy.deepcopy(getattr(self, "opponent_profiles", {}))
        stats_snapshot = copy.deepcopy(getattr(self, "opponent_stats", {}))

        seats = round_state.get("seats", [])
        stacks_snapshot = {s.get("name"): s.get("stack") for s in seats if s.get("name") is not None}

        node = {
            'reasoning': reasoning,
            'action': action_decision,
            'game_state': game_state.copy(),
            'prompt_components': copy.deepcopy(getattr(self, "last_prompt_components", None)),
            'beliefs': self.last_beliefs,
            'chosen_action_summary': self.last_chosen_action_summary,
            'opponent_profiles_snapshot': profiles_snapshot,
            'opponent_stats_snapshot': stats_snapshot,
        }
        node['game_state']['players_stacks'] = stacks_snapshot

        self.reasoning_history[self.current_round_count][street] = node

        model_name = self.get_model_config()['model'].split('/')[-1]
        alias = getattr(self.__class__, 'PLAYER_ALIAS', 'Unknown Player')
        print(f"\n[{model_name} ({alias})] Round {self.current_round_count} - {street.upper()}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Decision: {action_decision}")

        if street in self.actions_by_street:
            action_type = action_decision["action"].lower()
            if action_type in self.actions_by_street[street]:
                self.actions_by_street[street][action_type] += 1

        if action_decision["action"] == "RAISE":
            self.aggression_actions['raises'] += 1
        elif action_decision["action"] == "CALL":
            self.aggression_actions['calls'] += 1

        hs_label = (self.last_beliefs or {}).get("HandStrength", "").lower()

        if hs_label == "weak":
            category = "weak"
        elif hs_label == "medium":
            category = "medium"
        elif hs_label == "strong":
            category = "strong"
        else:
            category = "weak"

        if action_decision["action"].lower() in self.hand_strength_actions[category]:
            self.hand_strength_actions[category][action_decision["action"].lower()] += 1

        self.hand_stats['played'] += 1

        if action_decision["action"] == "FOLD":
            self.hand_stats['folded'] += 1
            return fold['action'], fold['amount']
        elif action_decision["action"] == "CALL":
            self.hand_stats['called'] += 1
            return call['action'], call['amount']
        elif action_decision["action"] == "RAISE":
            self.hand_stats['raised'] += 1

            hs_label = (self.last_beliefs or {}).get("HandStrength", "").lower()
            if hs_label == "strong":
                self.value_bets['made'] += 1
            else:
                self.bluffs['made'] += 1

            if 'amount' in action_decision:
                intended_amount = int(action_decision['amount'])
                actual_amount = max(min_raise, min(intended_amount, max_raise))
                return raise_action['action'], actual_amount
            else:
                return raise_action['action'], min_raise
        else:
            return fold['action'], fold['amount']

    def call_llm_api(self, game_state):
        config = self.get_model_config()
        if not config:
            print("No model config found. Defaulting to FOLD.")
            return {"action": "FOLD"}, "No model configuration"

        hole_cards = game_state.get("hole_cards", [])
        community_cards = game_state.get("community_cards", [])
        street = game_state.get("street", "unknown")
        pot_size = game_state.get("pot_size", 0)
        call_amount = game_state.get("call_amount", 0)
        min_raise = game_state.get("min_raise", 0)
        max_raise = game_state.get("max_raise", 0)
        pot_odds = float(game_state.get("pot_odds", 0.0))

        opponent_actions_text = self.format_opponent_actions_for_prompt(
            game_state.get("opponent_actions", {})
        )

        position_info = game_state.get("position_info", {})
        position_text = self.format_position_info_for_prompt(position_info)

        opponent_profiles_text = self.format_opponent_profiles_for_prompt()

        self.last_prompt_components = {}

        base_prompt = f"""
    You are a professional poker player, playing against other professional poker players, and you aim to make the most money in the long run.

    Current game state:
    - Your hole cards: {hole_cards}
    - Community cards: {community_cards}
    - Current street: {street}
    - Pot size: {pot_size}
    - Call amount: {call_amount}
    - Minimum raise: {min_raise}
    - Maximum raise: {max_raise}
    - Pot odds (0-1): {pot_odds:.2f}

    {position_text}

    Recent opponent actions this hand:
    {opponent_actions_text}

    Opponent profiling information (long-term tendencies):
    {opponent_profiles_text}

    - RiskTolerance: how willing this opponent is to take risks.
    - Aggressiveness: how often this opponent tends to bet/raise instead of check/call.
    - BluffFrequency: how often this opponent tends to bet/raise with weak or marginal hands.
    - CallingStationTendency: tendency to call under pressure rather than fold/raise.
    - ShowdownPropensity: tendency to continue to showdown rather than give up.

    You should USE this profiling information whenever it is available, especially when:
    - deciding whether a bet/raise is more likely to be a bluff or for value,
    - choosing between folding and calling,
    - deciding how thinly you can value bet.

    Important action rules:
    - Only consider FOLD when call_amount > 0.
    - Folding preflop too often will make you lose money in blinds in the long run.
    - If your hand is classified as STRONG, you should almost never FOLD.
    - Folding a STRONG hand is only reasonable in extreme cases.

    Action semantics:
    - CHECK is allowed only when call_amount == 0 (i.e., you can continue for free).
    - If call_amount > 0, you must choose CALL / RAISE / FOLD.

    You must first write a SELF-EXPLANATION of your thinking for this hand, then output a machine-readable DECISION.

    Use exactly the following SELF-EXPLANATION format:

    [SELF-EXPLANATION]
    NaturalLanguage: "<Use 1–2 short sentences ONLY. Explain your decision briefly without restating the game state.>"

    Beliefs:
    - HandStrength: {{weak / medium / strong}}
    - RiskAttitudeThisHand: {{conservative / neutral / aggressive}}
    - MainGoal: {{minimize_loss / take_small_edge / maximize_value / bluff}}
    - PerceivedOpponentRisk: {{low / medium / high}}
    - ProfileInfluence: "<Mention at most ONE opponent profile that influenced your decision.>"
    - IntendedReason: "<1 short phrase: pot_odds / board_texture / opponent_aggression / equity_edge / protection>"

    ChosenActionSummary:
    - IntendedActionType: {{fold / check / call / bet_small / bet_big}}
    - IntendedRiskLevel: {{low / medium / high}}
    [/SELF-EXPLANATION]

    Then output the final decision as a JSON object on a new line, prefixed by "DECISION:".

    DECISION:
    {{"action": "FOLD"}}
    or
    {{"action": "CALL"}}
    or
    {{"action": "CHECK"}}
    or
    {{"action": "RAISE", "amount": <integer_between_min_and_max_raise>}}

    Important:
    - The SELF-EXPLANATION must strictly follow the above template and use the given discrete value sets.
    - The JSON must be valid and parsable.
    - Only 4 actions: "FOLD", "CALL", "CHECK", "RAISE"
    - Only include "amount" field if action is "RAISE".
    - Do not put the JSON inside a code block (no ```).
    - Do not provide multiple versions of answers.
    - If you violate the format, your answer will be discarded.
    """

        extra_prompt = config.get("extra_prompt")
        if extra_prompt:
            base_prompt += f"\n{extra_prompt}\n"

        self.last_prompt_components["final_prompt"] = base_prompt

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY environment variable not set. Defaulting to FOLD.")
            return {"action": "FOLD"}, "Missing OPENROUTER_API_KEY"

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        use_complex_messages = config.get("use_complex_messages", True)
        if use_complex_messages:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": base_prompt}]
            }]
        else:
            messages = [{"role": "user", "content": base_prompt}]

        model_name = config.get("model", "unknown-model")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return {"action": "FOLD"}, f"[API-ERROR] {e}"

        reasoning, decision = self.parse_llm_response(llm_response)
        return decision, reasoning

    def get_formatted_opponent_actions(self, round_state):
        """Extract and format opponent action history from round_state."""
        formatted_actions = {}

        for seat in round_state.get('seats', []):
            uuid = seat.get('uuid')
            if uuid:
                self.seats_info[uuid] = {
                    'name': seat.get('name', f'Player_{uuid[:8]}'),
                    'stack': seat.get('stack', 0),
                    'state': seat.get('state', 'unknown')
                }
                self.player_names[uuid] = seat.get('name', f'Player_{uuid[:8]}')

        action_histories = round_state.get('action_histories', {})

        for street_name, actions in action_histories.items():
            if street_name not in formatted_actions:
                formatted_actions[street_name] = []

            for action in actions:
                uuid = action.get('uuid')
                if uuid and uuid != self.uuid:
                    player_name = self.player_names.get(uuid, f'Player_{uuid[:8]}')
                    action_type = action.get('action', 'UNKNOWN')
                    amount = action.get('amount', 0)

                    if action_type == 'FOLD':
                        action_desc = "FOLD"
                    elif action_type == 'CALL':
                        action_desc = f"CALL {amount}" if amount > 0 else "CALL"
                    elif action_type == 'RAISE':
                        action_desc = f"RAISE to {amount}"
                    elif action_type == 'SMALLBLIND':
                        action_desc = f"SMALL BLIND {amount}"
                    elif action_type == 'BIGBLIND':
                        action_desc = f"BIG BLIND {amount}"
                    else:
                        action_desc = f"{action_type} {amount}" if amount > 0 else action_type

                    formatted_actions[street_name].append({
                        'player': player_name,
                        'action': action_desc,
                        'stack': self.seats_info.get(uuid, {}).get('stack', 0)
                    })
                    self._update_opponent_stats(player_name, street_name, action_type, amount)

        return formatted_actions

    def format_opponent_actions_for_prompt(self, opponent_actions):
        """Format opponent action history as prompt text."""
        if not opponent_actions:
            return "Opponent Actions: No actions yet this round."

        actions_text = "Opponent Actions This Round:\n"

        for street, actions in opponent_actions.items():
            if actions:
                actions_text += f"- {street.capitalize()}:\n"
                for action_info in actions:
                    player = action_info['player']
                    action = action_info['action']
                    stack = action_info['stack']
                    actions_text += f"  * {player} (Stack: {stack}): {action}\n"

        if actions_text == "Opponent Actions This Round:\n":
            actions_text = "Opponent Actions: Only blinds posted so far."

        return actions_text

    def format_position_info_for_prompt(self, position_info):
        """
        Format position and table info as compact prompt text.
        """
        if not position_info:
            return "Table Info: Position information unavailable."

        total_players = position_info.get('total_players', 0)
        my_position = position_info.get('my_position', 'Unknown')
        my_stack = position_info.get('my_stack', 0)
        players_info = position_info.get('players_info', [])

        table_text = f"Table Info:\n"
        table_text += f"- Players: {total_players}, Your position: {my_position}, Your stack: {my_stack}\n"

        opponents = [p for p in players_info if not p.get('is_me', False)]
        if opponents:
            table_text += f"- Opponent stacks:"
            for player in opponents:
                table_text += f" {player['name']}({player['stack']})"

        return table_text

    def parse_llm_response(self, llm_response: str):
        """
        Parse the LLM response into (reasoning_text, decision_dict).

        Extracts the [SELF-EXPLANATION] block as reasoning, parses Beliefs
        and ChosenActionSummary into structured fields cached on
        self.last_beliefs / self.last_chosen_action_summary, and extracts
        the final action from the DECISION JSON.
        """
        self.last_beliefs = None
        self.last_chosen_action_summary = None

        reasoning = ""
        decision = {"action": "FOLD"}

        try:
            text = (llm_response or "").strip()

            se_match = re.search(
                r"\[SELF-EXPLANATION\](.*?)\[/SELF-EXPLANATION\]",
                text,
                re.DOTALL | re.IGNORECASE
            )
            se_block = None
            if se_match:
                se_block = se_match.group(0).strip()
                reasoning = se_block
            else:
                reasoning = text

            if se_block:
                beliefs = {}
                ca_summary = {}

                def pick(pattern, default=None, lower=True):
                    m = re.search(pattern, se_block, re.IGNORECASE)
                    if not m:
                        return default
                    v = m.group(1).strip()
                    return v.lower() if lower else v

                beliefs["HandStrength"] = pick(r"HandStrength:\s*([a-zA-Z_]+)")
                beliefs["RiskAttitudeThisHand"] = pick(r"RiskAttitudeThisHand:\s*([a-zA-Z_]+)")
                beliefs["MainGoal"] = pick(r"MainGoal:\s*([a-zA-Z_]+)")
                beliefs["PerceivedOpponentRisk"] = pick(r"PerceivedOpponentRisk:\s*([a-zA-Z_]+)")

                ca_summary["IntendedActionType"] = pick(r"IntendedActionType:\s*([a-zA-Z_]+)")
                ca_summary["IntendedRiskLevel"] = pick(r"IntendedRiskLevel:\s*([a-zA-Z_]+)")

                beliefs = {k: v for k, v in beliefs.items() if v is not None}
                ca_summary = {k: v for k, v in ca_summary.items() if v is not None}

                if beliefs:
                    self.last_beliefs = beliefs
                if ca_summary:
                    self.last_chosen_action_summary = ca_summary

            # Prefer the {...} immediately after "DECISION:". Greedy regex may
            # overshoot, so trim back to the outermost braces.
            decision_json = None
            m_dec = re.search(r"DECISION\s*:\s*(\{.*\})\s*$", text, re.IGNORECASE | re.DOTALL)
            if m_dec:
                tail = m_dec.group(1).strip()
                l = tail.find("{")
                r = tail.rfind("}")
                if l != -1 and r != -1 and r > l:
                    decision_json = tail[l:r + 1]

            # Fallback: the last JSON object in the whole response that
            # contains an "action" field.
            if not decision_json:
                json_candidates = re.findall(r"\{[\s\S]*?\}", text)
                for cand in reversed(json_candidates):
                    if re.search(r"\"action\"\s*:", cand, re.IGNORECASE):
                        decision_json = cand
                        break

            if not decision_json:
                raise ValueError("No valid decision JSON found in LLM response.")

            parsed = json.loads(decision_json)

            if not isinstance(parsed, dict) or "action" not in parsed:
                raise ValueError("Decision JSON missing required 'action' field.")

            parsed["action"] = str(parsed.get("action", "FOLD")).upper().strip()

            decision = parsed

        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            self.parse_failed += 1
            reasoning = f"[PARSE-ERROR]\nRawResponse:\n{llm_response}"
            decision = {"action": "FOLD"}

        return reasoning, decision

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

        for player in game_info['seats']:
            if player['uuid'] == self.uuid:
                self.initial_stack = player['stack']
                self.current_stack = player['stack']
                break

        self.total_gains_losses = 0
        self.round_gains_losses = {}
        self.reasoning_history = {}

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.current_round_count = round_count

        self.current_round_actions = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'river': []
        }

        for seat in seats:
            if seat['uuid'] == self.uuid:
                stack_before_round = self.current_stack
                self.current_stack = seat['stack']

                if round_count > 1:
                    previous_round = round_count - 1
                    round_result = self.current_stack - stack_before_round
                    self.round_gains_losses[previous_round] = round_result
                    self.total_gains_losses = self.current_stack - self.initial_stack
                break
        pass

    def receive_street_start_message(self, street, round_state):
        self.current_street = street
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                self.current_stack = seat['stack']
                break
        pass

    def receive_game_update_message(self, action, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                self.current_stack = seat['stack']
                break
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        round_count = round_state['round_count']

        for winner in winners:
            if winner['uuid'] == self.uuid:
                self.hand_stats['won'] += 1
                break

        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                new_stack = seat['stack']
                round_result = new_stack - self.current_stack
                self.current_stack = new_stack
                self.round_gains_losses[round_count] = round_result
                self.total_gains_losses = self.current_stack - self.initial_stack

                if self.current_street:
                    self.actions_by_street[self.current_street]['profit'] += round_result

                went_to_showdown = len(hand_info) > 0
                if went_to_showdown:
                    hs_label = (self.last_beliefs or {}).get("HandStrength", "").lower()
                    is_value_bet = (hs_label == "strong")

                    if is_value_bet:
                        if round_result > 0:
                            self.value_bets['successful'] += 1
                        self.value_bets['profit'] += round_result
                    else:
                        if round_result > 0:
                            self.bluffs['successful'] += 1
                        self.bluffs['profit'] += round_result

        went_to_showdown = len(hand_info) > 0
        for seat in round_state['seats']:
            pname = seat.get('name')
            if not pname or seat['uuid'] == self.uuid:
                continue
            stats = self.opponent_stats.setdefault(pname, {
                "hands_seen": 0,
                "voluntary_put_money": 0,
                "preflop_raises": 0,
                "postflop_raises": 0,
                "postflop_calls": 0,
                "showdowns": 0,
                "seen_cards_weak_aggressive": 0,
            })
            stats["hands_seen"] += 1
            if went_to_showdown:
                stats["showdowns"] += 1

        # Trigger second-person profile update
        self.update_opponent_profiles_after_hand(round_state)
        # Identify opponents who busted out this hand and record them
        for seat in round_state['seats']:
            uuid = seat.get('uuid')
            if uuid == self.uuid:
                continue  # skip self

            state = seat.get('state')
            stack = seat.get('stack', 0)

            # Heuristic: stack <= 0 and no longer 'participating' / 'allin' => busted
            if stack <= 0 and state not in ('participating', 'allin'):
                if not hasattr(self, "busted_opponents"):
                    self.busted_opponents = set()
                self.busted_opponents.add(uuid)
        pass


    def get_performance_stats(self):
        """Return comprehensive performance statistics."""
        hands_played = self.hand_stats['played']
        win_rate = (self.hand_stats['won'] / hands_played) * 100 if hands_played > 0 else 0
        fold_rate = (self.hand_stats['folded'] / hands_played) * 100 if hands_played > 0 else 0
        call_rate = (self.hand_stats['called'] / hands_played) * 100 if hands_played > 0 else 0
        raise_rate = (self.hand_stats['raised'] / hands_played) * 100 if hands_played > 0 else 0

        af_calls = self.aggression_actions['calls'] or 1
        aggression_factor = self.aggression_actions['raises'] / af_calls

        value_bet_success_rate = (self.value_bets['successful'] / self.value_bets['made']) * 100 if self.value_bets['made'] > 0 else 0
        bluff_success_rate = (self.bluffs['successful'] / self.bluffs['made']) * 100 if self.bluffs['made'] > 0 else 0

        return {
            'initial_stack': self.initial_stack,
            'current_stack': self.current_stack,
            'total_profit_loss': self.total_gains_losses,
            'profit_percentage': (self.total_gains_losses / self.initial_stack) * 100 if self.initial_stack > 0 else 0,
            'round_results': self.round_gains_losses,
            'hands_played': hands_played,
            'hands_won': self.hand_stats['won'],
            'hands_folded': self.hand_stats['folded'],
            'hands_called': self.hand_stats['called'],
            'hands_raised': self.hand_stats['raised'],
            'win_rate': win_rate,
            'fold_rate': fold_rate,
            'call_rate': call_rate,
            'raise_rate': raise_rate,
            'aggression_factor': aggression_factor,
            'actions_by_street': self.actions_by_street,
            'value_betting': {
                'attempts': self.value_bets['made'],
                'successful': self.value_bets['successful'],
                'success_rate': value_bet_success_rate,
                'profit': self.value_bets['profit']
            },
            'bluffing': {
                'attempts': self.bluffs['made'],
                'successful': self.bluffs['successful'],
                'success_rate': bluff_success_rate,
                'profit': self.bluffs['profit']
            },
            'hand_strength_decisions': self.hand_strength_actions,
            'failed_parses': self.parse_failed,
            'reasoning_history': self.reasoning_history,
            'opponent_profiles': self.opponent_profiles,
            'opponent_profile_history': self.opponent_profile_history,
        }

    def get_reasoning_summary(self):
        """Return a summary of the reasoning history."""
        model_name = self.get_model_config()['model'].split('/')[-1]
        summary = f"\n {model_name} Reasoning Summary:\n"
        summary += "=" * 50 + "\n"

        for round_num, round_data in self.reasoning_history.items():
            summary += f"\n Round {round_num}:\n"
            for street, street_data in round_data.items():
                summary += f"   {street.capitalize()}:\n"
                summary += f"     {street_data['reasoning']}\n"
                summary += f"     {street_data['action']}\n"

        return summary

    def get_position_info(self, round_state):
        """Return position, player count, seating order, and stack info."""
        seats = round_state.get('seats', [])
        total_players = len([seat for seat in seats if seat.get('state') == 'participating'])
        dealer_btn = round_state.get('dealer_btn', 0)

        my_position = None
        my_stack = 0
        players_by_position = []

        for i, seat in enumerate(seats):
            if seat.get('state') == 'participating':
                player_info = {
                    'name': seat.get('name', f'Player_{seat.get("uuid", "")[:8]}'),
                    'stack': seat.get('stack', 0),
                    'position': i,
                    'is_dealer': i == dealer_btn,
                    'is_me': seat.get('uuid') == self.uuid
                }

                if player_info['is_me']:
                    my_position = i
                    my_stack = player_info['stack']

                players_by_position.append(player_info)

        position_desc = "Unknown"
        if my_position is not None and total_players > 1:
            if total_players == 2:
                position_desc = "Dealer/SB" if my_position == dealer_btn else "BB"
            else:
                positions_from_dealer = (my_position - dealer_btn) % total_players
                if positions_from_dealer == 0:
                    position_desc = "Dealer"
                elif positions_from_dealer == 1:
                    position_desc = "Small Blind"
                elif positions_from_dealer == 2:
                    position_desc = "Big Blind"
                elif positions_from_dealer <= total_players // 2:
                    position_desc = "Early Position"
                elif positions_from_dealer <= 3 * total_players // 4:
                    position_desc = "Middle Position"
                else:
                    position_desc = "Late Position"

        return {
            'total_players': total_players,
            'my_position': position_desc,
            'my_stack': my_stack,
            'players_info': players_by_position
        }

    def _update_opponent_stats(self, opponent_name, street, action_type, amount):
        """Accumulate per-opponent behavioral counters for profile estimation."""
        stats = self.opponent_stats.setdefault(opponent_name, {
            "hands_seen": 0,
            "voluntary_put_money": 0,
            "preflop_raises": 0,
            "postflop_raises": 0,
            "postflop_calls": 0,
            "showdowns": 0,
            "seen_cards_weak_aggressive": 0,
        })

        if street == 'preflop' and action_type in ('CALL', 'RAISE'):
            stats["voluntary_put_money"] += 1

        if street == 'preflop' and action_type == 'RAISE':
            stats["preflop_raises"] += 1

        if street in ('flop', 'turn', 'river'):
            if action_type == 'RAISE':
                stats["postflop_raises"] += 1
            elif action_type == 'CALL':
                stats["postflop_calls"] += 1

    def update_opponent_profiles_after_hand(self, round_state):
        """Parallelized opponent-profile update at the end of each hand.

        Gathers opponents, fires parallel LLM profiling calls, then writes back
        smoothed trait updates on the main thread.
        """
        start_time = time.time()

        # Rule-based agents have no LLM backend; skip profiling for them.
        config = self.get_model_config() or {}
        model_name = config.get("model", "")
        if model_name == "monte-carlo-algorithm":
            return
        if model_name == "LoosePassive":
            return
        if model_name == "LooseAggressive":
            return
        if model_name == "Maniac":
            return
        if model_name == "TightPassive":
            return
        if model_name == "TightAggressive":
            return

        round_count = round_state.get("round_count", None)

        # Throttle to one update every N hands to avoid API overload.
        every_n = getattr(self, "profile_update_every_n_hands", 1)
        if every_n and round_count is not None and (round_count % every_n != 0):
            return

        opponents_to_update = []
        for seat in round_state.get("seats", []):
            uuid = seat.get("uuid")
            if not uuid or uuid == self.uuid:
                continue

            if uuid in getattr(self, "busted_opponents", set()):
                continue

            name = seat.get("name", f"Player_{uuid[:8]}")

            op_stats = self.opponent_stats.get(name)
            if not op_stats:
                continue

            summary_text = self.build_opponent_summary_text(name, op_stats)

            opponents_to_update.append({
                "uuid": uuid,
                "name": name,
                "summary_text": summary_text
            })

        if not opponents_to_update:
            return

        max_workers = min(len(opponents_to_update), getattr(self, "profile_max_workers", 8))
        timeout_sec = getattr(self, "profile_llm_timeout_sec", 12)

        new_profiles = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(self.call_opponent_profile_llm, opp["name"], opp["summary_text"]): opp["name"]
                for opp in opponents_to_update
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    prof = future.result(timeout=timeout_sec)
                    if prof is not None:
                        new_profiles[name] = prof
                except Exception as e:
                    print(f"[WARNING]({name}): {e}")

        for name, new_profile in new_profiles.items():
            old_profile = self.opponent_profiles.get(name, {})
            old_traits = old_profile.get("Traits", {})
            llm_traits = new_profile.get("Traits", {})

            updated_traits = {}
            for k in ["RiskTolerance", "Aggressiveness", "BluffFrequency",
                      "CallingStationTendency", "ShowdownPropensity"]:
                old_v = old_traits.get(k, 0.5)
                llm_v = llm_traits.get(k, old_v)
                updated_traits[k] = round(self.step_update(old=old_v, target=llm_v, step=0.05), 2)

            smoothed_profile = {
                "Traits": updated_traits,
                "QualitativeSummary": new_profile.get("QualitativeSummary", ""),
                "UpdateRationale": new_profile.get("UpdateRationale", "")
            }
            self.opponent_profiles[name] = smoothed_profile

            history = self.opponent_profile_history.setdefault(name, [])
            history.append({"round": round_count, "profile": smoothed_profile})


    def build_opponent_summary_text(self, opponent_name, stats):
        """Render opponent stats as the Evidence -> Belief bridge for the LLM."""
        hands_seen = stats.get("hands_seen", 0)
        vpip = stats["voluntary_put_money"] / hands_seen if hands_seen > 0 else 0.0
        pfr = stats["preflop_raises"] / hands_seen if hands_seen > 0 else 0.0

        postflop_agg_denom = stats["postflop_calls"] or 1
        postflop_agg = stats["postflop_raises"] / postflop_agg_denom

        summary = f"""
We are modelling opponent: {opponent_name}.

Observed statistics so far:
- Hands seen: {hands_seen}
- VPIP (voluntarily put money preflop): {vpip:.2f}
- PFR (preflop raise frequency): {pfr:.2f}
- Postflop aggression (raises over calls): {postflop_agg:.2f}
- Showdowns reached: {stats.get("showdowns", 0)}

Use these statistics to infer this opponent's risk tolerance, aggressiveness, and bluff frequency.
"""
        return summary.strip()

    def call_opponent_profile_llm(self, opponent_name, summary_text):
        """Call the LLM with the [OPPONENT-PROFILE] template; return a dict of Traits / QualitativeSummary / UpdateRationale, or None."""
        config = self.get_model_config()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("No OPENROUTER_API_KEY for opponent profiling, skip.")
            return None
        model_name = config.get('model', 'unknown-model')

        # Rule-based / monte agents have no OpenRouter backend to query.
        if model_name=="monte-carlo-algorithm":
            print(f"Model '{model_name}' seems not to be an OpenRouter LLM, skip opponent profiling.")
            return None

        if model_name=="LoosePassive":
            print(f"Model '{model_name}' seems not to be an OpenRouter LLM, skip opponent profiling.")
            return None

        if model_name == "LooseAggressive":
            print(f"Model '{model_name}' seems not to be an OpenRouter LLM, skip opponent profiling.")
            return None
        if model_name == "Maniac":
            print(f"Model '{model_name}' seems not to be an OpenRouter LLM, skip opponent profiling.")
            return None
        if model_name == "TightPassive":
            print(f"Model '{model_name}' seems not to be an OpenRouter LLM, skip opponent profiling.")
            return None
        if model_name == "TightAggressive":
            print(f"Model '{model_name}' seems not to be an OpenRouter LLM, skip opponent profiling.")
            return None



        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        profile_prompt = f"""
You are analysing a poker opponent based on observed long-term statistics.

{summary_text}

Please output an updated opponent profile in the following format:

[OPPONENT-PROFILE]
OpponentID: <string>

Traits:
- RiskTolerance: <0.0 - 1.0>  # estimated willingness to take risk
- Aggressiveness: <0.0 - 1.0> # estimated tendency to bet/raise rather than check/call
- BluffFrequency: <0.0 - 1.0> # estimated tendency to bet/raise with weak or marginal hands
- CallingStationTendency: <0.0 - 1.0> # tendency to call under pressure rather than fold/raise
- ShowdownPropensity: <0.0 - 1.0>     # tendency to continue to showdown rather than give up

QualitativeSummary: "<1 sentence describing this opponent's style>"

UpdateRationale: "<1 sentence explaining why you updated the profile key update direction based on the statistics>"
[/OPPONENT-PROFILE]

Hint:
- CallingStationTendency should correlate with high postflop CALL relative to RAISE.
- ShowdownPropensity should correlate with high showdowns reached relative to hands seen.
Important:
- Your numeric estimates are treated as directional signals.
- Focus on the correct direction and relative magnitude, not precision.
- Traits should be coarse estimates.
- If the evidence is weak or ambiguous, stay close to current tendencies.
- The system will only move traits by ±0.05 per hand; focus on direction.


"""

        messages = [{"role": "user", "content": profile_prompt}]
        model_name = config.get('model', 'unknown-model')

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2
            )
            content = resp.choices[0].message.content
            return self.parse_opponent_profile_response(content)
        except Exception as e:
            print(f"Error calling opponent profile LLM: {e}")
            return None

    def parse_opponent_profile_response(self, content: str):
        """Parse the [OPPONENT-PROFILE] block out of the LLM text."""
        try:
            risk = re.search(r"RiskTolerance:\s*([0-9.]+)", content)
            agg = re.search(r"Aggressiveness:\s*([0-9.]+)", content)
            bluff = re.search(r"BluffFrequency:\s*([0-9.]+)", content)
            cs = re.search(r"CallingStationTendency:\s*([0-9.]+)", content)
            sd = re.search(r"ShowdownPropensity:\s*([0-9.]+)", content)

            def safe_float(m, default=0.5):
                try:
                    return float(m.group(1)) if m else default
                except:
                    return default

            risk_v = safe_float(risk)
            agg_v = safe_float(agg)
            bluff_v = safe_float(bluff)
            cs_v = safe_float(cs)
            sd_v = safe_float(sd)

            qsum_match = re.search(r"QualitativeSummary:\s*\"(.*?)\"", content, re.DOTALL)
            rationale_match = re.search(r"UpdateRationale:\s*\"(.*?)\"", content, re.DOTALL)

            qsum = qsum_match.group(1).strip() if qsum_match else ""
            rationale = rationale_match.group(1).strip() if rationale_match else ""

            return {
                "Traits": {
                    "RiskTolerance": risk_v,
                    "Aggressiveness": agg_v,
                    "BluffFrequency": bluff_v,
                    "CallingStationTendency": cs_v,
                    "ShowdownPropensity": sd_v
                },
                "QualitativeSummary": qsum,
                "UpdateRationale": rationale
            }
        except Exception as e:
            print(f"Failed to parse opponent profile: {e}")
            return None

    def format_opponent_profiles_for_prompt(self):
        """Format self.opponent_profiles as compact text for the 1st-person prompt."""
        if not self.opponent_profiles:
            return "Opponent Profiles: No stable profiles yet."

        lines = ["Opponent Profiles (estimated):"]
        for name, prof in self.opponent_profiles.items():
            traits = prof.get("Traits", {})
            rt = traits.get("RiskTolerance", 0.5)
            ag = traits.get("Aggressiveness", 0.5)
            bf = traits.get("BluffFrequency", 0.5)
            cs = traits.get("CallingStationTendency", 0.5)
            sd = traits.get("ShowdownPropensity", 0.5)
            qsum = prof.get("QualitativeSummary", "")

            lines.append(f"- {name}:")
            lines.append(
                f"  * RiskTolerance={rt:.2f}, Aggressiveness={ag:.2f}, BluffFrequency={bf:.2f}, "
                f"CallingStationTendency={cs:.2f}, ShowdownPropensity={sd:.2f}"
            )
            if qsum:
                lines.append(f"  * Summary: {qsum}")

        return "\n".join(lines)

    def step_update(self, old, target, step=0.05):
        """Move `old` toward `target` by at most `step`."""
        if target > old:
            return min(old + step, 1.0)
        elif target < old:
            return max(old - step, 0.0)
        else:
            return old

    def _normalize_llm_decision(self, decision: dict, call_amount: int, min_raise: int, max_raise: int):
        """Normalize the LLM's decision to an engine-legal action dict.

        CHECK becomes CALL when call_amount == 0; otherwise it maps to FOLD.
        RAISE amounts are clipped into [min_raise, max_raise].
        """
        if not isinstance(decision, dict):
            return {"action": "FOLD"}, "Decision not dict -> FOLD"

        act = str(decision.get("action", "FOLD")).upper().strip()

        if act == "CHECK":
            if call_amount == 0:
                return {"action": "CALL"}, "CHECK mapped to CALL"
            return {"action": "FOLD"}, "CHECK invalid (call_amount>0) -> FOLD"

        if act == "CALL":
            return {"action": "CALL"}, "CALL normalized (no amount)"

        if act == "FOLD":
            return {"action": "FOLD"}, "FOLD normalized (no amount)"

        if act == "RAISE":
            if max_raise < min_raise or max_raise <= 0:
                if call_amount == 0:
                    return {"action": "CALL"}, "RAISE not allowed -> CALL"
                return {"action": "CALL"}, "RAISE not allowed -> CALL"

            intended = decision.get("amount", min_raise)
            try:
                intended = int(intended)
            except Exception:
                intended = min_raise

            actual = max(min_raise, min(intended, max_raise))
            return {"action": "RAISE", "amount": actual}, f"RAISE clipped to {actual}"

        if call_amount == 0:
            return {"action": "CALL"}, f"Unknown '{act}' -> CALL"
        return {"action": "FOLD"}, f"Unknown '{act}' -> FOLD"

    def _to_treys_cards(self, cards):
        # Engine supplies "DA"/"H2" (suit first); treys wants "Ad"/"2h".
        rank_map = {"A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T",
                    "9": "9", "8": "8", "7": "7", "6": "6", "5": "5",
                    "4": "4", "3": "3", "2": "2"}
        suit_map = {"S": "s", "H": "h", "D": "d", "C": "c"}

        out = []
        for c in cards or []:
            s = str(c).strip()
            if len(s) != 2:
                continue
            suit, rank = s[0].upper(), s[1].upper()
            if suit not in suit_map or rank not in rank_map:
                continue
            out.append(TCard.new(rank_map[rank] + suit_map[suit]))
        return out

    def _estimate_equity_treys(self, hole_card, community_card, n_opponents=1, n_sim=800):
        """Estimate hero equity in [0,1] by Monte Carlo over unknown cards."""
        hero = self._to_treys_cards(hole_card)
        board = self._to_treys_cards(community_card)

        if len(hero) != 2:
            return 0.5

        evaluator = TEvaluator()
        wins = 0
        ties = 0

        for _ in range(n_sim):
            deck = TDeck()

            known = hero + board
            for kc in known:
                if kc in deck.cards:
                    deck.cards.remove(kc)

            opp_hands = []
            for _ in range(max(1, int(n_opponents))):
                cards2 = deck.draw(2)
                opp_hands.append([cards2[0], cards2[1]])

            need = 5 - len(board)
            sim_board = board + (deck.draw(need) if need > 0 else [])

            hero_score = evaluator.evaluate(sim_board, hero)
            opp_scores = [evaluator.evaluate(sim_board, oh) for oh in opp_hands]
            best_opp = min(opp_scores)  # treys: lower score = stronger hand

            if hero_score < best_opp:
                wins += 1
            elif hero_score == best_opp:
                ties += 1

        return (wins + 0.5 * ties) / float(n_sim)

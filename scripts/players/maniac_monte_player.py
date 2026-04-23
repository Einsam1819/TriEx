from .llm_base_player import LLMBasePlayer
import random

# Per-street cap on a single raise. Keeps Maniac sizing within realistic bounds.
STREET_RAISE_CAP = {
    "preflop": 40,
    "flop": 60,
    "turn": 80,
    "river": 120
}

MULTIPLIERS = {
    "strong_base": 0.60,
    "strong_big": 0.90,
    "medium": 0.50,
    "weak_stab": 0.40,
    "weak_bluff": 0.55
}


class ManiacMontePlayer(LLMBasePlayer):
    """Ground-truth Maniac: very high VPIP/PFR, high bluff frequency, raise sizes capped."""
    PLAYER_ALIAS = "Noah Blake"

    def get_model_config(self):
        return {
            "model": "Maniac",
            "use_complex_messages": False,
            "enable_opponent_profiles": False
        }

    def _street_raise_cap(self, street: str) -> int:
        return int(STREET_RAISE_CAP.get(street, 80))

    def _clamp_raise_amount(self, intended: float, min_raise: int, max_raise: int, street: str):
        """Return a legal raise amount within [min_raise, min(max_raise, STREET_RAISE_CAP[street])], or None."""
        hard_cap = int(max_raise)
        hard_cap = min(hard_cap, self._street_raise_cap(street))

        if hard_cap < int(min_raise):
            return None

        amt = int(intended)
        amt = max(int(min_raise), min(amt, hard_cap))
        return amt

    def call_llm_api(self, game_state):
        pot_size = int(game_state.get("pot_size", 0))
        call_amount = int(game_state.get("call_amount", 0))
        min_raise = int(game_state.get("min_raise", 0))
        max_raise = int(game_state.get("max_raise", 0))
        street = str(game_state.get("street", "preflop"))

        win_rate = float(game_state.get("env_hand_strength", 0.5))

        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
        can_check = (call_amount == 0)
        r = random.random()

        pot_base = pot_size if pot_size > 0 else min_raise

        if win_rate >= 0.70:
            intended = MULTIPLIERS["strong_base"] * pot_base
            if r < 0.5:
                intended = MULTIPLIERS["strong_big"] * pot_base

            amt = self._clamp_raise_amount(intended, min_raise, max_raise, street)
            if amt is None:
                if can_check:
                    return {"action": "CALL", "amount": 0}, f"MANIAC: strong ({win_rate:.2%}) but capped; check."
                return {"action": "CALL", "amount": call_amount}, f"MANIAC: strong ({win_rate:.2%}) but capped; call."
            return {"action": "RAISE", "amount": amt}, f"MANIAC: strong ({win_rate:.2%}), raise {amt}."

        if win_rate >= 0.40:
            if r < 0.85 and min_raise <= max_raise:
                intended = MULTIPLIERS["medium"] * pot_base
                amt = self._clamp_raise_amount(intended, min_raise, max_raise, street)
                if amt is not None:
                    return {"action": "RAISE", "amount": amt}, f"MANIAC: pressure raise ({win_rate:.2%}) to {amt}."

            if can_check:
                return {"action": "CALL", "amount": 0}, f"MANIAC: medium ({win_rate:.2%}), check."
            return {"action": "CALL", "amount": call_amount}, f"MANIAC: medium ({win_rate:.2%}), call."

        if can_check:
            if r < 0.75 and min_raise <= max_raise:
                intended = MULTIPLIERS["weak_stab"] * pot_base
                amt = self._clamp_raise_amount(intended, min_raise, max_raise, street)
                if amt is not None:
                    return {"action": "RAISE", "amount": amt}, f"MANIAC: stab bluff on a free street, raise {amt}."
            return {"action": "CALL", "amount": 0}, f"MANIAC: weak ({win_rate:.2%}), check."

        # Facing a bet: bluff-raise or gamble call.
        if r < 0.55 and min_raise <= max_raise:
            intended = MULTIPLIERS["weak_bluff"] * pot_base
            amt = self._clamp_raise_amount(intended, min_raise, max_raise, street)
            if amt is not None:
                return {"action": "RAISE", "amount": amt}, f"MANIAC: bluff raise ({win_rate:.2%}) to {amt}."

        if pot_odds >= 0.15 and r < 0.60:
            return {"action": "CALL", "amount": call_amount}, f"MANIAC: gamble call (pot_odds={pot_odds:.2f})."

        return {"action": "FOLD", "amount": 0}, f"MANIAC: fold ({win_rate:.2%})."

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)
        self.nb_player = game_info["player_num"]
from .llm_base_player import LLMBasePlayer
import random

# Bet sizes as a fraction of pot, keyed by situation. Tune here only.
MULT = {
    "value_strong_pot": 0.80,   # win_rate >= 0.55: value bet or raise
    "semi_bluff_pot": 0.60,     # win_rate >= 0.35: semi-bluff or pressure
    "free_stab_pot": 0.75,      # weak hand with free check option
    "bluff_pot": 0.70,          # bluff raise with decent pot odds
    "pure_bluff_pot": 0.80      # pure bluff with poor pot odds
}


class LooseAggressiveMontePlayer(LLMBasePlayer):
    """Ground-truth Loose-Aggressive scripted player.

    Relatively high VPIP and PFR, high aggression factor, occasional bluffs
    with weak hands. win_rate comes from the parent's env_hand_strength.
    """
    PLAYER_ALIAS = "Ava Park"

    def get_model_config(self):
        return {
            "model": "LooseAggressive",
            "use_complex_messages": False,
            "enable_opponent_profiles": False
        }

    def _clamp_raise_amount(self, intended: float, min_raise: int, max_raise: int):
        """Clamp `intended` into the engine's legal raise range; return None if illegal."""
        min_raise = int(min_raise)
        max_raise = int(max_raise)

        if max_raise < min_raise:
            return None

        amt = int(intended)
        if amt < min_raise:
            amt = min_raise
        if amt > max_raise:
            amt = max_raise
        return amt

    def call_llm_api(self, game_state):
        pot_size = int(game_state["pot_size"])
        street = str(game_state["street"])
        call_amount = int(game_state["call_amount"])
        min_raise = int(game_state["min_raise"])
        max_raise = int(game_state["max_raise"])

        win_rate = float(game_state.get("env_hand_strength", 0.5))

        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
        can_check = (call_amount == 0)
        r = random.random()

        # Fall back to min_raise when pot_size is 0 (preflop, blinds only).
        pot_base = pot_size if pot_size > 0 else min_raise

        # Very strong: prefer max raise.
        if win_rate >= 0.75:
            amt = self._clamp_raise_amount(max_raise, min_raise, max_raise)
            if amt is None:
                if can_check:
                    return {"action": "CALL", "amount": 0}, f"LAG: strong ({win_rate:.2%}) but cannot raise; check."
                return {"action": "CALL", "amount": call_amount}, f"LAG: strong ({win_rate:.2%}) but cannot raise; call."
            return {"action": "RAISE", "amount": amt}, (
                f"LAG style: very strong hand ({win_rate:.2%}). "
                f"Aggressive raise, clamped to {amt}."
            )

        # Medium-strong: value bet / raise.
        if win_rate >= 0.55:
            intended = MULT["value_strong_pot"] * pot_base
            amt = self._clamp_raise_amount(intended, min_raise, max_raise)
            if amt is None:
                if can_check:
                    return {"action": "CALL", "amount": 0}, f"LAG: good ({win_rate:.2%}) but cannot raise; check."
                return {"action": "CALL", "amount": call_amount}, f"LAG: good ({win_rate:.2%}) but cannot raise; call."
            return {"action": "RAISE", "amount": amt}, (
                f"Solid hand ({win_rate:.2%}). "
                f"As LAG, raise for value and pressure, clamped to {amt}."
            )

        # Medium: usually raise, sometimes call to control pot.
        if win_rate >= 0.35:
            if r < 0.7 and min_raise <= max_raise:
                intended = MULT["semi_bluff_pot"] * pot_base
                amt = self._clamp_raise_amount(intended, min_raise, max_raise)
                if amt is not None:
                    return {"action": "RAISE", "amount": amt}, (
                        f"Medium hand ({win_rate:.2%}). "
                        f"LAG pressure raise, clamped to {amt}."
                    )
            if can_check:
                return {"action": "CALL", "amount": 0}, f"Medium hand ({win_rate:.2%}), check."
            return {"action": "CALL", "amount": call_amount}, f"Medium hand ({win_rate:.2%}), call to control pot."

        # Weak: probe when free, otherwise rely on pot odds with occasional bluff.
        if can_check:
            if r < 0.7 and min_raise <= max_raise:
                intended = MULT["free_stab_pot"] * pot_base
                amt = self._clamp_raise_amount(intended, min_raise, max_raise)
                if amt is not None:
                    return {"action": "RAISE", "amount": amt}, (
                        f"Weak hand ({win_rate:.2%}) on a free street. "
                        f"LAG probe bet, clamped to {amt}."
                    )
            return {"action": "CALL", "amount": 0}, f"Weak hand ({win_rate:.2%}), take free check."

        # Facing a bet.
        if pot_odds >= 0.25:
            if r < 0.25 and min_raise <= max_raise:
                intended = MULT["bluff_pot"] * pot_base
                amt = self._clamp_raise_amount(intended, min_raise, max_raise)
                if amt is not None:
                    return {"action": "RAISE", "amount": amt}, (
                        f"Weak hand ({win_rate:.2%}) but decent pot odds ({pot_odds:.2%}). "
                        f"Occasional bluff raise, clamped to {amt}."
                    )
            return {"action": "CALL", "amount": call_amount}, (
                f"Weak hand ({win_rate:.2%}) but decent pot odds ({pot_odds:.2%}). Call to realize equity."
            )

        # Poor pot odds: usually fold, rare pure bluff.
        if r < 0.15 and min_raise <= max_raise:
            intended = MULT["pure_bluff_pot"] * pot_base
            amt = self._clamp_raise_amount(intended, min_raise, max_raise)
            if amt is not None:
                return {"action": "RAISE", "amount": amt}, (
                    f"Very weak hand ({win_rate:.2%}) and poor pot odds ({pot_odds:.2%}). "
                    f"Rare pure bluff raise, clamped to {amt}."
                )

        return {"action": "FOLD", "amount": 0}, (
            f"Very weak hand ({win_rate:.2%}) with poor pot odds ({pot_odds:.2%}). Fold."
        )

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)
        self.nb_player = game_info["player_num"]
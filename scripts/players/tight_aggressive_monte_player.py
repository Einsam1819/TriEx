from .llm_base_player import LLMBasePlayer
import random

MULT = {
    "strong": 0.70,
    "good": 0.55,
    "rare_bluff": 0.40
}


class TightAggressiveMontePlayer(LLMBasePlayer):
    """Ground-truth Tight-Aggressive (TAG): low VPIP, raise-first, low bluff."""
    PLAYER_ALIAS = "Jade Park"

    def get_model_config(self):
        return {
            "model": "TightAggressive",
            "use_complex_messages": False,
            "enable_opponent_profiles": False
        }

    def _clamp_raise_amount(self, intended: float, min_raise: int, max_raise: int):
        """Clamp `intended` into [min_raise, max_raise]; return None if illegal."""
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

        if win_rate >= 0.72:
            intended = MULT["strong"] * pot_base
            amt = self._clamp_raise_amount(intended, min_raise, max_raise)

            if amt is None:
                if can_check:
                    return {"action": "CALL", "amount": 0}, f"TAG: strong ({win_rate:.2%}) but cannot raise; check."
                return {"action": "CALL", "amount": call_amount}, f"TAG: strong ({win_rate:.2%}) but cannot raise; call."

            return {"action": "RAISE", "amount": amt}, f"TAG: strong hand ({win_rate:.2%}), value raise to {amt}."

        if win_rate >= 0.55:
            intended = MULT["good"] * pot_base
            amt = self._clamp_raise_amount(intended, min_raise, max_raise)

            if can_check:
                if amt is None:
                    return {"action": "CALL", "amount": 0}, f"TAG: good ({win_rate:.2%}) but cannot raise; check."
                return {"action": "RAISE", "amount": amt}, f"TAG: good hand ({win_rate:.2%}), bet for value to {amt}."

            # Facing a bet: usually raise, sometimes call to control pot.
            if r < 0.75 and amt is not None:
                return {"action": "RAISE", "amount": amt}, f"TAG: good hand ({win_rate:.2%}), prefer raise to {amt}."
            return {"action": "CALL", "amount": call_amount}, f"TAG: good hand ({win_rate:.2%}), call to control pot."

        if win_rate >= 0.40:
            if can_check:
                return {"action": "CALL", "amount": 0}, f"TAG: marginal ({win_rate:.2%}), take free check."
            if pot_odds >= 0.25 and r < 0.60:
                return {"action": "CALL", "amount": call_amount}, f"TAG: marginal ({win_rate:.2%}), call with pot odds."
            return {"action": "FOLD", "amount": 0}, f"TAG: marginal ({win_rate:.2%}), fold facing cost."

        if can_check:
            return {"action": "CALL", "amount": 0}, f"TAG: weak ({win_rate:.2%}), check."
        else:
            if r < 0.05 and min_raise <= max_raise and pot_odds < 0.15:
                intended = MULT["rare_bluff"] * pot_base
                amt = self._clamp_raise_amount(intended, min_raise, max_raise)
                if amt is not None:
                    return {"action": "RAISE", "amount": amt}, f"TAG: rare bluff attempt to {amt}."
            return {"action": "FOLD", "amount": 0}, f"TAG: weak ({win_rate:.2%}), fold."

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)
        self.nb_player = game_info["player_num"]

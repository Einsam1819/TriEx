from .llm_base_player import LLMBasePlayer
import random


class LoosePassiveMontePlayer(LLMBasePlayer):
    """Ground-truth Loose-Passive (calling station): high VPIP, low PFR, sticky postflop."""
    PLAYER_ALIAS = "Noah Kim"

    def get_model_config(self):
        return {
            "model": "LoosePassive",
            "use_complex_messages": False,
            "enable_opponent_profiles": False
        }

    def call_llm_api(self, game_state):
        pot_size = int(game_state.get("pot_size", 0))
        call_amount = int(game_state.get("call_amount", 0))
        min_raise = int(game_state.get("min_raise", 0))
        max_raise = int(game_state.get("max_raise", 0))

        win_rate = float(game_state.get("env_hand_strength", 0.5))

        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
        can_check = (call_amount == 0)
        r = random.random()

        # Strong: occasional value raise, mostly call.
        if win_rate >= 0.75 and min_raise <= max_raise and r < 0.35:
            base = pot_size if pot_size > 0 else min_raise
            bet_amount = int(0.7 * base)
            bet_amount = max(min_raise, min(bet_amount, max_raise))
            return {"action": "RAISE", "amount": bet_amount}, f"LP: strong ({win_rate:.2%}), occasional value raise."

        if win_rate >= 0.55:
            if can_check:
                return {"action": "CALL", "amount": 0}, f"LP: decent ({win_rate:.2%}), check."
            return {"action": "CALL", "amount": call_amount}, f"LP: decent ({win_rate:.2%}), call."

        if win_rate >= 0.30:
            if can_check:
                return {"action": "CALL", "amount": 0}, f"LP: weakish ({win_rate:.2%}), free check."

            if pot_odds >= 0.18:
                return {"action": "CALL", "amount": call_amount}, f"LP: weakish ({win_rate:.2%}), call to see more."

            # Sticky call despite poor odds (the LP signature behavior).
            if r < 0.20:
                return {"action": "CALL", "amount": call_amount}, f"LP: sticky call despite poor odds."

            return {"action": "FOLD", "amount": 0}, f"LP: fold when too expensive."

        if can_check:
            return {"action": "CALL", "amount": 0}, f"LP: very weak ({win_rate:.2%}), check."

        if r < 0.10 and pot_odds >= 0.20:
            return {"action": "CALL", "amount": call_amount}, f"LP: occasional curiosity call."

        return {"action": "FOLD", "amount": 0}, f"LP: very weak ({win_rate:.2%}), fold."

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)
        self.nb_player = game_info["player_num"]

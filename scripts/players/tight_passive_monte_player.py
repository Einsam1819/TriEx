from .llm_base_player import LLMBasePlayer
import random


class TightPassiveMontePlayer(LLMBasePlayer):
    """Ground-truth Tight-Passive: low VPIP/PFR/AF, near-zero bluff."""
    PLAYER_ALIAS = "Lily Grant"

    def get_model_config(self):
        return {
            'model': "TightPassive",
            'use_complex_messages': False,
            "enable_opponent_profiles": False
        }

    def call_llm_api(self, game_state):
        pot_size = int(game_state.get('pot_size', 0))
        street = str(game_state.get('street', 'preflop'))
        call_amount = int(game_state.get('call_amount', 0))
        min_raise = int(game_state.get('min_raise', 0))
        max_raise = int(game_state.get('max_raise', 0))

        win_rate = float(game_state.get("env_hand_strength", 0.5))

        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
        can_check = (call_amount == 0)
        r = random.random()

        action_decision = None
        reasoning = ""

        # Very strong: 30% raise, 70% just call/check.
        if win_rate >= 0.80:
            if can_check:
                if r < 0.3 and min_raise <= max_raise:
                    pot_based = int(0.5 * pot_size) if pot_size > 0 else min_raise
                    bet_amount = max(min_raise, min(pot_based, max_raise))
                    action_decision = {"action": "RAISE", "amount": bet_amount}
                    reasoning = (
                        f"Very strong hand ({win_rate:.2%}). Although I am generally conservative, "
                        f"I occasionally raise for value with strong hands."
                    )
                else:
                    action_decision = {"action": "CALL", "amount": call_amount}
                    reasoning = (
                        f"Very strong hand ({win_rate:.2%}). I choose to play it cautiously by just checking/calling."
                    )
            else:
                if r < 0.3 and min_raise <= max_raise:
                    pot_based = int(0.6 * pot_size) if pot_size > 0 else min_raise
                    bet_amount = max(min_raise, min(pot_based, max_raise))
                    action_decision = {"action": "RAISE", "amount": bet_amount}
                    reasoning = (
                        f"Very strong hand ({win_rate:.2%}) with decent pot odds ({pot_odds:.2%}). "
                        f"As a tight-passive player, I rarely raise, but here I make a moderate value raise."
                    )
                else:
                    action_decision = {"action": "CALL", "amount": call_amount}
                    reasoning = (
                        f"Very strong hand ({win_rate:.2%}). I prefer a low-variance line and just call."
                    )

        elif win_rate >= 0.60:
            if can_check:
                if r < 0.1 and min_raise <= max_raise:
                    pot_based = int(0.4 * pot_size) if pot_size > 0 else min_raise
                    bet_amount = max(min_raise, min(pot_based, max_raise))
                    action_decision = {"action": "RAISE", "amount": bet_amount}
                    reasoning = (
                        f"Good hand ({win_rate:.2%}). Occasionally I lead out, but usually I stay passive."
                    )
                else:
                    action_decision = {"action": "CALL", "amount": call_amount}
                    reasoning = (
                        f"Good hand ({win_rate:.2%}). As a conservative player, checking/calling is my default choice."
                    )
            else:
                action_decision = {"action": "CALL", "amount": call_amount}
                reasoning = (
                    f"Good hand ({win_rate:.2%}) with pot odds {pot_odds:.2%}. "
                    f"I call rather than raise to avoid building a huge pot."
                )

        elif win_rate >= 0.40:
            if can_check:
                action_decision = {"action": "CALL", "amount": call_amount}
                reasoning = (
                    f"Medium-strength hand ({win_rate:.2%}). I take the free check instead of risking chips."
                )
            else:
                if pot_odds >= 0.35:
                    action_decision = {"action": "CALL", "amount": call_amount}
                    reasoning = (
                        f"Medium-strength hand ({win_rate:.2%}) and acceptable pot odds ({pot_odds:.2%}). "
                        f"As a tight-passive player, I make a cautious call."
                    )
                else:
                    action_decision = {"action": "FOLD", "amount": 0}
                    reasoning = (
                        f"Medium-strength hand ({win_rate:.2%}) but poor pot odds ({pot_odds:.2%}). "
                        f"I prefer to fold rather than chase marginal spots."
                    )

        else:
            if can_check:
                action_decision = {"action": "CALL", "amount": call_amount}
                reasoning = (
                    f"Weak hand ({win_rate:.2%}). Since it costs nothing to continue, I just check."
                )
            else:
                if pot_odds >= 0.45 and r < 0.1:
                    action_decision = {"action": "CALL", "amount": call_amount}
                    reasoning = (
                        f"Weak hand ({win_rate:.2%}), but very good pot odds ({pot_odds:.2%}). "
                        f"Even as a conservative player, I sometimes take this cheap spot."
                    )
                else:
                    action_decision = {"action": "FOLD", "amount": 0}
                    reasoning = (
                        f"Weak hand ({win_rate:.2%}) and insufficient pot odds ({pot_odds:.2%}). "
                        f"As a tight-passive player, I fold almost always in this situation."
                    )

        return action_decision, reasoning

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)
        self.nb_player = game_info['player_num']

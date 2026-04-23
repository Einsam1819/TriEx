from .llm_base_player import LLMBasePlayer


class GPTPlayer(LLMBasePlayer):
    PLAYER_ALIAS = "Alex Chen"

    def get_model_config(self):
        return {
            'model': "openai/gpt-4.1-mini",
            'use_complex_messages': False
        }
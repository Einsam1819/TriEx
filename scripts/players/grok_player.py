from .llm_base_player import LLMBasePlayer


class GrokPlayer(LLMBasePlayer):
    PLAYER_ALIAS = "Niko Grey"

    def get_model_config(self):
        return {
            'model': "x-ai/grok-3-mini",
            'use_complex_messages': True
        } 
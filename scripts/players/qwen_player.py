from .llm_base_player import LLMBasePlayer


class QwenPlayer(LLMBasePlayer):
    PLAYER_ALIAS = "Robert Garcia"

    def get_model_config(self):
        return {
            'model': "qwen/qwen3-32b",
            'use_complex_messages': True
        } 
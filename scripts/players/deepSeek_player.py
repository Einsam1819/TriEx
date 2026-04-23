from .llm_base_player import LLMBasePlayer


class DeepSeekPlayer(LLMBasePlayer):
    PLAYER_ALIAS = "Emily Zhang"

    def get_model_config(self):
        return {
            'model': 'deepseek/deepseek-v3.2',
            'use_complex_messages': False,
            }
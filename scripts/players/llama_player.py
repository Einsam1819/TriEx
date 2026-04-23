from .llm_base_player import LLMBasePlayer


class LlamaPlayer(LLMBasePlayer):
    PLAYER_ALIAS = "Sarah Johnson"

    def get_model_config(self):
        return {
            'model': "meta-llama/llama-4-maverick",
            'use_complex_messages': True
        }
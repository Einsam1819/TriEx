from .llm_base_player import LLMBasePlayer


class GeminiPlayer(LLMBasePlayer):
    PLAYER_ALIAS = "Jessica Liu"

    def get_model_config(self):
        return {
            'model': "google/gemini-2.5-flash-lite",
            'use_complex_messages': True,
            }
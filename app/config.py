from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "bria engine"
    app_env: str = "development"
    app_debug: bool = True
    api_v1_prefix: str = "/api/v1"
    #: Shared secret between the API and the engine (env ``ENGINE_API_KEY``). When set,
    #: ``POST /api/v1/forecast`` requires the header ``X-Engine-Key`` with this value;
    #: when empty the engine stays open (local development).
    engine_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

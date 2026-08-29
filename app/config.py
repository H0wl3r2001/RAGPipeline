from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "ollama"
    ollama_port: int = 11434
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_docs"
    embed_model: str = "nomic-embed-text"
    top_k: int = 4
    data_dir: str = "/data"

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()

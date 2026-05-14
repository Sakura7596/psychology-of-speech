from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置，通过环境变量或 .env 文件加载"""

    # 环境
    app_env: str = Field(default="dev", alias="APP_ENV")

    # DeepSeek API
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )

    # OpenAI API (备选)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # LLM 配置
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # 向量数据库
    chroma_persist_dir: str = Field(
        default="./data/embeddings", alias="CHROMA_PERSIST_DIR"
    )

    # 爬虫配置
    scraper_respect_robots: bool = Field(default=True, alias="SCRAPER_RESPECT_ROBOTS")
    scraper_max_retries: int = Field(default=3, alias="SCRAPER_MAX_RETRIES")
    scraper_rate_limit_delay: float = Field(default=2.0, alias="SCRAPER_RATE_LIMIT_DELAY")
    scraper_cases_dir: str = Field(default="data/cases", alias="SCRAPER_CASES_DIR")
    scraper_graph_path: str = Field(
        default="data/graph/psychology_graph.json", alias="SCRAPER_GRAPH_PATH"
    )

    # 日志
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # 离线模式
    @property
    def use_local_llm(self) -> bool:
        return self.app_env == "offline"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

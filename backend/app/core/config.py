from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource

ROOT = Path(__file__).resolve().parents[3]


class MailDotEnvSettingsSource(DotEnvSettingsSource):
    def _read_env_file(self, file_path: Path) -> dict[str, str | None]:
        values = dict(super()._read_env_file(file_path))
        # Normalize each file before merging so the last file wins across aliases.
        old_key = "MAIL_FROM" if self.case_sensitive else "mail_from"
        key = "SMTP_FROM" if self.case_sensitive else "smtp_from"
        if old_key in values:
            values.setdefault(key, values.pop(old_key))
        return values


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT / ".env", ROOT / ".env.local", ROOT / ".env.smtp.local"),
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )
    app_env: Literal["development", "test", "production"] = "development"
    public_base_url: str = "http://localhost:8080"
    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://papermind:papermind@localhost:15432/papermind"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:16379/0")
    mail_backend: Literal["smtp", "memory"] = "smtp"
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_use_ssl: bool = True
    smtp_starttls: bool = False
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")
    mail_from: str = Field(default="", validation_alias=AliasChoices("SMTP_FROM", "MAIL_FROM"))
    mail_from_name: str = "PaperMind"
    mail_encryption_key: SecretStr = SecretStr("")
    session_ttl_days: int = Field(default=7, ge=1, le=30)
    account_token_ttl_minutes: int = Field(default=60, ge=5, le=1440)
    email_code_ttl_minutes: int = Field(default=10, ge=1, le=30)
    email_code_resend_seconds: int = Field(default=60, ge=1, le=600)
    email_code_max_attempts: int = Field(default=5, ge=1, le=10)
    auth_ip_limit: int = Field(default=30, ge=1, le=1000)
    auth_email_limit: int = Field(default=8, ge=1, le=100)
    auth_window_seconds: int = Field(default=900, ge=1)
    user_max_papers: int = Field(default=100, ge=1)
    user_max_storage_mb: int = Field(default=1024, ge=1)
    user_daily_parse_pages: int = Field(default=200, ge=1)
    user_daily_model_tokens: int = Field(default=100000, ge=1)
    user_upload_concurrency: int = Field(default=2, ge=1, le=20)
    user_chat_concurrency: int = Field(default=2, ge=1, le=20)
    provider_concurrency: int = Field(default=8, ge=1, le=100)
    global_daily_model_tokens: int = Field(default=1000000, ge=1)
    storage_root: Path = ROOT / "data"
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    ingestion_max_attempts: int = Field(default=3, ge=1, le=5)
    global_daily_parse_pages: int = Field(default=2000, ge=1)
    max_upload_mb: int = Field(default=100, ge=1)
    max_upload_pages: int = Field(default=200, ge=1)
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY")
    )
    llm_model: str = "deepseek-flash"
    llm_timeout_s: int = Field(default=90, ge=5, le=600)
    llm_fast_base_url: str = "https://api.siliconflow.cn/v1"
    llm_fast_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LLM_FAST_API_KEY", "siliconflow_API_KEY"),
    )
    llm_fast_model: str = "Qwen/Qwen3.8-27B"
    llm_fast_timeout_s: int = Field(default=180, ge=5, le=600)
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "siliconflow_API_KEY"),
    )
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, ge=1)
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("RERANK_API_KEY", "siliconflow_API_KEY"),
    )
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_top_k: int = Field(default=40, ge=1, le=100)
    rerank_top_n: int = Field(default=8, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rrf_vector_weight: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    rrf_bm25_weight: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    rrf_metadata_weight: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    rerank_score_threshold: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    retrieval_max_subqueries: int = Field(default=3, ge=1, le=4)
    retrieval_merge_min_hits: int = Field(default=2, ge=2, le=10)
    retrieval_cross_ref_max_chunks: int = Field(default=32, ge=1, le=64)
    retrieval_background_weight: float = Field(default=0.4, gt=0, le=1, allow_inf_nan=False)
    retrieval_table_weight: float = Field(default=1.5, ge=1, le=3, allow_inf_nan=False)
    retrieval_experiment_weight: float = Field(default=1.3, ge=1, le=3, allow_inf_nan=False)
    retrieval_query_bytes: int = Field(default=1024, ge=64, le=4096)
    rerank_candidate_bytes: int = Field(default=64000, ge=1024, le=256000)
    retrieval_context_bytes: int = Field(default=12000, ge=1024, le=64000)
    retrieval_parent_bytes: int = Field(default=6000, ge=512, le=64000)
    retrieval_max_sources: int = Field(default=24, ge=8, le=64)
    retrieval_parent_max_chunks: int = Field(default=32, ge=1, le=64)
    retrieval_neighbor_window: int = Field(default=1, ge=0, le=2)
    paper_tool_max_bytes: int = Field(default=256000, ge=12000, le=1000000)
    rerank_timeout_s: float = Field(default=20, gt=0, le=120)
    rag_mode: Literal["fast", "agentic"] = "agentic"
    rag_max_retrieval_retries: int = Field(default=2, ge=0, le=2)
    rag_max_generation_retries: int = Field(default=1, ge=0, le=1)
    memory_recent_token_budget: int = Field(default=2000, ge=128, le=8000)
    memory_summarize_after_turns: int = Field(default=8, ge=2, le=100)
    chat_max_output_tokens: int = Field(default=2048, ge=128, le=4096)
    chat_initial_token_reserve: int = Field(default=2048, ge=128, le=16000)
    chat_event_ttl_s: int = Field(default=86400, ge=60, le=604800)
    chat_max_recoveries: int = Field(default=3, ge=1, le=5)
    checkpoint_pool_size: int = Field(default=4, ge=1, le=16)
    mineru_api_base: str = "https://mineru.net/api/v4"
    mineru_api_token: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("MINERU_API_TOKEN", "MinerU_API_KEY")
    )
    mineru_model_version: Literal["vlm", "pipeline"] = "vlm"
    mineru_enable_formula: bool = True
    mineru_enable_table: bool = True
    mineru_language: str = "ch"
    mineru_poll_timeout_s: float = Field(default=900, gt=0, le=3600)
    mineru_poll_interval_s: float = Field(default=3, gt=0, le=30)
    mineru_request_timeout_s: float = Field(default=60, gt=0, le=300)
    mineru_http_attempts: int = Field(default=3, ge=1, le=5)
    mineru_upload_hosts: tuple[str, ...] = ("mineru.oss-cn-shanghai.aliyuncs.com",)
    mineru_download_hosts: tuple[str, ...] = ("cdn-mineru.openxlab.org.cn",)
    mineru_max_archive_mb: int = Field(default=256, ge=1, le=1024)
    mineru_max_expanded_mb: int = Field(default=1024, ge=1, le=4096)
    mineru_max_archive_entries: int = Field(default=10000, ge=1, le=50000)
    tracing_enabled: bool = True
    trace_content_enabled: bool = False
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        assert isinstance(dotenv_settings, DotEnvSettingsSource)
        return (
            init_settings,
            env_settings,
            MailDotEnvSettingsSource(
                settings_cls,
                env_file=dotenv_settings.env_file,
                env_file_encoding=dotenv_settings.env_file_encoding,
            ),
            file_secret_settings,
        )

    @property
    def origin(self) -> str:
        url = urlsplit(self.public_base_url)
        return f"{url.scheme}://{url.netloc}"

    @property
    def secure_cookie(self) -> bool:
        return self.public_base_url.startswith("https://")

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        if self.embedding_dim != 1024:
            raise ValueError("当前数据库索引要求 EMBEDDING_DIM=1024")
        parsed = urlsplit(self.public_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("PUBLIC_BASE_URL 必须是完整的同源地址，不能包含路径")
        if self.smtp_use_ssl and self.smtp_starttls:
            raise ValueError("SMTP_USE_SSL 与 SMTP_STARTTLS 不可同时启用")
        if self.tracing_enabled and not (
            self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value()
        ):
            raise ValueError("启用追踪时必须配置 Langfuse 凭据")
        if self.app_env == "production":
            if not self.secure_cookie or parsed.hostname in {"localhost", "127.0.0.1"}:
                raise ValueError("生产环境必须配置公开 HTTPS 地址")
            if self.mail_backend != "smtp" or not self.smtp_host or not self.mail_from:
                raise ValueError("生产环境必须配置 SMTP 邮件服务")
            if not self.mail_encryption_key.get_secret_value():
                raise ValueError("必须配置 MAIL_ENCRYPTION_KEY 加密邮件任务")
            if not self.smtp_use_ssl and not self.smtp_starttls:
                raise ValueError("生产邮件必须使用 TLS")
            if ":papermind@" in self.database_url.get_secret_value():
                raise ValueError("生产数据库不能使用示例密码")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.providers.clients import chat_model, embeddings


def test_legacy_credentials_and_no_secret_repr(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-main")
    monkeypatch.setenv("siliconflow_API_KEY", "secret-fast")
    monkeypatch.setenv("MinerU_API_KEY", "secret-parser")
    settings = Settings(_env_file=None, tracing_enabled=False)
    assert settings.llm_api_key.get_secret_value() == "secret-main"
    assert settings.embedding_api_key.get_secret_value() == "secret-fast"
    assert settings.mineru_api_token.get_secret_value() == "secret-parser"
    assert "secret-main" not in repr(settings)


def test_invalid_runtime_configuration():
    with pytest.raises(ValidationError, match="Langfuse"):
        Settings(_env_file=None)
    with pytest.raises(ValidationError, match="HTTPS"):
        Settings(_env_file=None, app_env="production", tracing_enabled=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tracing_enabled=False, user_daily_model_tokens=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tracing_enabled=False, smtp_use_ssl=True, smtp_starttls=True)


@pytest.mark.parametrize("key", ["MAIL_FROM", "SMTP_FROM"])
def test_smtp_sender_environment_aliases(monkeypatch, key):
    monkeypatch.delenv("MAIL_FROM", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    monkeypatch.setenv(key, "sender@example.org")
    settings = Settings(_env_file=None, tracing_enabled=False)
    assert settings.mail_from == "sender@example.org"


@pytest.mark.parametrize("first,second", [("MAIL_FROM", "SMTP_FROM"), ("SMTP_FROM", "MAIL_FROM")])
def test_smtp_file_overrides_previous_sender_alias(tmp_path, monkeypatch, first, second):
    for key in (
        "MAIL_FROM",
        "SMTP_FROM",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    base = tmp_path / ".env"
    local = tmp_path / ".env.local"
    smtp = tmp_path / ".env.smtp.local"
    base.write_text(f"{first}=base@example.org\nSMTP_HOST=old.example.org\n")
    local.write_text(f"{first}=local@example.org\nSMTP_PASSWORD=old-password\n")
    smtp.write_text(
        f"{second}=sender@example.org\nSMTP_HOST=smtp.example.org\nSMTP_PORT=465\n"
        "SMTP_USERNAME=account@example.org\nSMTP_PASSWORD=smtp-secret\n"
    )
    settings = Settings(_env_file=(base, local, smtp), tracing_enabled=False)
    assert settings.mail_from == "sender@example.org"
    assert settings.smtp_host == "smtp.example.org"
    assert settings.smtp_port == 465
    assert settings.smtp_use_ssl and not settings.smtp_starttls
    assert settings.smtp_username == "account@example.org"
    assert settings.smtp_password.get_secret_value() == "smtp-secret"
    assert "smtp-secret" not in repr(settings)


@pytest.mark.parametrize("key", ["MAIL_FROM", "SMTP_FROM"])
def test_process_sender_overrides_smtp_file(tmp_path, monkeypatch, key):
    monkeypatch.delenv("MAIL_FROM", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    smtp = tmp_path / ".env.smtp.local"
    smtp.write_text("SMTP_FROM=file@example.org\n")
    monkeypatch.setenv(key, "process@example.org")
    assert Settings(_env_file=smtp, tracing_enabled=False).mail_from == "process@example.org"


def test_missing_optional_smtp_file_preserves_existing_config(tmp_path, monkeypatch):
    monkeypatch.delenv("MAIL_FROM", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    base = tmp_path / ".env"
    base.write_text("MAIL_FROM=existing@example.org\n")
    settings = Settings(_env_file=(base, tmp_path / ".env.smtp.local"), tracing_enabled=False)
    assert settings.mail_from == "existing@example.org"


def test_provider_factories_keep_selected_models_and_disable_auto_chunking():
    settings = Settings(
        _env_file=None,
        tracing_enabled=False,
        llm_api_key="test",
        llm_fast_api_key="test",
        embedding_api_key="test",
    )
    assert chat_model(settings).model_name == "deepseek-flash"
    assert chat_model(settings, "fast").model_name == "Qwen/Qwen3.8-27B"
    assert embeddings(settings).check_embedding_ctx_length is False

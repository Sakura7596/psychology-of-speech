import os
import pytest
from unittest.mock import patch


def test_default_config():
    """测试默认配置值"""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
        from src.config import Settings
        settings = Settings()
        assert settings.app_env == "dev"
        assert settings.llm_temperature == 0.3
        assert settings.llm_max_tokens == 4096


def test_env_override():
    """测试环境变量覆盖"""
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
        "APP_ENV": "prod",
        "LLM_TEMPERATURE": "0.7",
    }, clear=False):
        from src.config import Settings
        settings = Settings()
        assert settings.app_env == "prod"
        assert settings.llm_temperature == 0.7


def test_offline_mode():
    """测试离线模式配置"""
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
        "APP_ENV": "offline",
    }, clear=False):
        from src.config import Settings
        settings = Settings()
        assert settings.app_env == "offline"
        assert settings.use_local_llm is True


def test_missing_api_key_in_dev():
    """测试 dev 环境缺少 API key 时的处理"""
    with patch.dict(os.environ, {}, clear=True):
        from src.config import Settings
        with pytest.raises(Exception):
            Settings()

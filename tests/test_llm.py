import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.llm.client import LLMClient, LLMResponse
from src.llm.deepseek import DeepSeekAdapter
from src.llm.prompts import PromptTemplates


def test_llm_response_creation():
    """测试 LLMResponse 数据模型"""
    resp = LLMResponse(
        content="分析结果",
        model="deepseek-chat",
        tokens_used=100,
        finish_reason="stop",
    )
    assert resp.content == "分析结果"
    assert resp.tokens_used == 100


def test_deepseek_adapter_init():
    """测试 DeepSeek 适配器初始化"""
    adapter = DeepSeekAdapter(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
    assert adapter.model == "deepseek-chat"


def test_llm_client_init():
    """测试 LLM 客户端初始化"""
    client = LLMClient(adapter=DeepSeekAdapter(
        api_key="test-key",
        base_url="https://api.deepseek.com",
    ))
    assert client.adapter is not None


async def test_llm_client_generate():
    """测试 LLM 客户端生成（mock）"""
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = LLMResponse(
        content="测试响应",
        model="deepseek-chat",
        tokens_used=50,
        finish_reason="stop",
    )

    client = LLMClient(adapter=mock_adapter)
    response = await client.generate("测试提示词")

    assert response.content == "测试响应"
    mock_adapter.generate.assert_called_once_with("测试提示词", None)


async def test_llm_client_generate_with_system():
    """测试带系统提示的生成"""
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = LLMResponse(
        content="响应",
        model="deepseek-chat",
        tokens_used=30,
        finish_reason="stop",
    )

    client = LLMClient(adapter=mock_adapter)
    response = await client.generate("用户提示", system_prompt="系统指令")

    mock_adapter.generate.assert_called_once_with("用户提示", "系统指令")


async def test_deepseek_adapter_generate_success():
    """测试 DeepSeek 适配器成功生成响应"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={
        "choices": [{"message": {"content": "测试"}, "finish_reason": "stop"}],
        "model": "deepseek-chat",
        "usage": {"total_tokens": 42},
    })

    adapter = DeepSeekAdapter(api_key="test-key")
    with patch.object(adapter._client, "post", return_value=mock_response):
        result = await adapter.generate("hello")

    assert result.content == "测试"
    assert result.tokens_used == 42
    assert result.finish_reason == "stop"


def test_get_system_prompt():
    """测试获取系统提示词"""
    prompt = PromptTemplates.get_system_prompt("text_analyst")
    assert "语言学" in prompt or "文本" in prompt
    assert len(prompt) > 50


def test_get_analysis_prompt():
    """测试获取分析提示词"""
    prompt = PromptTemplates.get_analysis_prompt(
        "text_analyst",
        text="今天天气真好",
        depth="standard",
    )
    assert "今天天气真好" in prompt
    assert "standard" in prompt.lower() or "标准" in prompt


def test_get_report_prompt():
    """测试获取报告生成提示词"""
    analyses = {
        "text_analyst": {"result": "test"},
        "psychology_analyst": {"result": "test"},
    }
    prompt = PromptTemplates.get_report_prompt(
        text="原始文本",
        analyses=analyses,
        depth="standard",
    )
    assert "原始文本" in prompt
    assert "text_analyst" in prompt


def test_unknown_agent_raises():
    """测试未知 Agent 名称抛出异常"""
    with pytest.raises(KeyError):
        PromptTemplates.get_system_prompt("nonexistent_agent")

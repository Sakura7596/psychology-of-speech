import pytest
from src.guardrails.hallucination import HallucinationGuard
from src.guardrails.ethics import EthicsGuard
from src.guardrails.privacy import PrivacyGuard


def test_hallucination_check_valid():
    guard = HallucinationGuard()
    analysis = {"conclusion": "比喻修辞", "sources": ["HanLP"], "confidence": 0.8}
    result = guard.check(analysis)
    assert result["passed"] is True


def test_hallucination_check_no_sources():
    guard = HallucinationGuard()
    analysis = {"conclusion": "心理问题", "sources": [], "confidence": 0.9}
    result = guard.check(analysis)
    assert result["passed"] is False


def test_hallucination_check_low_confidence():
    guard = HallucinationGuard()
    analysis = {"conclusion": "推测", "sources": ["LLM"], "confidence": 0.1}
    result = guard.check(analysis)
    assert "推测" in result.get("warning", "") or result["passed"] is True


def test_ethics_disclaimer_injection():
    guard = EthicsGuard()
    report = "分析结果：比喻修辞。"
    result = guard.inject_disclaimer(report)
    assert "非专业" in result or "仅供参考" in result or "声明" in result


def test_ethics_detect_diagnostic_language():
    guard = EthicsGuard()
    result = guard.check_diagnostic_language("说话者患有抑郁症")
    assert result["has_diagnostic"] is True


def test_ethics_clean_text():
    guard = EthicsGuard()
    result = guard.check_diagnostic_language("说话者使用了比喻修辞")
    assert result["has_diagnostic"] is False


def test_privacy_mask_phone():
    guard = PrivacyGuard()
    result = guard.mask_pii("联系 13812345678")
    assert "138****5678" in result


def test_privacy_mask_email():
    guard = PrivacyGuard()
    result = guard.mask_pii("发送到 test@example.com")
    assert "[邮箱已脱敏]" in result
    assert "test@example.com" not in result


def test_privacy_no_pii():
    guard = PrivacyGuard()
    result = guard.mask_pii("今天天气真好")
    assert result == "今天天气真好"


def test_privacy_check_retention():
    guard = PrivacyGuard()
    policy = guard.get_retention_policy()
    assert "本地" in policy or "local" in policy

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.scraper.base import RawContent, ScrapingStatus, ScrapedCase
from src.scraper.cleaners import ContentCleaner
from src.scraper.validator import ContentValidator
from src.scraper.storage import StorageManager
from src.scraper.analyzer import ContentAnalyzer


# === Base ===

def test_raw_content_creation():
    raw = RawContent(source="zhihu", url="https://example.com", title="Test", text="内容")
    assert raw.source == "zhihu"
    assert raw.status == ScrapingStatus.SUCCESS
    assert raw.error == ""


def test_raw_content_with_error():
    raw = RawContent(
        source="douban", url="https://example.com", title="", text="",
        status=ScrapingStatus.FAILED, error="connection timeout",
    )
    assert raw.status == ScrapingStatus.FAILED
    assert raw.error == "connection timeout"


def test_scraped_case_creation():
    case = ScrapedCase(
        source="zhihu", url="https://example.com", raw_text="text",
        case_data={"type": "dating"}, confidence=0.8,
    )
    assert case.source == "zhihu"
    assert case.case_data["type"] == "dating"


# === ContentCleaner ===

def test_clean_html_strips_scripts():
    cleaner = ContentCleaner()
    html = '<html><body><p>你好</p><script>evil()</script></body></html>'
    result = cleaner.clean_html(html)
    assert "evil" not in result
    assert "你好" in result


def test_clean_html_strips_styles():
    cleaner = ContentCleaner()
    html = '<html><head><style>body{color:red}</style></head><body>内容</body></html>'
    result = cleaner.clean_html(html)
    assert "color:red" not in result
    assert "内容" in result


def test_clean_text_removes_urls():
    cleaner = ContentCleaner()
    text = "看看这个 https://example.com/path 很有趣"
    result = cleaner.clean_text(text)
    assert "https" not in result
    assert "很有趣" in result


def test_clean_text_removes_mentions():
    cleaner = ContentCleaner()
    text = "@用户A 你说得对"
    result = cleaner.clean_text(text)
    assert "@用户A" not in result
    assert "你说得对" in result


def test_clean_text_removes_hashtag_markers():
    cleaner = ContentCleaner()
    text = "#恋爱日记# 今天很开心"
    result = cleaner.clean_text(text)
    assert "恋爱日记" in result
    assert "#" not in result


def test_is_relationship_content_positive():
    cleaner = ContentCleaner()
    assert cleaner.is_relationship_content("我和男朋友吵架了") is True
    assert cleaner.is_relationship_content("分手后好难过") is True
    assert cleaner.is_relationship_content("暧昧对象突然不回消息") is True


def test_is_relationship_content_negative():
    cleaner = ContentCleaner()
    assert cleaner.is_relationship_content("今天天气不错") is False
    assert cleaner.is_relationship_content("Python编程入门") is False


def test_mask_pii_delegates():
    cleaner = ContentCleaner()
    text = "我的电话是13812345678"
    result = cleaner.mask_pii(text)
    assert "13812345678" not in result


def test_extract_dialogue():
    cleaner = ContentCleaner()
    text = '他说"你很好"然后就走了'
    dialogues = cleaner.extract_dialogue(text)
    assert len(dialogues) == 1
    assert dialogues[0]["text"] == "你很好"


# === ContentValidator ===

def _valid_case():
    return {
        "type": "dating", "subtype": "breadcrumbing",
        "text": "在吗？好久没聊了。最近怎么样？",
        "keywords": ["在吗", "好久没聊"],
        "analysis": "面包屑策略：偶尔发消息保持联系但不推进关系，以最小化投入维持对方兴趣。",
        "psychological_state": "不想认真但想保持对方的兴趣作为备选",
        "theories": ["现代约会心理学"],
    }


def test_validator_valid_case():
    validator = ContentValidator()
    passed, issues = validator.validate_case(_valid_case())
    assert passed is True
    assert len(issues) == 0


def test_validator_rejects_missing_fields():
    validator = ContentValidator()
    case = {"type": "dating"}
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("缺少字段" in i for i in issues)


def test_validator_rejects_short_text():
    validator = ContentValidator()
    case = _valid_case()
    case["text"] = "短"
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("过短" in i for i in issues)


def test_validator_rejects_long_text():
    validator = ContentValidator()
    case = _valid_case()
    case["text"] = "A" * 501
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("过长" in i for i in issues)


def test_validator_rejects_few_keywords():
    validator = ContentValidator()
    case = _valid_case()
    case["keywords"] = ["一个"]
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("关键词" in i for i in issues)


def test_validator_rejects_short_analysis():
    validator = ContentValidator()
    case = _valid_case()
    case["analysis"] = "短分析"
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("分析" in i for i in issues)


def test_validator_rejects_duplicate():
    validator = ContentValidator(existing_cases=[{"text": "重复文本内容测试"}])
    case = _valid_case()
    case["text"] = "重复文本内容测试"
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("重复" in i for i in issues)


def test_validator_rejects_invalid_type():
    validator = ContentValidator()
    case = _valid_case()
    case["type"] = "invalid_type"
    passed, issues = validator.validate_case(case)
    assert any("类型" in i for i in issues)


def test_validator_rejects_no_theories():
    validator = ContentValidator()
    case = _valid_case()
    case["theories"] = []
    passed, issues = validator.validate_case(case)
    assert passed is False
    assert any("理论" in i for i in issues)


def test_generate_case_id():
    validator = ContentValidator()
    case_id = validator.generate_case_id("romantic_breakup", "zhihu", {"rb_001", "rb_002"})
    assert case_id == "rb_003"


def test_generate_case_id_no_existing():
    validator = ContentValidator()
    case_id = validator.generate_case_id("dating", "douban", set())
    assert case_id == "dt_001"


def test_generate_case_id_unknown_type():
    validator = ContentValidator()
    case_id = validator.generate_case_id("unknown", "zhihu", set())
    assert case_id == "sc_001"


# === ContentAnalyzer (mocked LLM) ===

async def test_analyzer_parses_valid_json():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = Mock(
        content=json.dumps({
            "type": "dating", "subtype": "breadcrumbing",
            "text": "test text", "keywords": ["a", "b"],
            "analysis": "面包屑策略分析内容",
            "psychological_state": "state",
            "theories": ["理论"],
        }, ensure_ascii=False),
    )
    analyzer = ContentAnalyzer(llm_client=mock_llm)
    result = await analyzer.analyze_to_case("some text", "zhihu", "https://example.com")
    assert result is not None
    assert result["type"] == "dating"
    assert result["_source"] == "zhihu"


async def test_analyzer_handles_skip():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = Mock(
        content=json.dumps({"skip": True, "reason": "广告内容"}, ensure_ascii=False),
    )
    analyzer = ContentAnalyzer(llm_client=mock_llm)
    result = await analyzer.analyze_to_case("广告", "zhihu", "https://example.com")
    assert result is None


async def test_analyzer_handles_markdown_code_block():
    mock_llm = AsyncMock()
    case_data = {
        "type": "dating", "subtype": "test",
        "text": "test text", "keywords": ["a", "b"],
        "analysis": "analysis content here",
        "psychological_state": "state",
        "theories": ["t"],
    }
    mock_llm.generate.return_value = Mock(
        content=f"```json\n{json.dumps(case_data, ensure_ascii=False)}\n```",
    )
    analyzer = ContentAnalyzer(llm_client=mock_llm)
    result = await analyzer.analyze_to_case("text", "zhihu", "url")
    assert result is not None
    assert result["type"] == "dating"


async def test_analyzer_handles_llm_error():
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("API error")
    analyzer = ContentAnalyzer(llm_client=mock_llm)
    result = await analyzer.analyze_to_case("text", "zhihu", "url")
    assert result is None


async def test_batch_analyze():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = Mock(
        content=json.dumps({
            "type": "dating", "subtype": "test",
            "text": "test", "keywords": ["a", "b"],
            "analysis": "analysis content",
            "psychological_state": "state",
            "theories": ["t"],
        }, ensure_ascii=False),
    )
    analyzer = ContentAnalyzer(llm_client=mock_llm)
    contents = [
        RawContent(source="zhihu", url="u1", title="t1", text="text1"),
        RawContent(source="douban", url="u2", title="t2", text="text2"),
    ]
    results = await analyzer.batch_analyze(contents, concurrency=2)
    assert len(results) == 2


# === StorageManager (temp dir) ===

def test_storage_save_case(tmp_path):
    storage = StorageManager(
        cases_dir=str(tmp_path),
        graph_path=str(tmp_path / "graph.json"),
    )
    case = {
        "type": "dating", "subtype": "test",
        "text": "test text", "keywords": ["a"],
        "analysis": "analysis", "psychological_state": "state",
        "theories": ["theory"],
    }
    case_id = storage.save_case(case)
    assert case_id is not None

    with open(tmp_path / "scraped_dating_cases.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 1
    assert saved[0]["id"] == case_id
    assert saved[0]["text"] == "test text"


def test_storage_save_case_appends(tmp_path):
    storage = StorageManager(
        cases_dir=str(tmp_path),
        graph_path=str(tmp_path / "graph.json"),
    )
    case1 = {
        "type": "dating", "subtype": "test1",
        "text": "text1", "keywords": ["a"],
        "analysis": "analysis1", "psychological_state": "state1",
        "theories": ["t"],
    }
    case2 = {
        "type": "dating", "subtype": "test2",
        "text": "text2", "keywords": ["b"],
        "analysis": "analysis2", "psychological_state": "state2",
        "theories": ["t"],
    }
    storage.save_case(case1)
    storage.save_case(case2)

    with open(tmp_path / "scraped_dating_cases.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 2


def test_storage_strips_internal_fields(tmp_path):
    storage = StorageManager(
        cases_dir=str(tmp_path),
        graph_path=str(tmp_path / "graph.json"),
    )
    case = {
        "type": "dating", "subtype": "test",
        "text": "text", "keywords": ["a"],
        "analysis": "analysis", "psychological_state": "state",
        "theories": ["t"],
        "_source": "zhihu", "_url": "https://example.com",
    }
    storage.save_case(case)

    with open(tmp_path / "scraped_dating_cases.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert "_source" not in saved[0]
    assert "_url" not in saved[0]


def test_storage_batch_save(tmp_path):
    storage = StorageManager(
        cases_dir=str(tmp_path),
        graph_path=str(tmp_path / "graph.json"),
    )
    cases = [
        {
            "type": "dating", "subtype": "test1",
            "text": "text1", "keywords": ["a"],
            "analysis": "analysis1", "psychological_state": "state1",
            "theories": ["t"],
        },
        {
            "type": "romantic_breakup", "subtype": "test2",
            "text": "text2", "keywords": ["b"],
            "analysis": "analysis2", "psychological_state": "state2",
            "theories": ["t"],
        },
    ]
    ids = storage.save_cases_batch(cases)
    assert len(ids) == 2

    with open(tmp_path / "scraped_dating_cases.json", "r", encoding="utf-8") as f:
        dating = json.load(f)
    assert len(dating) == 1

    with open(tmp_path / "scraped_romantic_breakup_cases.json", "r", encoding="utf-8") as f:
        breakup = json.load(f)
    assert len(breakup) == 1


# === Pipeline (fully mocked) ===

async def test_pipeline_dry_run():
    from src.scraper.pipeline import ScrapePipeline

    mock_scraper = AsyncMock()
    mock_scraper.scrape.return_value = [
        RawContent(
            source="test", url="url", title="title",
            text="我和男朋友吵架了，他说你从来都不关心我",
        ),
    ]

    mock_cleaner = Mock()
    mock_cleaner.clean_text.return_value = "我和男朋友吵架了，他说你从来都不关心我"
    mock_cleaner.is_relationship_content.return_value = True
    mock_cleaner.mask_pii.return_value = "我和男朋友吵架了，他说你从来都不关心我"

    mock_analyzer = AsyncMock()
    mock_analyzer.batch_analyze.return_value = [
        {
            "type": "romantic_conflict", "subtype": "accusation",
            "text": "你从来都不关心我", "keywords": ["从来不", "不关心"],
            "analysis": "绝对化指责分析",
            "psychological_state": "感到被忽视",
            "theories": ["认知扭曲理论"],
        },
    ]

    mock_validator = Mock()
    mock_validator.validate_case.return_value = (True, [])

    mock_storage = Mock()

    pipeline = ScrapePipeline(
        scrapers={"test": mock_scraper},
        cleaner=mock_cleaner,
        analyzer=mock_analyzer,
        validator=mock_validator,
        storage=mock_storage,
    )

    stats = await pipeline.run(query="吵架", sources=["test"], dry_run=True)

    assert stats["scraped"] == 1
    assert stats["cleaned"] == 1
    assert stats["analyzed"] == 1
    assert stats["validated"] == 1
    assert stats["stored"] == 0
    assert "dry_run_cases" in stats
    assert len(stats["dry_run_cases"]) == 1
    mock_storage.save_case.assert_not_called()


async def test_pipeline_stores_when_not_dry_run():
    from src.scraper.pipeline import ScrapePipeline

    mock_scraper = AsyncMock()
    mock_scraper.scrape.return_value = [
        RawContent(source="test", url="url", title="title", text="恋爱内容"),
    ]

    mock_cleaner = Mock()
    mock_cleaner.clean_text.return_value = "恋爱内容"
    mock_cleaner.is_relationship_content.return_value = True
    mock_cleaner.mask_pii.return_value = "恋爱内容"

    mock_analyzer = AsyncMock()
    mock_analyzer.batch_analyze.return_value = [
        {
            "type": "dating", "subtype": "test",
            "text": "恋爱内容", "keywords": ["恋爱"],
            "analysis": "分析内容", "psychological_state": "state",
            "theories": ["t"],
        },
    ]

    mock_validator = Mock()
    mock_validator.validate_case.return_value = (True, [])

    mock_storage = Mock()
    mock_storage.save_case.return_value = "dt_001"

    pipeline = ScrapePipeline(
        scrapers={"test": mock_scraper},
        cleaner=mock_cleaner,
        analyzer=mock_analyzer,
        validator=mock_validator,
        storage=mock_storage,
    )

    stats = await pipeline.run(query="恋爱", sources=["test"], dry_run=False)

    assert stats["stored"] == 1
    mock_storage.save_case.assert_called_once()
    mock_storage.update_graph.assert_called_once()


async def test_pipeline_handles_scraper_failure():
    from src.scraper.pipeline import ScrapePipeline

    mock_scraper = AsyncMock()
    mock_scraper.scrape.side_effect = Exception("network error")

    mock_cleaner = Mock()
    mock_analyzer = AsyncMock()
    mock_analyzer.batch_analyze.return_value = []
    mock_validator = Mock()
    mock_storage = Mock()

    pipeline = ScrapePipeline(
        scrapers={"test": mock_scraper},
        cleaner=mock_cleaner,
        analyzer=mock_analyzer,
        validator=mock_validator,
        storage=mock_storage,
    )

    stats = await pipeline.run(query="test", sources=["test"])
    assert stats["scraped"] == 0
    assert len(stats["errors"]) > 0

import asyncio
import logging

from src.scraper.base import BaseScraper, RawContent, ScrapingStatus

logger = logging.getLogger(__name__)


class ScrapePipeline:
    def __init__(
        self,
        scrapers: dict[str, BaseScraper],
        cleaner,
        analyzer,
        validator,
        storage,
    ):
        self._scrapers = scrapers
        self._cleaner = cleaner
        self._analyzer = analyzer
        self._validator = validator
        self._storage = storage

    async def run(
        self,
        query: str,
        sources: list[str] | None = None,
        max_items_per_source: int = 20,
        dry_run: bool = False,
    ) -> dict:
        if sources is None:
            sources = list(self._scrapers.keys())

        stats = {
            "scraped": 0, "cleaned": 0, "analyzed": 0,
            "validated": 0, "stored": 0, "errors": [],
        }

        # Phase 1: Crawl
        all_raw: list[RawContent] = []
        crawl_tasks = [
            self._scrapers[s].scrape(query, max_items_per_source)
            for s in sources if s in self._scrapers
        ]
        crawl_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)
        for result in crawl_results:
            if isinstance(result, list):
                all_raw.extend(result)
            elif isinstance(result, Exception):
                stats["errors"].append(f"爬取错误: {result}")
        stats["scraped"] = len(all_raw)

        # Phase 2: Clean + Filter
        cleaned = []
        for raw in all_raw:
            if raw.status != ScrapingStatus.SUCCESS:
                continue
            cleaned_text = self._cleaner.clean_text(raw.text)
            if not self._cleaner.is_relationship_content(cleaned_text):
                continue
            cleaned_text = self._cleaner.mask_pii(cleaned_text)
            raw.text = cleaned_text
            cleaned.append(raw)
        stats["cleaned"] = len(cleaned)

        # Phase 3: LLM Analyze
        analyzed = await self._analyzer.batch_analyze(cleaned)
        stats["analyzed"] = len(analyzed)

        # Phase 4: Validate
        validated = []
        for case in analyzed:
            passed, issues = self._validator.validate_case(case)
            if passed:
                validated.append(case)
            else:
                stats["errors"].append(f"验证失败: {issues}")
        stats["validated"] = len(validated)

        # Phase 5: Store
        if not dry_run:
            for case in validated:
                try:
                    self._storage.save_case(case)
                    self._storage.update_graph(case)
                    stats["stored"] += 1
                except Exception as e:
                    stats["errors"].append(f"存储失败: {e}")
        else:
            stats["dry_run_cases"] = validated

        return stats

"""CLI for batch scraping"""
import argparse
import asyncio
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="心理学话语分析 - 数据采集工具")
    subparsers = parser.add_subparsers(dest="command")

    scrape_parser = subparsers.add_parser("scrape", help="采集数据")
    scrape_parser.add_argument("query", help="搜索关键词")
    scrape_parser.add_argument(
        "--sources", nargs="+", default=None,
        help="数据源 (zhihu, douban, xiaohongshu, psychology_blog)",
    )
    scrape_parser.add_argument("--max-items", type=int, default=20, help="每个源最大采集数")
    scrape_parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入知识库")
    scrape_parser.add_argument("--no-robots", action="store_true", help="忽略 robots.txt 限制")
    scrape_parser.add_argument("--output", help="输出统计 JSON 路径")

    gen_parser = subparsers.add_parser("generate", help="使用 LLM 生成合成数据")
    gen_parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入知识库")
    gen_parser.add_argument("--output", help="输出统计 JSON 路径")

    subparsers.add_parser("list-sources", help="列出可用数据源")

    args = parser.parse_args()

    if args.command == "scrape":
        asyncio.run(run_scrape(args))
    elif args.command == "generate":
        asyncio.run(run_generate(args))
    elif args.command == "list-sources":
        print("可用数据源:")
        print("  zhihu           - 知乎问答")
        print("  douban          - 豆瓣小组")
        print("  xiaohongshu     - 小红书笔记")
        print("  psychology_blog - 心理学博客（壹心理、简单心理）")
    else:
        parser.print_help()


async def run_scrape(args):
    from src.config import get_settings
    from src.llm.client import LLMClient
    from src.llm.deepseek import DeepSeekAdapter
    from src.scraper.pipeline import ScrapePipeline
    from src.scraper.http_client import ResilientHttpClient
    from src.scraper.cleaners import ContentCleaner
    from src.scraper.analyzer import ContentAnalyzer
    from src.scraper.validator import ContentValidator
    from src.scraper.storage import StorageManager
    from src.scraper.sources.zhihu import ZhihuScraper
    from src.scraper.sources.douban import DoubanScraper
    from src.scraper.sources.xiaohongshu import XiaohongshuScraper
    from src.scraper.sources.psychology_blog import PsychologyBlogScraper

    settings = get_settings()

    respect_robots = not getattr(args, 'no_robots', False)
    http = ResilientHttpClient(respect_robots=respect_robots)
    llm_adapter = DeepSeekAdapter(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    llm_client = LLMClient(adapter=llm_adapter)

    scrapers = {
        "zhihu": ZhihuScraper(http),
        "douban": DoubanScraper(http),
        "xiaohongshu": XiaohongshuScraper(http),
        "psychology_blog": PsychologyBlogScraper(http),
    }

    pipeline = ScrapePipeline(
        scrapers=scrapers,
        cleaner=ContentCleaner(),
        analyzer=ContentAnalyzer(llm_client),
        validator=ContentValidator(),
        storage=StorageManager(),
    )

    print(f"开始采集: '{args.query}'")
    print(f"数据源: {args.sources or '全部'}")
    print(f"模式: {'预览' if args.dry_run else '写入'}")
    print()

    stats = await pipeline.run(
        query=args.query,
        sources=args.sources,
        max_items_per_source=args.max_items,
        dry_run=args.dry_run,
    )

    print("采集完成:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计已保存到: {args.output}")

    await http.close()


async def run_generate(args):
    from src.config import get_settings
    from src.llm.client import LLMClient
    from src.llm.deepseek import DeepSeekAdapter
    from src.scraper.synthetic import SyntheticDataGenerator
    from src.scraper.cleaners import ContentCleaner
    from src.scraper.analyzer import ContentAnalyzer
    from src.scraper.validator import ContentValidator
    from src.scraper.storage import StorageManager

    settings = get_settings()
    llm_adapter = DeepSeekAdapter(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    llm_client = LLMClient(adapter=llm_adapter)

    generator = SyntheticDataGenerator(llm_client)
    cleaner = ContentCleaner()
    analyzer = ContentAnalyzer(llm_client)
    validator = ContentValidator()
    storage = StorageManager()

    dry_run = getattr(args, 'dry_run', False)

    print("开始生成合成数据...")
    raw_items = await generator.generate_batch(concurrency=3)
    print(f"生成了 {len(raw_items)} 条原始数据")

    # Clean + Filter
    cleaned = []
    for raw in raw_items:
        cleaned_text = cleaner.clean_text(raw.text)
        if cleaner.is_relationship_content(cleaned_text):
            raw.text = cleaned_text
            cleaned.append(raw)
    print(f"清洗后 {len(cleaned)} 条")

    # Analyze
    print("正在用 LLM 分析...")
    analyzed = await analyzer.batch_analyze(cleaned, concurrency=3)
    print(f"分析完成 {len(analyzed)} 条")

    # Validate
    validated = []
    for case in analyzed:
        passed, issues = validator.validate_case(case)
        if passed:
            validated.append(case)
    print(f"验证通过 {len(validated)} 条")

    # Store
    if not dry_run:
        stored = 0
        for case in validated:
            try:
                storage.save_case(case)
                storage.update_graph(case)
                stored += 1
            except Exception as e:
                print(f"存储失败: {e}")
        print(f"已存储 {stored} 条案例")
    else:
        print(f"预览模式，未写入。共 {len(validated)} 条待写入。")
        for case in validated[:5]:
            print(f"  - [{case.get('type')}/{case.get('subtype')}] {case.get('text', '')[:50]}...")

    stats = {
        "generated": len(raw_items),
        "cleaned": len(cleaned),
        "analyzed": len(analyzed),
        "validated": len(validated),
        "stored": len(validated) if not dry_run else 0,
    }

    output = getattr(args, 'output', None)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计已保存到: {output}")


if __name__ == "__main__":
    main()

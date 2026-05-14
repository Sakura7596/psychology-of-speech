import logging
from urllib.parse import urljoin, quote

from bs4 import BeautifulSoup

from src.scraper.base import BaseScraper, RawContent, ScrapingStatus

logger = logging.getLogger(__name__)


class PsychologyBlogScraper(BaseScraper):
    SEED_URLS = [
        "https://www.xinli001.com/info/article",
        "https://www.jiandanxinli.com/",
    ]

    SEARCH_SITES = [
        "https://www.xinli001.com/search?q={query}",
        "https://www.jiandanxinli.com/search?q={query}",
    ]

    @property
    def source_name(self) -> str:
        return "psychology_blog"

    def build_search_url(self, query: str, page: int = 1) -> str:
        return self.SEARCH_SITES[0].format(query=quote(query))

    async def scrape(self, query: str, max_items: int = 20) -> list[RawContent]:
        results = []

        for search_template in self.SEARCH_SITES:
            search_url = search_template.format(query=quote(query))
            try:
                response = await self._http.get(search_url)
                soup = BeautifulSoup(response.text, "lxml")

                links = soup.select(
                    "a[href*='/info/'], a[href*='/article/'], a[href*='/wiki/']"
                )

                seen_urls = set()
                for link in links[:max_items * 2]:
                    try:
                        href = link.get("href", "")
                        if not href:
                            continue
                        if not href.startswith("http"):
                            href = urljoin(search_url, href)
                        if href in seen_urls:
                            continue
                        seen_urls.add(href)

                        title = link.get_text(strip=True)
                        if len(title) < 4:
                            continue

                        detail = await self._http.get(href)
                        detail_soup = BeautifulSoup(detail.text, "lxml")

                        content_el = detail_soup.select_one(
                            "article, .article-content, .content, .wiki-content, .info-content"
                        )
                        if not content_el:
                            content_el = detail_soup.select_one("main, .main-content")
                        if not content_el:
                            continue

                        text = content_el.get_text(strip=True)
                        if len(text) < 50:
                            continue

                        if not self._is_psychology_related(text, title):
                            continue

                        results.append(RawContent(
                            source="psychology_blog",
                            url=href,
                            title=title,
                            text=text[:2000],
                        ))

                        if len(results) >= max_items:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to fetch article {href}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to search {search_url}: {e}")
                continue

        logger.info(f"PsychologyBlog: scraped {len(results)} items for '{query}'")
        return results

    def _is_psychology_related(self, text: str, title: str) -> bool:
        psychology_keywords = [
            "心理", "情感", "情绪", "认知", "行为", "人格", "依恋",
            "焦虑", "抑郁", "沟通", "关系", "恋爱", "婚姻", "家庭",
            "自我", "成长", "疗愈", "咨询", "治疗", "创伤",
        ]
        combined = f"{title} {text}"
        return any(kw in combined for kw in psychology_keywords)

import logging
from urllib.parse import quote

from bs4 import BeautifulSoup

from src.scraper.base import BaseScraper, RawContent, ScrapingStatus

logger = logging.getLogger(__name__)


class DoubanScraper(BaseScraper):
    SEARCH_URL = "https://www.douban.com/group/search?cat=1013&q={query}"

    GROUP_URLS = {
        "dating": "https://www.douban.com/group/jiayouzhe/",
        "breakup": "https://www.douban.com/group/fenshouzhe/",
        "relationship": "https://www.douban.com/group/haoyouzhe/",
    }

    @property
    def source_name(self) -> str:
        return "douban"

    def build_search_url(self, query: str, page: int = 1) -> str:
        start = (page - 1) * 25
        return f"{self.SEARCH_URL.format(query=quote(query))}&start={start}"

    async def scrape(self, query: str, max_items: int = 20) -> list[RawContent]:
        results = []
        url = self.build_search_url(query)
        try:
            response = await self._http.get(url, headers={
                "Referer": "https://www.douban.com/",
            })
            soup = BeautifulSoup(response.text, "lxml")

            items = soup.select("td.td-subject a, .result .title a")
            for item in items[:max_items]:
                try:
                    title = item.get_text(strip=True)
                    link = item.get("href", "")

                    if not link or "douban.com" not in link:
                        continue

                    detail = await self._http.get(link, headers={
                        "Referer": url,
                    })
                    detail_soup = BeautifulSoup(detail.text, "lxml")

                    content_el = detail_soup.select_one(
                        ".topic-content, .note-content, .article-content"
                    )
                    if not content_el:
                        continue

                    text = content_el.get_text(strip=True)
                    if len(text) < 20:
                        continue

                    author_el = detail_soup.select_one(
                        ".topic-doc .from a, .note-author, .author-name"
                    )
                    author = author_el.get_text(strip=True) if author_el else ""

                    results.append(RawContent(
                        source="douban",
                        url=link,
                        title=title,
                        text=text[:2000],
                        author=author,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse douban item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Douban scrape failed: {e}")
            return [RawContent(
                source="douban", url=url, title="", text="",
                status=ScrapingStatus.FAILED, error=str(e),
            )]

        logger.info(f"Douban: scraped {len(results)} items for '{query}'")
        return results

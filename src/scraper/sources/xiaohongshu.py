import logging
from urllib.parse import quote

from bs4 import BeautifulSoup

from src.scraper.base import BaseScraper, RawContent, ScrapingStatus

logger = logging.getLogger(__name__)


class XiaohongshuScraper(BaseScraper):
    SEARCH_URL = "https://www.xiaohongshu.com/search_result?keyword={query}&source=web_search_result_notes"

    @property
    def source_name(self) -> str:
        return "xiaohongshu"

    def build_search_url(self, query: str, page: int = 1) -> str:
        return self.SEARCH_URL.format(query=quote(query))

    async def scrape(self, query: str, max_items: int = 20) -> list[RawContent]:
        results = []
        url = self.build_search_url(query)
        try:
            response = await self._http.get(url, headers={
                "Referer": "https://www.xiaohongshu.com/",
                "Origin": "https://www.xiaohongshu.com",
            })
            soup = BeautifulSoup(response.text, "lxml")

            cards = soup.select(
                ".note-item, .feeds-page .note-item, section.note-item"
            )

            if not cards:
                cards = soup.select("[data-note-id], .search-result-item")

            for card in cards[:max_items]:
                try:
                    title_el = card.select_one(
                        ".title, .note-title, span.title"
                    )
                    title = title_el.get_text(strip=True) if title_el else ""

                    desc_el = card.select_one(
                        ".desc, .note-desc, .content"
                    )
                    text = desc_el.get_text(strip=True) if desc_el else title

                    if len(text) < 10:
                        continue

                    link_el = card.select_one("a[href*='/explore/'], a[href*='/discovery/']")
                    link = ""
                    if link_el:
                        link = link_el.get("href", "")
                        if link and not link.startswith("http"):
                            link = f"https://www.xiaohongshu.com{link}"

                    author_el = card.select_one(
                        ".author-wrapper .name, .user-name, .nickname"
                    )
                    author = author_el.get_text(strip=True) if author_el else ""

                    results.append(RawContent(
                        source="xiaohongshu",
                        url=link,
                        title=title,
                        text=text[:2000],
                        author=author,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse xiaohongshu card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Xiaohongshu scrape failed: {e}")
            return [RawContent(
                source="xiaohongshu", url=url, title="", text="",
                status=ScrapingStatus.FAILED, error=str(e),
            )]

        logger.info(f"Xiaohongshu: scraped {len(results)} items for '{query}'")
        return results

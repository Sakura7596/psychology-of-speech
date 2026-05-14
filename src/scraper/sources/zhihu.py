import logging
from urllib.parse import quote

from bs4 import BeautifulSoup

from src.scraper.base import BaseScraper, RawContent, ScrapingStatus

logger = logging.getLogger(__name__)


class ZhihuScraper(BaseScraper):
    SEARCH_URL = "https://www.zhihu.com/search?type=content&q={query}"

    @property
    def source_name(self) -> str:
        return "zhihu"

    def build_search_url(self, query: str, page: int = 1) -> str:
        return self.SEARCH_URL.format(query=quote(query))

    async def scrape(self, query: str, max_items: int = 20) -> list[RawContent]:
        results = []
        url = self.build_search_url(query)
        try:
            response = await self._http.get(url, headers={
                "Referer": "https://www.zhihu.com/",
            })
            soup = BeautifulSoup(response.text, "lxml")

            cards = soup.select(".SearchResult-Card, .List-item")
            for card in cards[:max_items]:
                try:
                    title_el = card.select_one("h2, .ContentItem-title")
                    title = title_el.get_text(strip=True) if title_el else ""

                    content_el = card.select_one(".RichContent-inner, .CopyrightRichText")
                    if not content_el:
                        content_el = card.select_one("span.RichText")
                    text = content_el.get_text(strip=True) if content_el else ""

                    if len(text) < 20:
                        continue

                    link_el = card.select_one("a[href*='/question/'], a[href*='/answer/']")
                    link = link_el["href"] if link_el else ""
                    if link and not link.startswith("http"):
                        link = f"https://www.zhihu.com{link}"

                    author_el = card.select_one(".AuthorInfo-name, .UserLink-link")
                    author = author_el.get_text(strip=True) if author_el else ""

                    results.append(RawContent(
                        source="zhihu",
                        url=link,
                        title=title,
                        text=text[:2000],
                        author=author,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse zhihu card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Zhihu scrape failed: {e}")
            return [RawContent(
                source="zhihu", url=url, title="", text="",
                status=ScrapingStatus.FAILED, error=str(e),
            )]

        logger.info(f"Zhihu: scraped {len(results)} items for '{query}'")
        return results

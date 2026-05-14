import asyncio
import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


class ScraperConnectionError(Exception):
    pass


class ScrapingNotAllowedError(Exception):
    pass


class ResilientHttpClient:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 2.0,
        rate_limit_delay: float = 1.0,
        respect_robots: bool = True,
        timeout: float = 30.0,
    ):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._rate_limit_delay = rate_limit_delay
        self._respect_robots = respect_robots
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._ua = UserAgent()
        self._rate_limiter: dict[str, float] = {}
        self._robots_cache: dict[str, RobotFileParser] = {}

    async def get(self, url: str, **kwargs) -> httpx.Response:
        domain = urlparse(url).netloc
        if self._respect_robots:
            await self._check_robots(url, domain)
        await self._rate_limit(domain)
        headers = {**self._get_headers(), **kwargs.pop("headers", {})}

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.get(url, headers=headers, **kwargs)
                if response.status_code == 429:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(f"429 rate limited on {domain}, retry in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                if response.status_code == 403:
                    self._ua = UserAgent()
                    logger.warning(f"403 on {domain}, rotating UA")
                    delay = self._base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(f"HTTP {e.response.status_code} on {domain}, retry in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise ScraperConnectionError(f"HTTP {e.response.status_code}: {url}") from e
            except httpx.RequestError as e:
                last_error = e
                delay = self._base_delay * (2 ** attempt)
                logger.warning(f"Request error on {domain}, retry in {delay}s: {e}")
                await asyncio.sleep(delay)
                continue

        raise ScraperConnectionError(f"Failed after {self._max_retries} retries: {url}") from last_error

    async def post(self, url: str, **kwargs) -> httpx.Response:
        domain = urlparse(url).netloc
        await self._rate_limit(domain)
        headers = {**self._get_headers(), **kwargs.pop("headers", {})}
        response = await self._client.post(url, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def _get_headers(self) -> dict:
        return {
            "User-Agent": self._ua.random,
            "Accept": "text/html,application/xhtml+xml,application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }

    async def _check_robots(self, url: str, domain: str) -> None:
        if domain in self._robots_cache:
            rp = self._robots_cache[domain]
        else:
            rp = RobotFileParser()
            robots_url = f"https://{domain}/robots.txt"
            try:
                resp = await self._client.get(robots_url, timeout=10.0)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.parse([])
            except Exception:
                rp.parse([])
            self._robots_cache[domain] = rp

        if not rp.can_fetch("*", url):
            raise ScrapingNotAllowedError(f"robots.txt disallows: {url}")

    async def _rate_limit(self, domain: str) -> None:
        now = time.monotonic()
        last = self._rate_limiter.get(domain, 0)
        wait = self._rate_limit_delay - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._rate_limiter[domain] = time.monotonic()

    async def close(self) -> None:
        await self._client.aclose()

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ScrapingStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"


@dataclass
class RawContent:
    source: str
    url: str
    title: str
    text: str
    author: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)
    status: ScrapingStatus = ScrapingStatus.SUCCESS
    error: str = ""


@dataclass
class ScrapedCase:
    source: str
    url: str
    raw_text: str
    case_data: dict
    confidence: float
    validation: dict = field(default_factory=dict)


class BaseScraper(ABC):
    def __init__(self, http_client):
        self._http = http_client

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...

    @abstractmethod
    async def scrape(self, query: str, max_items: int = 20) -> list[RawContent]:
        ...

    @abstractmethod
    def build_search_url(self, query: str, page: int = 1) -> str:
        ...

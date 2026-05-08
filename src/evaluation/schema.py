# src/evaluation/schema.py
from dataclasses import dataclass, field
import json


@dataclass
class BenchmarkItem:
    """评测基准项"""
    id: str
    text: str
    expected: dict
    category: str = "general"
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)


def validate_benchmark(items: list[BenchmarkItem]) -> bool:
    """验证基准数据集"""
    ids = set()
    for item in items:
        if not item.id:
            raise ValueError("Item ID cannot be empty")
        if item.id in ids:
            raise ValueError(f"Duplicate ID: {item.id}")
        ids.add(item.id)
        if not item.text:
            raise ValueError(f"Empty text for item {item.id}")
        if not item.expected:
            raise ValueError(f"Empty expected for item {item.id}")
    return True


def load_benchmark(path: str) -> list[BenchmarkItem]:
    """加载基准数据集"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [BenchmarkItem(**item) for item in data]

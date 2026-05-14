import re

from bs4 import BeautifulSoup


RELATIONSHIP_KEYWORDS = [
    "恋爱", "男朋友", "女朋友", "分手", "暧昧", "表白", "追求",
    "约会", "感情", "恋人", "男友", "女友", "前任", "相亲",
    "暗恋", "出轨", "冷暴力", "异地恋", "PUA", "舔狗",
    "渣男", "渣女", "备胎", "绿茶", "心动", "喜欢",
    "爱", "在一起", "离开", "挽回", "复合", "吵架",
    "安全感", "吃醋", "想你", "聊天", "已读不回",
]


class ContentCleaner:
    def clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return self.clean_text(text)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)#', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n\s*\n', '\n', text)
        return text

    def is_relationship_content(self, text: str) -> bool:
        return any(kw in text for kw in RELATIONSHIP_KEYWORDS)

    def mask_pii(self, text: str) -> str:
        from src.guardrails.privacy import PrivacyGuard
        return PrivacyGuard().mask_pii(text)

    def extract_dialogue(self, text: str) -> list[dict]:
        pattern = r'[""「]([^""」]+)[""」]'
        matches = re.findall(pattern, text)
        dialogues = []
        for m in matches:
            dialogues.append({"text": m, "type": "quote"})
        return dialogues

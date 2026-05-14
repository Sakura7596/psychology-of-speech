import re


class PrivacyGuard:
    PHONE_PATTERN = r'(1[3-9]\d)\d{4}(\d{4})'
    EMAIL_PATTERN = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    ID_CARD_PATTERN = r'(\d{6})\d{8}(\d{4})'

    def mask_pii(self, text: str) -> str:
        text = re.sub(self.PHONE_PATTERN, r'\1****\2', text)
        text = re.sub(self.EMAIL_PATTERN, '[邮箱已脱敏]', text)
        text = re.sub(self.ID_CARD_PATTERN, r'\1****\2', text)
        return text

    def detect_pii(self, text: str) -> dict:
        phones = re.findall(self.PHONE_PATTERN, text)
        emails = re.findall(self.EMAIL_PATTERN, text)
        ids = re.findall(self.ID_CARD_PATTERN, text)
        return {
            "has_pii": bool(phones or emails or ids),
            "phones": len(phones),
            "emails": len(emails),
            "id_cards": len(ids),
        }

    def get_retention_policy(self) -> str:
        return "数据处理优先在本地完成。分析文本不会上传至第三方服务器。用户可随时删除本地数据。"

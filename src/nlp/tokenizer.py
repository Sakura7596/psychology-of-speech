import re
import jieba
import jieba.posseg


class Tokenizer:
    """分词器 - HanLP 主力 + jieba 后备"""

    def __init__(self, use_hanlp: bool = True):
        self._use_hanlp = use_hanlp
        self._hanlp_tokenizer = None
        self._hanlp_pos = None

    def _get_hanlp_tokenizer(self):
        if self._hanlp_tokenizer is None:
            import hanlp
            self._hanlp_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        return self._hanlp_tokenizer

    def _get_hanlp_pos(self):
        if self._hanlp_pos is None:
            import hanlp
            self._hanlp_pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        return self._hanlp_pos

    def tokenize(self, text: str) -> list[str]:
        """分词"""
        if not text or not text.strip():
            return []
        if self._use_hanlp:
            try:
                tokenizer = self._get_hanlp_tokenizer()
                return tokenizer(text)
            except Exception:
                pass
        return list(jieba.cut(text))

    def tokenize_with_pos(self, text: str) -> list[tuple[str, str]]:
        """分词 + 词性标注"""
        if not text or not text.strip():
            return []
        if self._use_hanlp:
            try:
                tokens = self.tokenize(text)
                pos_tagger = self._get_hanlp_pos()
                pos_tags = pos_tagger(tokens)
                return list(zip(tokens, pos_tags))
            except Exception:
                pass
        return [(p.word, p.flag) for p in jieba.posseg.cut(text)]

    def split_sentences(self, text: str) -> list[str]:
        """断句"""
        sentences = re.split(r'[。！？!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def add_custom_words(self, words: list[str]) -> None:
        """添加自定义词到 jieba 词典"""
        for word in words:
            jieba.add_word(word)

    def load_custom_dict(self, dict_path: str) -> None:
        """加载自定义词典文件"""
        jieba.load_userdict(dict_path)

import numpy as np


class EmbeddingModel:
    """中文 Embedding 模型 - text2vec-base-chinese"""

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self._model_name = model_name
        self._model = None
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, text: str) -> list[float]:
        """编码单个文本"""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        model = self._get_model()
        vecs = model.encode(texts)
        result = []
        for v in vecs:
            if hasattr(v, "tolist"):
                result.append(v.tolist())
            else:
                result.append(v)
        return result

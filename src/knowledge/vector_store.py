from src.knowledge.embedding import EmbeddingModel


class VectorStore:
    """ChromaDB 向量库封装"""

    def __init__(self, persist_dir: str = "./data/embeddings", collection_name: str = "default"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding = EmbeddingModel()

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        collection = self._get_collection()
        embedding = self._embedding.encode(text)
        kwargs = {
            "ids": [doc_id],
            "embeddings": [embedding],
            "documents": [text],
        }
        if metadata:
            kwargs["metadatas"] = [metadata]
        collection.add(**kwargs)

    def add_batch(self, ids: list[str], texts: list[str], metadatas: list[dict] | None = None) -> None:
        collection = self._get_collection()
        embeddings = self._embedding.encode_batch(texts)
        kwargs = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": texts,
        }
        if metadatas:
            kwargs["metadatas"] = metadatas
        collection.add(**kwargs)

    def query(self, text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        collection = self._get_collection()
        embedding = self._embedding.encode(text)

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(n_results, max(collection.count(), 1)),
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        items = []
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                items.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })
        return items

    def delete(self, doc_id: str) -> None:
        collection = self._get_collection()
        collection.delete(ids=[doc_id])

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def close(self) -> None:
        """释放 ChromaDB 资源，解除文件锁"""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None

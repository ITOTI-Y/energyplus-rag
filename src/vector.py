from abc import ABC, abstractmethod
from chunk import Chunk

from loguru import logger


class IVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query: list[float], top_k: int, chunk_type: str | None = None, score_threshold: float | None = None) -> list[dict]:
        pass


class QdrantVectorStore(IVectorStore):
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        prefer_grpc: bool = True,
        dimension: int = 768,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )
        self.collection_name = collection_name
        self.distance = Distance
        self.vector_params = VectorParams
        self.dimension = dimension
        self._create_collection()

    def _create_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.vector_params(
                    size=self.dimension,
                    distance=self.distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=chunk.to_qdrant_payload(),
            ))

        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
            )

        logger.info(f"Added {len(chunks)} chunks to {self.collection_name}")

    def search(
        self,
        query: list[float],
        top_k: int = 10,
        chunk_type: str | None = None,
        score_threshold: float | None = None,
    ):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = None
        if chunk_type:
            search_filter = Filter(
                must=[FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type),
                )]
            )

        query_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        results = [{
            "content": result.payload["content"] if result.payload else "",
            "section": result.payload["section"] if result.payload else "",
            "chunk_type": result.payload["chunk_type"] if result.payload else "",
            "page_idx": result.payload["page_idx"] if result.payload else "",
            "file_name": result.payload["file_name"] if result.payload else "",
            "score": result.score,
            "metadata": {
                k: v for k, v in result.payload.items() if k not in ["content", "section", "chunk_type", "page_idx"] # type: ignore
            }
        } for result in query_results]

        return results

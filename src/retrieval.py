from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient


class RetrievalResult(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    text: str = Field(..., description="The text of the retrieved document")
    score: float = Field(...,
                         description="The score of the retrieved document")
    source: str = Field(...,
                        description="The source of the retrieved document")
    chunk_type: str = Field(..., description="The type of the retrieved chunk")
    header: str | None = Field(
        None, description="The header of the retrieved chunk")


class HybridRetriever:
    def __init__(
            self,
            qdrant_client: QdrantClient,
            collection_name: str,
            embedding_model: str = "gemini-embedding-001",
            use_rerank: bool = True,
    ):
        pass

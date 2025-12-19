from abc import ABC, abstractmethod
from enum import Enum
from time import sleep

import numpy as np
from dotenv import load_dotenv

load_dotenv()


class IEmbeddingModel(ABC):
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass


class GeminiTaskType(Enum):
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class GeminiEmbeddingModel(IEmbeddingModel):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        dimension: int = 768,
    ):
        from google.genai import Client

        self.client: Client = Client(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension

    def embed_batch(
        self,
        texts: list[str],
        task_type: GeminiTaskType = GeminiTaskType.RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        from google.genai import types

        values = []
        for batched_texts in range(0, len(texts), 100):
            sleep(2)
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts[batched_texts:min(batched_texts+100, len(texts))],  # type: ignore
                config=types.EmbedContentConfig(
                    task_type=task_type.value,
                    output_dimensionality=self.dimension,
                ),
            )
            if not result.embeddings:
                values = [[0.0] * self.dimension for _ in texts]
            else:
                for embedding in result.embeddings:
                    if self.dimension != 3072:
                        emb = np.array(embedding.values)
                        emb = emb / np.linalg.norm(emb)
                        values.append(emb.tolist())
                    else:
                        values.append(embedding.values)
        return values

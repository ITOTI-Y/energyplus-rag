from collections.abc import Generator
from pathlib import Path

from src.chunk import Chunk, DocumentProcessor
from src.embedding import GeminiEmbeddingModel
from src.parse import parse_pdf
from src.vector import QdrantVectorStore


class RAGSystem:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        qdrant_collection_name: str,
        gemini_api_key: str,
    ):
        self.document_processor = DocumentProcessor()
        self.vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection_name,
        )
        self.embedding_model = GeminiEmbeddingModel(api_key=gemini_api_key)

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: str | None = None,
        score_threshold: float | None = 0.5,
    ) -> list[dict]:
        embeddings = self.embedding_model.embed_batch([query])[0]
        results = self.vector_store.search(
            embeddings, top_k, chunk_type, score_threshold
        )
        return results

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: str | None = None,
        score_threshold: float | None = 0.5,
    ) -> str:
        results = self.search(query, top_k, chunk_type, score_threshold)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results):
            section = result.get("section", "Unknown")
            content = result.get("content", "")
            score = result.get("score", 0.0)
            page_idx = result.get("page_idx", "Unknown")
            file_name = result.get("file_name", "Unknown")

            context_parts.append(
                f"[Document {i + 1}] (Source: {file_name}, Section: {section}, Page: {page_idx}, Score: {score:.2f})\n\n{content}\n\n---\n\n"
            )

        return "\n".join(context_parts)

    def chunk(
        self,
        json_file: str,
    ) -> list[Chunk]:
        content_list = self.document_processor.load_content_list(json_file)
        chunks = self.document_processor.process_document(content_list)
        return chunks

    def parse(
        self,
        pdf_dir: str,
        output_dir: str | Path,
    ) -> Generator[str, None, None]:
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        total_files = len(pdf_files)

        for i, pdf_file in enumerate(pdf_files):
            yield f"Parsing {pdf_file.name} ({i + 1}/{total_files})"

            json_file = ""
            for progress_msg in parse_pdf(pdf_file, output_dir):
                if progress_msg.startswith("RESULT:"):
                    json_file = progress_msg[7:]
                else:
                    yield progress_msg

            yield f"Chunking {json_file}..."
            try:
                chunks = self.chunk(json_file)
                yield f"Chunked {json_file} into {len(chunks)} chunks"
            except Exception as e:
                yield f"Error chunking {json_file}: {e!s}"
                continue

            yield f"Embedding {json_file}..."
            try:
                embeddings = self.embed([chunk.content for chunk in chunks])
                yield f"Embedded {json_file} into {len(embeddings)} embeddings"
            except Exception as e:
                yield f"Error embedding {json_file}: {e!s}"
                continue

            yield f"Adding {json_file} to vector store..."
            try:
                self.vector_store.add(chunks, embeddings)
                yield f"Added {json_file} to vector store"
            except Exception as e:
                yield f"Error adding {json_file} to vector store: {e!s}"
                continue

            yield f"{pdf_file.name} parsed and added to vector store"

        yield f"All {total_files} PDF files parsed and added to vector store"

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        result = self.embedding_model.embed_batch(texts)
        return result

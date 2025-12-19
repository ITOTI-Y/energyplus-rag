import argparse
import os
import shutil
import tempfile

import gradio as gr
from dotenv import load_dotenv

from src.model import Qwen3Model
from src.rag import RAGSystem
from src.ui import ChatbotUI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

RAG_SYSTEM_PROMPT = """
You are a professional EnergyPlus assistant. Please base your answers to user questions on the following reference documents.

# Documentation
{context}

## Answer requirements
1. Answer the question using the information in the reference document in preference to the information in the reference document.
2. If the information is not in the reference document, clearly inform the user.
3. Answers should be accurate, professional, and concise.
4. cite document sources if necessary
"""


def main():
    parser = argparse.ArgumentParser(description="EnergyPlus-RAG")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")
    args = parser.parse_args()

    model = Qwen3Model()

    def chat_interface(query: str, chatbot: list, use_rag: bool):
        if not query.strip():
            yield "", chatbot, ""
            return
        chatbot.append({"role": "user", "content": query})
        chatbot.append({"role": "assistant", "content": ""})

        messages = []

        rag_context = ""
        if use_rag:
            try:
                rag_context = rag_system.build_context(query)
            except Exception as e:
                rag_context = f"RAG Retrieval Failed with error: {e!s}"

        if use_rag and rag_context:
            system_message = RAG_SYSTEM_PROMPT.format(context=rag_context)
            messages.append({"role": "system", "content": system_message})

        for msg in chatbot[:-1]:
            messages.append(msg)

        full_response = ""
        full_thinking = ""

        if use_rag:
            if rag_context:
                full_thinking = f"**Reference documents:**\n\n {rag_context} \n\n Thinking Process:\n\n"
            else:
                full_thinking = "**No reference documents found**\n\n Thinking Process:\n\n"

        for thinking, response in model.chat_stream(messages):
            full_thinking += thinking
            full_response += response
            chatbot[-1]["content"] = full_response
            yield "", chatbot, full_thinking

    def upload_handler(pdf_file: list[gr.File]):
        if not pdf_file:
            yield "No PDF file uploaded"
            return

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in pdf_file:
                    shutil.copy(file.name, temp_dir) # type: ignore
                    yield f"Copied {file.name} to {temp_dir}" # type: ignore

                yield f"Parsing {len(pdf_file)} PDF files..."

                yield from (progress_msg for progress_msg in rag_system.parse(temp_dir, "./output"))

            yield f"{len(pdf_file)} PDF files uploaded and parsed successfully"
        except Exception as e:
            yield f"Failed to upload and parse PDF files: {e!s}"

    rag_system = RAGSystem(
        qdrant_url=QDRANT_ENDPOINT or "",
        qdrant_api_key=QDRANT_API_KEY or "",
        qdrant_collection_name=QDRANT_COLLECTION_NAME or "",
        gemini_api_key=GEMINI_API_KEY or "",
    )
    try:
        ui = ChatbotUI(port=args.port)
        ui.launch(chat_interface, upload_handler)
    finally:
        model.close()

def data_preprocess(out_put_dir: str):
    from pathlib import Path
    rag_system = RAGSystem(
        qdrant_url=QDRANT_ENDPOINT or "",
        qdrant_api_key=QDRANT_API_KEY or "",
        qdrant_collection_name=QDRANT_COLLECTION_NAME or "",
        gemini_api_key=GEMINI_API_KEY or "",
    )
    json_files = Path(out_put_dir).glob("*/*.json")

    for json_file in json_files:
        chunks = rag_system.chunk(str(json_file))
        embeddings = rag_system.embed([chunk.content for chunk in chunks])
        rag_system.vector_store.add(chunks, embeddings)
    pass

if __name__ == "__main__":
    # data_preprocess("./output")
    main()

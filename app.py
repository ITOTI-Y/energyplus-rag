import argparse

import gradio as gr
from dotenv import load_dotenv

from src.ui import ChatbotUI

load_dotenv()


def chat_interface(query: str, chatbot: gr.Chatbot, use_hybrid: bool, use_expansion: bool):
    pass


def init_system(use_rerank: bool = True):
    pass

def main():
    parser = argparse.ArgumentParser(description="EnergyPlus-RAG")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable reranking")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")
    args = parser.parse_args()

    init_system(
        use_rerank=not args.no_rerank,
    )

    ui = ChatbotUI(port=args.port)
    ui.launch(chat_interface)


if __name__ == "__main__":
    main()

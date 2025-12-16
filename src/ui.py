from collections.abc import Callable

import gradio as gr


class ChatbotUI:
    def __init__(
        self,
        theme: gr.Theme | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        if theme is None:
            theme = gr.themes.Soft()  # type: ignore
        self.theme = theme
        self.host = host
        self.port = port

    def _create_ui(self, chat_interface: Callable, upload_handler: Callable):
        with gr.Blocks() as demo:
            gr.Markdown(
                """
                # EnergyPlus-RAG
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=800,
                        label="Chatbot",
                    )

                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Enter your query here...",
                            label="Query",
                            scale=4,
                            lines=2,
                        )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Submit", variant="primary", scale=1)
                            use_rag_btn = gr.Checkbox(
                                label="Use RAG",
                                value=True,
                            )

                    with gr.Row():
                        clear_btn = gr.Button("Clear", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("Thinking Process")

                    thinking_output = gr.Markdown(
                        value="",
                        height=500,
                    )

                    gr.Markdown("Document Upload")
                    pdf_file = gr.File(
                        label="Upload PDF File",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("Upload and Parse", variant="secondary")
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        lines=3
                    )

            gr.Markdown("Question Examples")
            _ = gr.Examples(
                inputs=query_input,
                examples=[
                    ["EnergyPlus中Zone对象的主要参数有哪些?"],
                    ["如何设置建筑材料的热导率?"],
                    ["HVAC系统的控制策略有哪些选项?"],
                    ["如何定义建筑的外墙构造?"],
                    ["Timestep参数如何设置?"],
                ],
            )

            submit_btn.click(
                chat_interface,
                inputs=[query_input, chatbot, use_rag_btn],
                outputs=[query_input, chatbot, thinking_output],
            )

            query_input.submit(
                chat_interface,
                inputs=[query_input, chatbot, use_rag_btn],
                outputs=[query_input, chatbot, thinking_output],
            )

            clear_btn.click(
                fn=lambda: ("", [], ""),
                inputs=None,
                outputs=[query_input, chatbot, thinking_output],
            )
            if upload_handler:
                upload_btn.click(
                    upload_handler,
                    inputs=[pdf_file],
                    outputs=[upload_status],
                )
        return demo

    def launch(self, chat_interface: Callable, upload_handler: Callable):
        app = self._create_ui(chat_interface, upload_handler)
        app.launch(
            server_name=self.host,
            server_port=self.port,
            share=False,
            show_error=True,
        )

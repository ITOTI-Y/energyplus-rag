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

    def _create_ui(self, chat_interface: Callable):
        with gr.Blocks() as demo:

            gr.Markdown(
                """
                # EnergyPlus-RAG
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Chatbot",
                    )

                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Enter your query here...",
                            label="Query",
                            scale=4,
                            lines=2
                        )
                        submit_btn = gr.Button(
                            "Submit", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("Query Configuration")

                    use_hybrid = gr.Checkbox(
                        label="Use Hybrid Search (Keyword + Vector)",
                        value=True,
                    )

                    use_expansion = gr.Checkbox(
                        label="Use Expansion",
                        value=True,
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
                ]
            )

            submit_btn.click(
                chat_interface,
                inputs=[query_input, chatbot, use_hybrid, use_expansion],
                outputs=[query_input, chatbot],
            )

            query_input.submit(
                chat_interface,
                inputs=[query_input, chatbot, use_hybrid, use_expansion],
                outputs=[query_input, chatbot],
            )

            clear_btn.click(
                fn=lambda: ([], "", ""),
                inputs=None,
                outputs=[query_input, chatbot],
            )
        return demo

    def launch(self, chat_interface: Callable):
        app = self._create_ui(chat_interface)
        app.launch(
            server_name=self.host,
            server_port=self.port,
            share=False,
            show_error=True,
        )

import atexit
import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Generator

import requests
from loguru import logger


class IModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, str]:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str) -> Generator[tuple[str, str], None, None]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def chat_stream(self, messages: list[dict]) -> Generator[tuple[str, str]]:
        pass


class OllamaManager:
    def __init__(
        self,
        model_name: str,
        host: str = "127.0.0.1",
        port: int = 11434,
        auto_pull: bool = True,
        auto_serve: bool = True,
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._serve_process: subprocess.Popen | None = None

        if auto_serve:
            self.ensure_serving()

        if auto_pull:
            self.ensure_model_pulled()

        atexit.register(self.close)

    def is_serving(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def ensure_serving(self, timeout: int = 10):
        if self.is_serving():
            logger.info(f"Ollama is already serving {self.base_url}")
            return

        logger.info(f"Starting Ollama server at {self.base_url}")
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"

        self._serve_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_serving():
                logger.info(f"Ollama server started at {self.base_url}")
                return
            time.sleep(1)
        raise RuntimeError("Failed to start Ollama server")

    def is_model_pulled(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(
                    m.get("name", "").startswith(self.model_name.split(":")[0])
                    for m in models
                )
        except requests.RequestException:
            return False
        return False

    def ensure_model_pulled(self):
        if self.is_model_pulled():
            logger.info(f"Model {self.model_name} is already pulled")
            return

        logger.info(f"Pulling model {self.model_name}")
        process = subprocess.Popen(
            ["ollama", "pull", self.model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout:
            for line in process.stdout:
                print(f"  {line.strip()}")

        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"Failed to pull model {self.model_name}")

        logger.info(f"Model {self.model_name} pulled successfully")

    def close(self):
        if self._serve_process:
            logger.info(f"Stopping Ollama server at {self._serve_process.pid}")

            self._serve_process.terminate()
            try:
                self._serve_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._serve_process.kill()
                self._serve_process.wait()
            self._serve_process = None
            logger.info("Ollama server stopped")


class Qwen3Model(IModel):
    def __init__(
        self,
        model_name: str = "hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q4_K_M",
        host: str = "127.0.0.1",
        port: int = 11434,
        auto_pull: bool = True,
        auto_serve: bool = True,
    ):
        self.manager = OllamaManager(
            model_name=model_name,
            host=host,
            port=port,
            auto_pull=auto_pull,
            auto_serve=auto_serve,
        )
        self.model_name = model_name
        self.base_url = self.manager.base_url

    def generate(self, prompt: str) -> tuple[str, str]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 0.0,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("thinking", ""), result.get("response", "")

    def generate_stream(self, prompt: str) -> Generator[tuple[str, str], None, None]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 0.0,
                },
            },
            stream=True,
            timeout=300,
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data.get("thinking", ""), data.get("response", "")

    def chat_stream(
        self, messages: list[dict]
    ) -> Generator[tuple[str, str]]:
        def normalize_message(msgs: list[dict]):
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = "".join(text_parts)
                msg["content"] = content
            return msgs

        messages = normalize_message(messages)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 20,
                },
            },
            stream=True,
            timeout=300,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data:
                    content = data.get("message", {}).get("content", "")
                    thinking = data.get("message", {}).get("thinking", "")
                    yield thinking, content

    def close(self):
        self.manager.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

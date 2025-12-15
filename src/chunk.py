import hashlib
import json
import re

from markdownify import markdownify as md
from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    content: str = Field(description="The content of the chunk")
    chunk_type: str = Field(description="The type of the chunk")
    page_idx: int = Field(description="The page index of the chunk")
    bbox: list[int] = Field(description="The bounding box of the chunk")
    section_path: list[str] | None = Field(
        None, description="The section path of the chunk"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="The metadata of the chunk",
    )

    @property
    def chunk_id(self) -> str:
        return hashlib.md5(
            f"{self.page_idx}_{self.bbox}_{self.content[:100]}".encode()
        ).hexdigest()[:16]

    def to_qdrant_payload(self) -> dict:
        return {
            "content": self.content,
            "chunk_type": self.chunk_type,
            "page_idx": self.page_idx,
            "bbox": self.bbox,
            "section_path": self.section_path,
            "section_name": self.section_path[-1] if self.section_path else None,
            **self.metadata,
        }


class TableProcessor:
    @staticmethod
    def parse_html_table(html: str) -> str:
        markdown = md(html,table_infer_header=True)
        return markdown

    @staticmethod
    def linearize_table(
        html: str,
        caption: str = "",
        footnote: str = "",
    ) -> str:
        table_body = TableProcessor.parse_html_table(html)
        if not table_body:
            return ""
        return f"Table: {caption}\n\n{table_body}\n\n{footnote}"

class DocumentProcessor:

    def __init__(
        self,
        max_chunk_size: int = 1000,
        overlap: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.table_processor = TableProcessor()
        self.section_stack = []
        self.overlap_buffer = []

    def load_content_list(
        self,
        json_path: str
    ) -> list[dict]:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def process_document(
        self,
        content_list: list[dict],
    ) -> list[Chunk]:
        chunks = []
        text_buffer = []
        last_page = -1

        for item in content_list:
            if item.get('type') == 'discarded':
                continue

            page_idx = item.get('page_idx', 0)
            bbox = item.get('bbox', [0,0,0,0])

            if item.get('type') == 'text':
                text = item.get('text', '').strip()
                text_level = item.get('text_level')

                if text_level is not None:
                    if text_buffer:
                        chunks.extend(self._flush_text_buffer(text_buffer, last_page))
                        text_buffer = []

                    self._update_section_stack(text, text_level)

                else:
                    text_buffer.append({
                        'text': text,
                        'bbox': bbox,
                        'page_idx': page_idx,
                    })
                    last_page = page_idx

            elif item.get('type') == 'table':
                if text_buffer:
                    chunks.extend(self._flush_text_buffer(text_buffer, last_page))
                    text_buffer = []

                table_chunk = self._process_table(item, page_idx, bbox)
                if table_chunk:
                    chunks.append(table_chunk)

        if text_buffer:
            chunks.extend(self._flush_text_buffer(text_buffer, last_page))

        return chunks

    def _update_section_stack(self, title: str, level: int):
        cleaned_title = title.strip()

        while len(self.section_stack) >= level:
            if self.section_stack:
                self.section_stack.pop()

        self.section_stack.append(cleaned_title)

    def _process_table(self, item: dict, page_idx: int, bbox: list[int]) -> Chunk | None:

        table_body = item.get('table_body', '')

        if not table_body:
            return None

        caption = ' '.join(item.get('table_caption', []))
        footnote = ' '.join(item.get('table_footnote', []))

        linearized_table = self.table_processor.linearize_table(table_body, caption, footnote)

        if not linearized_table:
            return None

        section_context = ' > '.join(self.section_stack) if self.section_stack else ""
        enhanced_content = f"Section: {section_context}\n\n{linearized_table}" if section_context else linearized_table

        return Chunk(
            content=enhanced_content,
            chunk_type='table',
            page_idx=page_idx,
            bbox=bbox,
            section_path=self.section_stack.copy(),
            metadata={
                'caption': caption,
                'footnote': footnote,
                'original_html': table_body,
            },
        )

    def _flush_text_buffer(
        self,
        buffer: list[dict],
        page_idx: int,
    ) -> list[Chunk]:

        chunks = []

        text = ""
        for item in buffer:
            item['text'] = item['text'].strip()
            if not item['text']:
                continue
            if len(text) + len(item['text']) > self.max_chunk_size and text:
                section = ' > '.join(self.section_stack) if self.section_stack else ""
                enhanced_content = f"Section: {section}\n\n{text}" if section else text
                chunks.append(Chunk(
                    content=enhanced_content,
                    chunk_type='text',
                    page_idx=page_idx,
                    bbox=item['bbox'],
                    section_path=self.section_stack.copy(),
                ))
                text = ""
            else:
                text += f"{item['text']}\n\n"

        if text:
            section = ' > '.join(self.section_stack) if self.section_stack else ""
            enhanced_content = f"Section: {section}\n\n{text}" if section else text
            chunks.append(Chunk(
                content=enhanced_content,
                chunk_type='text',
                page_idx=page_idx,
                bbox=item['bbox'],
                section_path=self.section_stack.copy(),
            ))

        return chunks

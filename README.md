# EnergyPlus-RAG

An intelligent Q&A system for EnergyPlus documentation powered by RAG (Retrieval-Augmented Generation) technology.

## Features

- **PDF Document Parsing**: Automatically parse EnergyPlus PDF documentation using MinerU
- **Intelligent Chunking**: Smart document chunking with section awareness and overlap handling
- **Vector Storage**: Store document embeddings in Qdrant vector database for efficient retrieval
- **Semantic Search**: Retrieve relevant document sections using Gemini embedding model
- **AI-Powered Q&A**: Answer questions using Qwen3-30B model with chain-of-thought reasoning
- **Web Interface**: User-friendly Gradio-based web UI with real-time streaming responses
- **Document Upload**: Upload and process new PDF documents directly through the web interface

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| Web UI | Gradio |
| Vector Database | Qdrant |
| Embedding Model | Gemini Embedding API |
| LLM | Qwen3-30B (via Ollama) |
| PDF Parsing | MinerU |

## Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai/) installed and running
- [MinerU](https://github.com/opendatalab/MinerU) installed
- Qdrant Cloud account or self-hosted Qdrant instance
- Google Gemini API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/energyplus-rag.git
cd energyplus-rag
```

2. Install dependencies using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

3. Install Ollama and pull the Qwen3 model:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# The model will be automatically pulled on first run
```

4. Install MinerU:

```bash
pip install mineru
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_ENDPOINT=https://your-qdrant-instance.qdrant.io
QDRANT_COLLECTION_NAME=energyplus_docs
MINERU_VRAM=30
```

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for embeddings |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_ENDPOINT` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | Name of the vector collection |
| `MINERU_VRAM` | GPU VRAM allocation for MinerU (default: 30GB) |

## Usage

### Start the Application

```bash
python app.py --port 8000
```

The web interface will be available at `http://localhost:8000`.

### Web Interface

1. **Chat**: Enter your EnergyPlus-related questions in the query input box
2. **RAG Toggle**: Enable/disable RAG retrieval using the checkbox
3. **Document Upload**: Upload PDF files to expand the knowledge base
4. **Thinking Process**: View the model's reasoning and retrieved documents in the sidebar

### Example Questions

- What are the main parameters of the Zone object in EnergyPlus?
- How do I set the thermal conductivity of building materials?
- What control strategies are available for HVAC systems?
- How do I define the exterior wall construction?
- How should the Timestep parameter be configured?

## Project Structure

```
energyplus-rag/
├── app.py              # Main entry point
├── pyproject.toml      # Project configuration and dependencies
├── src/
│   ├── chunk.py        # Document chunking with section awareness
│   ├── embedding.py    # Gemini embedding model integration
│   ├── model.py        # Ollama/Qwen3 model management
│   ├── parse.py        # PDF parsing using MinerU
│   ├── rag.py          # RAG system orchestration
│   ├── retrieval.py    # Retrieval utilities
│   ├── ui.py           # Gradio web interface
│   └── vector.py       # Qdrant vector store operations
├── data/               # Input data directory
├── output/             # Parsed document output
└── .env                # Environment variables (not tracked)
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF Doc   │────▶│   MinerU    │────▶│   Chunker   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │◀───▶│  Gradio UI  │◀───▶│  RAG System │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
             │   Qdrant    │           │   Gemini    │           │   Qwen3     │
             │ (Vectors)   │           │ (Embedding) │           │   (LLM)     │
             └─────────────┘           └─────────────┘           └─────────────┘
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [EnergyPlus](https://energyplus.net/) - Building energy simulation program
- [MinerU](https://github.com/opendatalab/MinerU) - PDF parsing tool
- [Qdrant](https://qdrant.tech/) - Vector search engine
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Gradio](https://gradio.app/) - Web UI framework

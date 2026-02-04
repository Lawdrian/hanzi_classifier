# Hanzi Classifier + Chatbot

## Project Overview
This repo contains two related codebases:
- **Classifier**: A PyTorch-based Hanzi character classifier with training and inference utilities, plus an MCP server for serving predictions.
- **Chatbot**: A Streamlit UI that orchestrates tools and calls the classifier MCP server for Hanzi classification.

## Startup

### Classifier (ML + MCP server)
```sh
# 1) Enter the codebase
cd classifier

# 2) Create venv
uv venv .venv

# 3) Sync dependencies
uv sync

# (Optional) CUDA 11.8 wheels
# uv sync --index-url https://download.pytorch.org/whl/cu118

# 4) Run training (optional)
uv run python -m src.train

# 5) Run MCP server
uv run python mcp_server.py
```

**API keys:** None required for the classifier.

### Chatbot (Streamlit UI)
```sh
# 1) Enter the codebase
cd chatbot

# 2) Create venv
uv venv .venv

# 3) Sync dependencies
uv sync

# 4) Add .env with required variables
# GROQ_API_KEY=your_groq_key
# LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
# LANGFUSE_SECRET_KEY=your_langfuse_secret_key
# LANGFUSE_HOST=https://cloud.langfuse.com

# 5) Run the app
uv run streamlit run main.py
```
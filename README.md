# BIS Standards Recommendation Engine - RAG Pipeline

A production-quality Retrieval-Augmented Generation (RAG) system built to recommend Bureau of Indian Standards (BIS) for Building Materials based on product descriptions. 

## Features
- **Document Ingestion**: Parses BIS SP 21 PDFs and chunks text using a sliding window strategy.
- **High-Speed Retrieval**: Uses `sentence-transformers` (BAAI/bge-small-en-v1.5) and `FAISS` for ultra-fast, semantic top-k retrieval.
- **LLM Grounding**: Integrates with Google Gemini API (gemini-1.5-flash) to strictly extract hallucination-free standards from the retrieved context. (Includes a blazing-fast mock fallback mode if no API key is provided).
- **Latency Optimized**: Achieves < 5s query latency (typically < 1s).
- **Evaluation Ready**: Compatible with standard Hackathon `eval_script.py`.

## Directory Structure
```
/src
  ingest.py        # Parses PDF into chunks
  index.py         # Embeds chunks and builds FAISS index
  retriever.py     # Semantic search logic
  llm_generator.py # Prompting & JSON formatting via LLM
/data
  /raw             # Put dataset.pdf here
  /processed       # Contains extracted chunks.json
  /index           # Contains faiss.index
inference.py       # Main pipeline script
eval_script.py     # Evaluation script
```

## Setup & Installation

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your Google Gemini API key (optional but recommended for >80% Hit Rate):
```env
GEMINI_API_KEY=your_api_key_here
```
*(If no API key is provided, the system falls back to a Regex-based Mock Mode for testing).*

## Usage

### 1. Ingestion & Indexing (Data Prep)
Place your `dataset.pdf` in the root directory (or `data/raw/` if updated in scripts) and run:
```bash
python src/ingest.py
python src/index.py
```
This builds the `faiss.index`.

### 2. Run Inference
To process your queries and generate the output JSON:
```bash
python inference.py --input sample_output.json --output test_output.json
```

### 3. Evaluate
Run the official evaluation script to see Hit Rate @3, MRR @5, and Latency:
```bash
python eval_script.py --results test_output.json
```

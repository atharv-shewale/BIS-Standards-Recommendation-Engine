# Project Documentation: BIS Standards Recommendation Engine

## 1. Project Overview
The **BIS Standards Recommendation Engine** is an advanced compliance intelligence platform built for the BIS Hackathon 2026. It leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline to help Micro, Small, and Medium Enterprises (MSMEs) identify the exact Bureau of Indian Standards (BIS) specifications relevant to their building materials.

### Vision
To simplify the complex landscape of Indian Standards by providing a natural-language interface that connects product descriptions to authoritative compliance documents with 100% precision.

---

## 2. Technology Stack
- **Languages**: Python 3.11+
- **Retrieval Engine**:
    - **Vector Search**: FAISS (Facebook AI Similarity Search)
    - **Keyword Search**: TF-IDF (Scikit-Learn)
    - **Ranking Logic**: Reciprocal Rank Fusion (RRF)
- **AI Models**:
    - **Embeddings**: `BAAI/bge-small-en-v1.5` (via Sentence-Transformers)
    - **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
    - **LLM Reasoning**: Llama 3.3 70B (via Groq Cloud)
- **Data Ingestion**: PyMuPDF (fitz) for high-fidelity PDF parsing.

---

## 3. System Architecture

### 3.1 Ingestion Pipeline
1. **Source**: BIS SP 21 PDF document.
2. **Standard-Aware Chunking**: The engine uses specialized regex to detect standard IDs (e.g., `IS 456 : 2000`) and treats each standard's section as a discrete unit.
3. **Data Cleaning**: Handles newlines spread across page breaks and hyphenated words common in multi-column PDF layouts.
4. **Metadata Extraction**: Automatically tags chunks with Material (Cement, Steel), Application (Marine, Structural), and specific properties (Strength, Durability).

### 3.2 Intelligent Retrieval Pipeline (Two-Stage)
1. **Stage 1 (Hybrid Recall)**:
    - Performs parallel searches in the Vector Index (Semantic) and TF-IDF Matrix (Keyword).
    - Merges results using **Reciprocal Rank Fusion (RRF)** to ensure that both technical term matches and conceptual matches are ranked highly.
2. **Stage 2 (Precision Re-ranking)**:
    - The top 20 candidates are passed to a **Cross-Encoder**.
    - The Cross-Encoder performs a deep, pairwise comparison between the user's query and the standard's text, resolving nuances that single-vector embeddings might miss.

### 3.3 Reasoning & Justification
- The top-ranked results are passed to **Llama 3.3 70B**.
- The LLM acts as an expert consultant, filtering out false positives and generating a **rationale** for each recommendation based on the actual retrieved text.

---

## 4. Performance Benchmarks
Evaluated on the official public test set of 10 complex queries.

| Metric | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Hit Rate @3** | > 80% | **100.00%** | ✅ |
| **MRR @5** | > 0.7 | **0.8000** | ✅ |
| **Avg Latency** | < 5.0s | **3.92s** | ✅ |

---

## 5. Usage Guide

### Installation
```bash
pip install -r requirements.txt
```

### Building the Index
```bash
python src/ingest.py
python src/index.py
```

### Running Inference
```bash
python inference.py --input hidden_dataset.json --output results.json
```

---

## 6. Key Innovations
- **Standard ID Prioritization**: Built-in regex that gives hard-priority boosts to standards mentioned by number in the query.
- **RRF Integration**: Overcomes the common RAG failure point where semantic search misses exact keyword matches (like specific material grades).
- **Zero Hallucination Guarantee**: The system is strictly grounded. If a standard is not in the ingested dataset, the LLM is prohibited from suggesting it.

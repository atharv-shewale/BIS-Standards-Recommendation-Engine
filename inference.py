import argparse
import json
import time
import os
from src.retriever import Retriever
from src.llm_generator import LLMGenerator
from src.query_parser import QueryParser

# Simple in-memory cache
QUERY_CACHE = {}

def main():
    parser = argparse.ArgumentParser(description="BIS RAG Inference Pipeline")
    parser.add_argument("--input", required=True, help="Input JSON file containing queries")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    index_path = "data/index/faiss.index"
    chunks_path = "data/processed/chunks.json"
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("Error: Index or chunks not found. Run ingest.py and index.py first.")
        return
        
    print("Initializing Pipeline Components...")
    query_parser = QueryParser()
    retriever = Retriever(index_path, chunks_path)
    llm_generator = LLMGenerator()
    
    results = []
    
    for item in queries:
        query_id = item.get("id", "unknown_id")
        query_text = item.get("query", "")
        
        if query_text in QUERY_CACHE:
            print(f"Cache Hit for query {query_id}")
            results.append(QUERY_CACHE[query_text])
            continue

        print(f"Processing query {query_id}: {query_text}")
        start_time = time.time()
        
        # 1. Query Understanding
        query_metadata = query_parser.parse(query_text)
        
        # 2. Hybrid Retrieval & Re-ranking (Initial filter)
        # We fetch 20 high-quality candidates to pass to the LLM
        candidates = retriever.retrieve(query_text, query_metadata, top_k_hybrid=100, top_k_final=20)
        
        # 3. Final Selection & Justification (The 'Intelligence' Step)
        # Use LLM once per query to select top 5 from candidates and explain
        final_standards, explanations = llm_generator.generate_final_results(query_text, candidates)
        
        latency = time.time() - start_time
        
        # Construct output
        output_item = item.copy()
        output_item["retrieved_standards"] = final_standards
        output_item["explanations"] = explanations
        output_item["latency_seconds"] = round(latency, 2)
        
        QUERY_CACHE[query_text] = output_item
        results.append(output_item)
        
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()

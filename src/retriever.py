import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

import pickle
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, index_path, chunks_path, model_name='BAAI/bge-small-en-v1.5', reranker_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        print(f"Loading TF-IDF index...")
        tfidf_path = index_path.replace("faiss.index", "tfidf.pkl")
        with open(tfidf_path, "rb") as f:
            data = pickle.load(f)
            self.tfidf_vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["matrix"]
            
        print(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print(f"Loading cross-encoder {reranker_name}...")
        self.reranker = CrossEncoder(reranker_name)

    def retrieve(self, query, query_metadata, top_k_hybrid=50, top_k_final=5):
        # 1. Semantic Search
        query_embedding = self.model.encode([query])
        # FAISS IndexFlatL2 uses Euclidean distance
        sem_distances, sem_indices = self.index.search(np.array(query_embedding).astype('float32'), top_k_hybrid)
        
        # 2. Keyword Search (TF-IDF)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        tfidf_indices = np.argsort(tfidf_similarities)[-top_k_hybrid:][::-1]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # Combine ranks from semantic and keyword search
        rrf_scores = {}
        k = 60 # Default RRF constant
        
        # Semantic Ranks
        for rank, idx in enumerate(sem_indices[0]):
            if idx == -1: continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (k + rank + 1))
            
        # Keyword Ranks
        for rank, idx in enumerate(tfidf_indices):
            if tfidf_similarities[idx] <= 0: continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (k + rank + 1))
            
        # 4. Standard ID Boost (Hard Priority)
        # If the query mentions a specific standard, prioritize chunks from that standard
        if "standard_ids" in query_metadata:
            for idx in rrf_scores:
                chunk_std = self.chunks[idx]["metadata"]["standard_id"].replace(" ", "").upper()
                for q_std in query_metadata["standard_ids"]:
                    q_std_clean = q_std.replace(" ", "").upper()
                    if q_std_clean in chunk_std:
                        rrf_scores[idx] += 1.0 # Significant boost
        
        # Get top candidates for re-ranking
        candidate_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:20]
        
        if not candidate_indices:
            return []
            
        # 5. Cross-Encoder Re-ranking
        # Pass (query, text) pairs to the cross-encoder
        pairs = [[query, self.chunks[idx]["text"]] for idx in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine RRF and Re-ranker (Optional, or just use re-ranker)
        # We'll use the re-ranker scores as primary
        final_candidates = []
        for i, idx in enumerate(candidate_indices):
            final_candidates.append({
                "id": self.chunks[idx]["id"],
                "text": self.chunks[idx]["text"],
                "standard_id": self.chunks[idx]["metadata"]["standard_id"],
                "score": float(rerank_scores[i])
            })
            
        # Sort by cross-encoder score
        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # De-duplicate by standard_id
        unique_candidates = []
        seen_ids = set()
        for c in final_candidates:
            norm_id = c["standard_id"].replace(" ", "").lower()
            if norm_id not in seen_ids:
                unique_candidates.append(c)
                seen_ids.add(norm_id)
        
        return unique_candidates[:top_k_final]

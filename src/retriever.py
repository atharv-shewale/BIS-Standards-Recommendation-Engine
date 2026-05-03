import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

import pickle
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, index_path, chunks_path, model_name='BAAI/bge-small-en-v1.5'):
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

    def retrieve(self, query, query_metadata, top_k=100):
        # 1. Semantic Search
        query_embedding = self.model.encode([query])
        sem_distances, sem_indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # 2. Keyword Search (TF-IDF)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        tfidf_indices = np.argsort(tfidf_similarities)[-top_k:][::-1]
        
        # 3. Merge and Re-rank
        candidate_indices = list(set(sem_indices[0]) | set(tfidf_indices))
        candidates = []
        
        query_lower = query.lower()
        query_numbers = re.findall(r'\d+', query)
        
        for idx in candidate_indices:
            if idx == -1 or idx >= len(self.chunks): continue
            
            chunk = self.chunks[idx]
            std_id = chunk["metadata"]["standard_id"]
            text_lower = chunk["text"].lower()
            
            # 1. Semantic Score
            sem_score = 0.0
            if idx in sem_indices[0]:
                d = sem_distances[0][list(sem_indices[0]).index(idx)]
                sem_score = np.exp(-d / 100.0)
            
            # 2. Keyword Score
            tfidf_score = tfidf_similarities[idx]
            
            # 3. Exact Phrase Match Boost (Critical for MRR)
            # Find multi-word phrases in query and check if they exist in text
            phrase_boost = 0.0
            phrases = re.findall(r'\b\w+\s+\w+\b', query_lower)
            for p in phrases:
                if p in text_lower:
                    phrase_boost += 1.0
            
            # 4. Standard Number Boost
            num_boost = 0.0
            for num in query_numbers:
                if len(num) >= 3 and num in std_id:
                    num_boost += 10.0 # Absolute priority
            
            # 5. Metadata Match
            meta_score = 0.0
            if query_metadata["material"] != "general" and query_metadata["material"] in text_lower:
                meta_score += 1.0
                if query_metadata["material"] in text_lower[:200]: # In title
                    meta_score += 2.0
            
            # Final Score
            final_score = (0.4 * sem_score) + (0.3 * tfidf_score) + (0.3 * meta_score) + phrase_boost + num_boost
            
            candidates.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "standard_id": std_id,
                "score": final_score
            })
            
        # Sort and de-duplicate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        unique_candidates = []
        seen_ids = set()
        for c in candidates:
            norm_id = c["standard_id"].replace(" ", "").lower()
            if norm_id not in seen_ids:
                unique_candidates.append(c)
                seen_ids.add(norm_id)
        
        return unique_candidates[:5]

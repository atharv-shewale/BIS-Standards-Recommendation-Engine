import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def build_index(chunks_path, index_path, model_name='BAAI/bge-small-en-v1.5'):
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    texts = [chunk['text'] for chunk in chunks]
    
    # 1. FAISS Semantic Index
    print(f"Loading embedding model {model_name}...")
    model = SentenceTransformer(model_name)
    print("Generating semantic embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    # 2. TF-IDF Keyword Index
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # Save everything
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    
    tfidf_path = index_path.replace("faiss.index", "tfidf.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump({"vectorizer": tfidf, "matrix": tfidf_matrix}, f)
        
    print(f"Indices saved: FAISS ({index.ntotal} vectors), TF-IDF ({tfidf_matrix.shape[0]} docs)")

if __name__ == "__main__":
    build_index("data/processed/chunks.json", "data/index/faiss.index")

from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module import (small model, CPU)
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def cosine_similarity(text1: str, text2: str) -> float:
    """Return cosine similarity between two texts (0..1)."""
    model = get_embedding_model()
    emb1 = model.encode(text1, convert_to_numpy=True)
    emb2 = model.encode(text2, convert_to_numpy=True)
    norm1 = emb1 / np.linalg.norm(emb1)
    norm2 = emb2 / np.linalg.norm(emb2)
    sim = np.dot(norm1, norm2)
    return float(max(0.0, min(1.0, sim)))  # clamp to [0,1]
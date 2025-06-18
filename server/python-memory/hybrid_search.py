"""
Hybrid Search

Lexical Search: Finds Exact keyword matches
Semantic Search: Finds similar meaning matches
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import logging

# Download required nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between vectors a and b.
    If a is 2D and b is 2D, returns a matrix of similarities.
    """
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Handle 1D vectors
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    # Calculate dot product
    dot_product = np.dot(a, b.T)
    
    # Calculate norms
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarities = dot_product / (norm_a @ norm_b.T)
    
    return similarities

# Implements hybrid search
class HybridSearch:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        logger.info(f"HybridSearch initialized with language: {language}")

    def preprocess_text(self, text: str, use_stemming: bool = True) -> List[str]:
        """
        Preprocess text by tokenizing, removing stopwords, and optionally stemming.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs 
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9][$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove special char but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Apply stemming if required
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def lexical_search(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform BM25 lexical search
    """
        if not documents:
            return []
        
        # Exract and preprocess document texts
        doc_texts = []
        valid_indices = []

        for i, doc in enumerate(documents):
            content = doc.get('content', '') or doc.get('memory', '')
            if isinstance(content, dict):
                content = content.get('content', '')

            if content:
                doc_texts.append(self.preprocess_text(str(content)))
                valid_indices.append(i)
        
        if not doc_texts:
            return []
        
        # Create BM25 index
        bm25 = BM25Okapi(doc_texts)

        # preprocess query
        query_tokens = self.preprocess_text(query)

        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)

        # Get tok-k results
        top_indices = np.argsort(scores)[::-1][::top_k]

        # Return results with original document indices 
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only include non-zero scores
                original_idx = valid_indices[idx]
                results.append((original_idx, float(scores[idx])))
        return results 

    def semantic_search(self, query_embedding: np.ndarray, document_embeddings: List[np.ndarray], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform semantic search using cosine similarity"""
        if not document_embeddings:
            return []
        
        # Convert to numpy array
        if isinstance(document_embeddings, list):
            doc_matrix = np.vstack([emb for emb in document_embeddings if emb is not None])
        else:
            doc_matrix = document_embeddings
        
        # Reshape query embedding if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, doc_matrix)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results 
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((int(idx), float(similarities[idx])))
        
        return results

    def normalize_scores(self, scores: List[Tuple[int, float]])  -> Dict[int, float]:
        # Normalize scores to [0, 1] range

        if not scores:
            return {}

        max_score = max(score for _, score in scores)
        min_score = min(score for _, score in scores)

        if max_score == min_score:
            return {idx: 1.0 for idx, _ in scores}

        normalized = {}
        for idx, score in scores:
            normalized[idx] = (score - min_score) / (max_score - min_score)

        return normalized

    def hybrid_search(self, query: str, query_embedding: np.ndarray, documents: List[Dict[str, Any]], document_embeddings: List[np.ndarray], top_k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        # Perform hybrid search combining lexical and semantic search
        lexical_results = self.semantic_search(query_embedding, document_embeddings, top_k * 2)
        lexical_scores = self.normalize_scores(lexical_results)

        # Perform semantic search
        semantic_results = self.semantic_search(query_embedding, document_embeddings, top_k * 2)
        semantic_scores = self.normalize_scores(semantic_results)

        # Combine scores
        combined_scores = {}
        all_indices = set([idx for idx, _ in lexical_results] + [idx for idx, _ in semantic_results])

        for idx in all_indices:
            lex_score = lexical_scores.get(idx, 0)
            sem_score = semantic_scores.get(idx, 0)

            # weighted combination
            combined_score = (1 - alpha) * lex_score + alpha * sem_score
            combined_scores[idx] = {
                'combined_score': combined_score,
                'lexical_score': lex_score,
                'semantic_score': sem_score

            }

            # Sort by combined score
            sorted_indices = sorted(combined_scores.items(), key=lambda x: combined_scores[x]['combined_score'], reverse=True)

            # Prepare final results
            results = []
            for idx in sorted_indices[:top_k]:
                if idx < len(documents):
                    result = {
                        'document': documents[idx],
                        'scores': combined_scores[idx],
                        'rank': len(results) + 1
                    }
                    results.append(result)
            logger.info(f"Hybrid search completed: {len(results)} results returned")
            return results

# Global search engine instance
_search_engine = None

def get_search_engine() -> HybridSearch:
    """Get or create the search engine singleton instance"""
    global _search_engine 
    if _search_engine is None:
        _search_engine = HybridSearch()
    return _search_engine
    
    


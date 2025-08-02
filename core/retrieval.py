from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict
from .vector_store import VectorStore
# === ADD THESE IMPORTS AT TOP OF FILE ===
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict
from .vector_store import VectorStore



class HybridRetriever:
    def __init__(self, vector_store: VectorStore, alpha: float = 0.7):
        self.vector_store = vector_store
        self.alpha = alpha
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def build_bm25_index(self, documents: List[str], metadata: List[Dict]):
        """Build BM25 index for keyword search"""
        self.documents = documents
        self.doc_metadata = metadata

        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"Built BM25 index with {len(documents)} documents")

    def hybrid_search(self, query: str, k: int = 10) -> List[Dict]:
        """Combine semantic and keyword search"""
        semantic_results = self.vector_store.similarity_search(query, k=k * 2)

        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2]
            bm25_results = []

            for idx in bm25_indices:
                if idx < len(self.documents):
                    bm25_results.append({
                        'text': self.documents[idx],
                        'metadata': self.doc_metadata[idx],
                        'bm25_score': float(bm25_scores[idx])
                    })
        else:
            bm25_results = []

        combined_results = self._combine_results(semantic_results, bm25_results, query)
        return combined_results[:k]

    def _combine_results(self, semantic_results: List[Dict], bm25_results: List[Dict], query: str) -> List[Dict]:
        """Combine semantic and BM25 results with re-ranking"""
        result_map = {}

        for result in semantic_results:
            text = result['text']
            result_map[text] = {
                'text': text,
                'metadata': result['metadata'],
                'semantic_score': result.get('similarity_score', 0),
                'bm25_score': 0
            }

        for result in bm25_results:
            text = result['text']
            if text in result_map:
                result_map[text]['bm25_score'] = result['bm25_score']
            else:
                result_map[text] = {
                    'text': text,
                    'metadata': result['metadata'],
                    'semantic_score': 0,
                    'bm25_score': result['bm25_score']
                }

        results = list(result_map.values())
        for result in results:
            semantic_norm = result['semantic_score']
            bm25_norm = result['bm25_score'] / 10.0 if result['bm25_score'] > 0 else 0
            result['hybrid_score'] = self.alpha * semantic_norm + (1 - self.alpha) * bm25_norm

        if len(results) > 1:
            results = self._rerank_with_cross_encoder(query, results)

        return sorted(results, key=lambda x: x.get('final_score', x['hybrid_score']), reverse=True)

    def _rerank_with_cross_encoder(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results using cross-encoder"""
        if not results:
            return results

        pairs = [(query, result['text']) for result in results]
        ce_scores = self.cross_encoder.predict(pairs)

        for i, result in enumerate(results):
            result['cross_encoder_score'] = float(ce_scores[i])
            result['final_score'] = 0.7 * result['hybrid_score'] + 0.3 * result['cross_encoder_score']

        return results


class MultiVectorRetriever:
    """Advanced multi-vector retrieval combining dense, sparse, and hybrid methods - ENHANCEMENT 4"""

    def __init__(self, vector_store, dense_weight=0.5, bm25_weight=0.3, tfidf_weight=0.2):
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight

        # Dense retrieval (existing)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Sparse retrieval components
        self.bm25 = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.8,
            min_df=2
        )
        self.tfidf_matrix = None

        # Document storage
        self.documents = []
        self.doc_metadata = []

        print("✅ Multi-vector retriever initialized")

    def build_sparse_indices(self, documents: List[str], metadata: List[Dict]):
        """Build both BM25 and TF-IDF indices - ENHANCEMENT 4"""
        if not documents:
            print("❌ No documents provided for indexing")
            return

        self.documents = documents
        self.doc_metadata = metadata

        print(f"🔨 Building sparse indices for {len(documents)} documents...")

        # Build BM25 index
        try:
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print("✅ BM25 index built successfully")
        except Exception as e:
            print(f"❌ BM25 index building failed: {e}")

        # Build TF-IDF index
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            print(f"✅ TF-IDF index built: {self.tfidf_matrix.shape}")
        except Exception as e:
            print(f"❌ TF-IDF index building failed: {e}")
            self.tfidf_matrix = None

    def dense_retrieval(self, query: str, k: int = 10) -> List[Dict]:
        """Dense vector similarity search - ENHANCEMENT 4"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            for result in results:
                result['retrieval_method'] = 'dense'
            return results
        except Exception as e:
            print(f"❌ Dense retrieval failed: {e}")
            return []

    def bm25_retrieval(self, query: str, k: int = 10) -> List[Dict]:
        """BM25 sparse retrieval - ENHANCEMENT 4"""
        if not self.bm25 or not self.documents:
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)

            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:k]

            results = []
            for idx in top_indices:
                if idx < len(self.documents) and scores[idx] > 0:
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.doc_metadata[idx],
                        'similarity_score': float(scores[idx]),
                        'retrieval_method': 'bm25'
                    })

            return results
        except Exception as e:
            print(f"❌ BM25 retrieval failed: {e}")
            return []

    def tfidf_retrieval(self, query: str, k: int = 10) -> List[Dict]:
        """TF-IDF sparse retrieval - ENHANCEMENT 4"""
        if self.tfidf_matrix is None or not self.documents:
            return []

        try:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])

            # Compute cosine similarities
            similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()

            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.doc_metadata[idx],
                        'similarity_score': float(similarities[idx]),
                        'retrieval_method': 'tfidf'
                    })

            return results
        except Exception as e:
            print(f"❌ TF-IDF retrieval failed: {e}")
            return []

    def multi_vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """Combined multi-vector retrieval - ENHANCEMENT 4"""
        print(f"🔍 Multi-vector search for: {query}")

        # Get results from all methods
        dense_results = self.dense_retrieval(query, k=k * 2)
        bm25_results = self.bm25_retrieval(query, k=k * 2)
        tfidf_results = self.tfidf_retrieval(query, k=k * 2)

        print(f"📊 Retrieved: Dense={len(dense_results)}, BM25={len(bm25_results)}, TF-IDF={len(tfidf_results)}")

        # Combine and normalize scores
        combined_results = self._combine_and_score(dense_results, bm25_results, tfidf_results)

        # Re-rank with cross-encoder if available
        if len(combined_results) > 1:
            combined_results = self._cross_encoder_rerank(query, combined_results)

        return combined_results[:k]

    def _combine_and_score(self, dense_results: List[Dict], bm25_results: List[Dict],
                           tfidf_results: List[Dict]) -> List[Dict]:
        """Combine results from different retrieval methods - ENHANCEMENT 4"""
        result_map = {}

        # Normalize scores for each method
        max_dense = max([r.get('similarity_score', 0) for r in dense_results]) if dense_results else 1
        max_bm25 = max([r.get('similarity_score', 0) for r in bm25_results]) if bm25_results else 1
        max_tfidf = max([r.get('similarity_score', 0) for r in tfidf_results]) if tfidf_results else 1

        # Process dense results
        for result in dense_results:
            text = result['text']
            normalized_score = result.get('similarity_score', 0) / max_dense
            result_map[text] = {
                'text': text,
                'metadata': result['metadata'],
                'dense_score': normalized_score,
                'bm25_score': 0.0,
                'tfidf_score': 0.0,
                'methods': ['dense']
            }

        # Process BM25 results
        for result in bm25_results:
            text = result['text']
            normalized_score = result.get('similarity_score', 0) / max_bm25

            if text in result_map:
                result_map[text]['bm25_score'] = normalized_score
                result_map[text]['methods'].append('bm25')
            else:
                result_map[text] = {
                    'text': text,
                    'metadata': result['metadata'],
                    'dense_score': 0.0,
                    'bm25_score': normalized_score,
                    'tfidf_score': 0.0,
                    'methods': ['bm25']
                }

        # Process TF-IDF results
        for result in tfidf_results:
            text = result['text']
            normalized_score = result.get('similarity_score', 0) / max_tfidf

            if text in result_map:
                result_map[text]['tfidf_score'] = normalized_score
                result_map[text]['methods'].append('tfidf')
            else:
                result_map[text] = {
                    'text': text,
                    'metadata': result['metadata'],
                    'dense_score': 0.0,
                    'bm25_score': 0.0,
                    'tfidf_score': normalized_score,
                    'methods': ['tfidf']
                }

        # Calculate combined scores
        combined_results = []
        for item in result_map.values():
            # Weighted combination
            combined_score = (
                    self.dense_weight * item['dense_score'] +
                    self.bm25_weight * item['bm25_score'] +
                    self.tfidf_weight * item['tfidf_score']
            )

            # Bonus for appearing in multiple methods
            method_bonus = len(item['methods']) * 0.1
            final_score = combined_score + method_bonus

            combined_results.append({
                'text': item['text'],
                'metadata': item['metadata'],
                'similarity_score': final_score,
                'dense_score': item['dense_score'],
                'bm25_score': item['bm25_score'],
                'tfidf_score': item['tfidf_score'],
                'methods_used': item['methods'],
                'retrieval_method': 'multi_vector'
            })

        # Sort by final score
        combined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return combined_results

    def _cross_encoder_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results using cross-encoder - ENHANCEMENT 4"""
        try:
            if not results:
                return results

            # Prepare pairs for cross-encoder
            pairs = [(query, result['text']) for result in results]

            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)

            # Update results with cross-encoder scores
            for i, result in enumerate(results):
                result['cross_encoder_score'] = float(ce_scores[i])
                # Combine with original score
                result['final_score'] = (
                        0.7 * result['similarity_score'] +
                        0.3 * result['cross_encoder_score']
                )

            # Sort by final score
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results

        except Exception as e:
            print(f"❌ Cross-encoder re-ranking failed: {e}")
            return results

    def get_retrieval_stats(self) -> Dict:
        """Get multi-vector retrieval statistics - ENHANCEMENT 4"""
        return {
            'total_documents': len(self.documents),
            'bm25_available': self.bm25 is not None,
            'tfidf_available': self.tfidf_matrix is not None,
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'weights': {
                'dense': self.dense_weight,
                'bm25': self.bm25_weight,
                'tfidf': self.tfidf_weight
            }
        }


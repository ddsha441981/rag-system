from typing import List, Dict, Tuple
import re
import nltk
from collections import Counter
import numpy as np


try:
    from transformers import pipeline, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available. Install with: pip install transformers")


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


class ContextualCompressor:
    """Free contextual compression using extractive and abstractive methods - ENHANCEMENT 6"""

    def __init__(self):

        if TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading BART summarization model...")
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # Use CPU (free)
                )
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                print("✅ BART model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading BART model: {e}")
                self.summarizer = None
                self.tokenizer = None
        else:
            self.summarizer = None
            self.tokenizer = None

    def compress_context(self, context_chunks: List[Dict], query: str,
                         max_tokens: int = 2000, compression_method: str = "auto") -> List[Dict]:
        if not context_chunks:
            return context_chunks

        print(f"🗜️ Compressing {len(context_chunks)} chunks (target: {max_tokens} tokens)")

        current_tokens = self._estimate_token_count(context_chunks)
        print(f"📊 Current tokens: {current_tokens}")

        if current_tokens <= max_tokens:
            print("✅ Context already within token limit")
            return context_chunks

        if compression_method == "auto":
            if self.summarizer and current_tokens > max_tokens * 2:
                method = "abstractive"
            else:
                method = "extractive"
        else:
            method = compression_method

        print(f"🎯 Using {method} compression method")

        if method == "abstractive" and self.summarizer:
            return self._abstractive_compression(context_chunks, query, max_tokens)
        else:
            return self._extractive_compression(context_chunks, query, max_tokens)

    def _estimate_token_count(self, chunks: List[Dict]) -> int:
        total_text = ' '.join([chunk.get('text', '') for chunk in chunks])

        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(total_text))
            except:
                pass

        return len(total_text) // 4

    def _abstractive_compression(self, chunks: List[Dict], query: str, max_tokens: int) -> List[Dict]:
        print("🤖 Performing abstractive compression...")

        compressed_chunks = []

        for chunk in chunks:
            text = chunk.get('text', '')
            if len(text.split()) > 100:
                try:

                    current_words = len(text.split())
                    target_words = min(current_words // 2, 150)

                    summary = self.summarizer(
                        text,
                        max_length=target_words,
                        min_length=max(target_words // 2, 30),
                        do_sample=False
                    )

                    compressed_chunk = chunk.copy()
                    compressed_chunk['text'] = summary[0]['summary_text']
                    compressed_chunk['compressed'] = True
                    compressed_chunk['compression_method'] = 'abstractive'
                    compressed_chunks.append(compressed_chunk)

                except Exception as e:
                    print(f"⚠️ Abstractive compression failed for chunk: {e}")
                    compressed_chunks.append(chunk)
            else:
                compressed_chunks.append(chunk)

        return compressed_chunks

    def _extractive_compression(self, chunks: List[Dict], query: str, max_tokens: int) -> List[Dict]:

        print("✂️ Performing extractive compression...")


        scored_chunks = []
        for chunk in chunks:
            relevance_score = self._calculate_relevance_score(chunk.get('text', ''), query)
            scored_chunks.append((chunk, relevance_score))


        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        selected_chunks = []
        current_tokens = 0

        for chunk, score in scored_chunks:
            chunk_tokens = self._estimate_token_count([chunk])

            if current_tokens + chunk_tokens <= max_tokens:

                if chunk_tokens > 300:
                    compressed_chunk = self._compress_chunk_sentences(chunk, query, 200)
                    selected_chunks.append(compressed_chunk)
                    current_tokens += self._estimate_token_count([compressed_chunk])
                else:
                    selected_chunks.append(chunk)
                    current_tokens += chunk_tokens
            else:
                break

        print(f"✅ Compressed to {len(selected_chunks)} chunks ({current_tokens} tokens)")
        return selected_chunks

    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """Calculate relevance score between text and query - ENHANCEMENT 6"""

        text_lower = text.lower()
        query_lower = query.lower()

        # Get words
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        if not query_words or not text_words:
            return 0.0


        intersection = text_words.intersection(query_words)
        union = text_words.union(query_words)
        jaccard = len(intersection) / len(union) if union else 0.0


        phrase_bonus = 0.2 if query_lower in text_lower else 0.0


        length_penalty = 0.0 if len(text.split()) < 10 else 0.0

        return min(jaccard + phrase_bonus - length_penalty, 1.0)

    def _compress_chunk_sentences(self, chunk: Dict, query: str, max_tokens: int) -> Dict:
        """Compress chunk by selecting key sentences - ENHANCEMENT 6"""
        text = chunk.get('text', '')
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= 2:
            return chunk


        sentence_scores = []
        for sentence in sentences:
            score = self._calculate_relevance_score(sentence, query)
            sentence_scores.append((sentence, score))


        sentence_scores.sort(key=lambda x: x[1], reverse=True)


        selected_sentences = []
        current_tokens = 0

        for sentence, score in sentence_scores:
            sentence_tokens = len(sentence.split()) * 1.3  # Rough token estimation

            if current_tokens + sentence_tokens <= max_tokens:
                selected_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break

        compressed_chunk = chunk.copy()
        compressed_chunk['text'] = ' '.join(selected_sentences)
        compressed_chunk['compressed'] = True
        compressed_chunk['compression_method'] = 'extractive'
        compressed_chunk['original_sentences'] = len(sentences)
        compressed_chunk['compressed_sentences'] = len(selected_sentences)

        return compressed_chunk

    def get_compression_stats(self) -> Dict:
        """Get compression capabilities - ENHANCEMENT 6"""
        return {
            'abstractive_available': self.summarizer is not None,
            'extractive_available': True,
            'supported_methods': ['extractive'] + (['abstractive'] if self.summarizer else [])
        }

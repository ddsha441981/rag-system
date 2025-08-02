from typing import List, Dict, Optional
import nltk
from collections import Counter
import numpy as np
import re

# Try importing transformers for advanced summarization
try:
    from transformers import pipeline, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available for abstractive summarization")

# Ensure NLTK data
required_nltk_data = ['punkt', 'stopwords']
for data in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
    except LookupError:
        print(f"📥 Downloading NLTK {data}...")
        nltk.download(data)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class DocumentSummarizer:
    """Multi-method document summarization using free models - ENHANCEMENT 9"""

    def __init__(self):
        # Initialize free summarization models
        if TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading summarization models...")
                self.bart_summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # Use CPU
                )
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                print("✅ BART summarization model loaded")
            except Exception as e:
                print(f"❌ Error loading BART: {e}")
                self.bart_summarizer = None
                self.tokenizer = None
        else:
            self.bart_summarizer = None
            self.tokenizer = None

        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("⚠️ NLTK stopwords not available, using basic set")
            self.stop_words = set(
                ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

        print("✅ Document summarizer initialized")

    def summarize_document(self, text: str, summary_type: str = "auto",
                           max_length: int = 300, min_length: int = 50) -> Dict:
        """Generate document summary with multiple methods - ENHANCEMENT 9"""

        if len(text.split()) < 50:
            return {
                'summary': text,
                'summary_type': 'original',
                'compression_ratio': 1.0,
                'word_count': len(text.split()),
                'method': 'no_compression_needed'
            }

        print(f"📝 Summarizing document ({len(text.split())} words) -> target: {max_length} words")

        summaries = {}

        # Method 1: Extractive summarization (always available)
        extractive_summary = self._extractive_summarization(text, max_length)
        summaries['extractive'] = extractive_summary

        # Method 2: Abstractive summarization (if BART available)
        if self.bart_summarizer and len(text.split()) > 100:
            abstractive_summary = self._abstractive_summarization(text, max_length, min_length)
            summaries['abstractive'] = abstractive_summary

        # Method 3: Frequency-based summarization
        frequency_summary = self._frequency_based_summarization(text, max_length)
        summaries['frequency'] = frequency_summary

        # Choose best summary
        if summary_type == "auto":
            best_summary = self._choose_best_summary(summaries, text)
        else:
            best_summary = summaries.get(summary_type, summaries['extractive'])

        return {
            'summary': best_summary['text'],
            'summary_type': best_summary['method'],
            'compression_ratio': best_summary['compression_ratio'],
            'word_count': len(best_summary['text'].split()),
            'original_word_count': len(text.split()),
            'all_summaries': {k: v['text'] for k, v in summaries.items()}
        }

    def _abstractive_summarization(self, text: str, max_length: int, min_length: int) -> Dict:
        """Abstractive summarization using BART - ENHANCEMENT 9"""
        try:
            print("🤖 Generating abstractive summary...")

            # Handle long texts by chunking
            if len(text.split()) > 1000:
                chunks = self._split_text_for_summarization(text, 800)
                chunk_summaries = []

                for chunk in chunks:
                    chunk_summary = self.bart_summarizer(
                        chunk,
                        max_length=max_length // len(chunks) + 50,
                        min_length=min_length // len(chunks),
                        do_sample=False
                    )
                    chunk_summaries.append(chunk_summary[0]['summary_text'])

                # Combine chunk summaries
                combined_summary = ' '.join(chunk_summaries)

                # Summarize again if still too long
                if len(combined_summary.split()) > max_length:
                    final_summary = self.bart_summarizer(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    combined_summary = final_summary[0]['summary_text']
            else:
                summary = self.bart_summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                combined_summary = summary[0]['summary_text']

            return {
                'text': combined_summary,
                'method': 'abstractive',
                'compression_ratio': len(combined_summary.split()) / len(text.split()),
                'model': 'BART'
            }

        except Exception as e:
            print(f"❌ Abstractive summarization failed: {e}")
            return self._extractive_summarization(text, max_length)

    def _extractive_summarization(self, text: str, max_length: int) -> Dict:
        """Extractive summarization by sentence ranking - ENHANCEMENT 9"""
        print("✂️ Generating extractive summary...")

        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return {
                'text': text,
                'method': 'extractive',
                'compression_ratio': 1.0,
                'selected_sentences': len(sentences)
            }

        # Score sentences
        sentence_scores = self._score_sentences(sentences, text)

        # Select top sentences within word limit
        selected_sentences = []
        current_words = 0

        # Sort sentences by score and select greedily
        scored_sentences = list(zip(sentences, sentence_scores))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= max_length:
                selected_sentences.append((sentence, score))
                current_words += sentence_words

        # Sort selected sentences by original order
        original_order = {sentence: i for i, sentence in enumerate(sentences)}
        selected_sentences.sort(key=lambda x: original_order.get(x[0], 0))

        summary_text = ' '.join([sent[0] for sent in selected_sentences])

        return {
            'text': summary_text,
            'method': 'extractive',
            'compression_ratio': len(summary_text.split()) / len(text.split()),
            'selected_sentences': len(selected_sentences),
            'total_sentences': len(sentences)
        }

    def _frequency_based_summarization(self, text: str, max_length: int) -> Dict:
        """Frequency-based extractive summarization - ENHANCEMENT 9"""
        print("📊 Generating frequency-based summary...")

        sentences = sent_tokenize(text)

        # Calculate word frequencies
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        word_freq = Counter(words)

        # Score sentences based on word frequencies
        sentence_scores = []
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum() and word not in self.stop_words]

            if sentence_words:
                score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)
            else:
                score = 0

            sentence_scores.append(score)

        # Select sentences
        indexed_sentences = list(enumerate(zip(sentences, sentence_scores)))
        indexed_sentences.sort(key=lambda x: x[1][1], reverse=True)

        selected_indices = []
        current_words = 0

        for idx, (sentence, score) in indexed_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= max_length:
                selected_indices.append(idx)
                current_words += sentence_words

        # Sort by original order
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices]
        summary_text = ' '.join(summary_sentences)

        return {
            'text': summary_text,
            'method': 'frequency',
            'compression_ratio': len(summary_text.split()) / len(text.split()),
            'selected_sentences': len(summary_sentences)
        }

    def _score_sentences(self, sentences: List[str], full_text: str) -> List[float]:
        """Score sentences for extractive summarization - ENHANCEMENT 9"""
        # Calculate word frequencies
        words = word_tokenize(full_text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        word_freq = Counter(words)

        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0.0
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum() and word not in self.stop_words]

            if not sentence_words:
                sentence_scores.append(0)
                continue

            # Frequency score
            freq_score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)

            # Position score (first and last sentences often important)
            position_score = 0.0
            if i == 0 or i == len(sentences) - 1:
                position_score = 0.3
            elif i < len(sentences) * 0.2:  # First 20%
                position_score = 0.1

            # Length score (prefer moderate length)
            length_score = 0.0
            if 10 <= len(sentence_words) <= 30:
                length_score = 0.2

            # Numeric content (often contains facts)
            numeric_score = 0.1 if any(char.isdigit() for char in sentence) else 0.0

            total_score = freq_score + position_score + length_score + numeric_score
            sentence_scores.append(total_score)

        return sentence_scores

    def _split_text_for_summarization(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """Split text into chunks for processing - ENHANCEMENT 9"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len((current_chunk + " " + sentence).split()) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _choose_best_summary(self, summaries: Dict, original_text: str) -> Dict:
        """Choose best summary based on quality metrics - ENHANCEMENT 9"""
        best_summary = None
        best_score = 0

        for summary_type, summary_data in summaries.items():
            # Simple quality scoring
            compression_ratio = summary_data['compression_ratio']
            summary_length = len(summary_data['text'].split())

            # Prefer moderate compression and reasonable length
            compression_score = 1.0 - abs(0.3 - compression_ratio)  # Target 30% compression
            length_score = 1.0 if 30 <= summary_length <= 250 else 0.5

            # Bonus for abstractive if available
            method_bonus = 0.2 if summary_data['method'] == 'abstractive' else 0.0

            total_score = compression_score * 0.4 + length_score * 0.4 + method_bonus

            if total_score > best_score:
                best_score = total_score
                best_summary = summary_data

        return best_summary or summaries['extractive']

    def batch_summarize_documents(self, documents: List[Dict], **kwargs) -> List[Dict]:
        """Summarize multiple documents - ENHANCEMENT 9"""
        print(f"📚 Batch summarizing {len(documents)} documents...")

        summaries = []
        for i, doc in enumerate(documents):
            text = doc.get('text', '')
            if text:
                try:
                    summary_result = self.summarize_document(text, **kwargs)
                    summaries.append({
                        'document_index': i,
                        'original_metadata': doc.get('metadata', {}),
                        'summary': summary_result['summary'],
                        'summary_type': summary_result['summary_type'],
                        'compression_ratio': summary_result['compression_ratio'],
                        'original_words': summary_result['original_word_count'],
                        'summary_words': summary_result['word_count']
                    })
                except Exception as e:
                    print(f"⚠️ Error summarizing document {i}: {e}")
                    summaries.append({
                        'document_index': i,
                        'error': str(e),
                        'summary': text[:300] + "..." if len(text) > 300 else text
                    })

        print(f"✅ Completed batch summarization")
        return summaries

    def get_summarization_stats(self) -> Dict:
        """Get summarization capabilities - ENHANCEMENT 9"""
        return {
            'abstractive_available': self.bart_summarizer is not None,
            'extractive_available': True,
            'frequency_based_available': True,
            'supported_methods': ['extractive', 'frequency'] + (['abstractive'] if self.bart_summarizer else []),
            'max_input_length': 1024 if self.tokenizer else "unlimited"
        }
# === END NEW FILE ===

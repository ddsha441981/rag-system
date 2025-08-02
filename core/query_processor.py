
import nltk
from nltk.corpus import wordnet
import requests
from collections import defaultdict
# === END ENHANCEMENT 5 IMPORTS ===

import spacy
from typing import List, Dict
from .vector_store import VectorStore

# === ADD NLTK DATA DOWNLOAD ===
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("📥 Downloading WordNet for synonym expansion...")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
# === END NLTK SETUP ===


class QueryProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spacy English model: python -m spacy download en_core_web_sm")
            self.nlp = None


        self.synonym_cache = {}
        self.expansion_enabled = True
        print("✅ Query processor with expansion initialized")
        # === END ENHANCEMENT 5 ADDITIONS ===


    # def analyze_query(self, query: str) -> Dict:
    #     """Analyze query intent and extract key information"""
    #     if not self.nlp:
    #         return {
    #             'original_query': query,
    #             'entities': [],
    #             'keywords': query.split(),
    #             'query_type': 'general',
    #             'processed_query': query
    #         }
    #
    #     doc = self.nlp(query)
    #
    #     entities = [(ent.text, ent.label_) for ent in doc.ents]
    #     keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    #
    #     question_words = ['what', 'how', 'when', 'where', 'who', 'why', 'which']
    #     query_lower = query.lower()
    #
    #     query_type = "general"
    #     if any(qw in query_lower for qw in question_words):
    #         query_type = "question"
    #     elif "compare" in query_lower or "vs" in query_lower:
    #         query_type = "comparison"
    #     elif "summarize" in query_lower or "summary" in query_lower:
    #         query_type = "summary"
    #
    #     return {
    #         'original_query': query,
    #         'entities': entities,
    #         'keywords': keywords,
    #         'query_type': query_type,
    #         'processed_query': ' '.join(keywords)
    #     }

    def analyze_query(self, query: str) -> Dict:
        """Enhanced query analysis with better entity recognition"""
        if not self.nlp:
            return {
                'original_query': query,
                'entities': [],
                'keywords': query.split(),
                'query_type': 'general',
                'processed_query': query
            }

        doc = self.nlp(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Enhanced keyword extraction for technical terms
        keywords = []
        for token in doc:
            if not token.is_stop and not token.is_punct:
                keywords.append(token.lemma_.lower())

        # Add original words for technical terms that might not be lemmatized correctly
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        keywords.extend(query_words)

        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(keywords))

        # Enhanced query type detection
        query_lower = query.lower()
        question_words = ['what', 'how', 'when', 'where', 'who', 'why', 'which', 'define', 'explain']

        query_type = "general"
        if any(qw in query_lower for qw in question_words):
            query_type = "question"
        elif "compare" in query_lower or "vs" in query_lower or "difference" in query_lower:
            query_type = "comparison"
        elif "summarize" in query_lower or "summary" in query_lower:
            query_type = "summary"

        return {
            'original_query': query,
            'entities': entities,
            'keywords': keywords,
            'query_type': query_type,
            'processed_query': ' '.join(keywords[:10])  # Limit keywords
        }


    def expand_query_with_synonyms(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with synonyms using WordNet and free APIs - ENHANCEMENT 5"""
        if not self.expansion_enabled:
            return [query]

        print(f"🔄 Expanding query: {query}")

        # Get key terms from query
        key_terms = self._extract_key_terms(query)

        # Get synonyms for key terms
        expanded_queries = [query]  # Always include original

        for term in key_terms[:3]:  # Limit to top 3 terms
            synonyms = self._get_synonyms(term, max_synonyms=2)

            # Create variations by replacing original term with synonyms
            for synonym in synonyms:
                if synonym.lower() != term.lower():
                    expanded_query = query.replace(term, synonym)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)

        # Add semantic variations
        if key_terms:
            # Create query with all synonyms
            all_synonyms = []
            for term in key_terms[:2]:  # Top 2 terms only
                synonyms = self._get_synonyms(term, max_synonyms=1)
                all_synonyms.extend(synonyms)

            if all_synonyms:
                synonym_query = f"{query} {' '.join(all_synonyms[:3])}"
                expanded_queries.append(synonym_query)

        # Limit total expansions
        result = expanded_queries[:max_expansions + 1]
        print(f"📈 Query expanded to {len(result)} variations")
        return result

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important terms from query for expansion - ENHANCEMENT 5"""
        if not self.nlp:
            # Simple fallback
            words = query.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            return [word for word in words if word not in stop_words and len(word) > 2]

        doc = self.nlp(query)
        key_terms = []

        # Extract nouns, verbs, and adjectives (not stop words)
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and
                    not token.is_stop and
                    not token.is_punct and
                    len(token.text) > 2):
                key_terms.append(token.lemma_.lower())

        return key_terms

    def _get_synonyms(self, word: str, max_synonyms: int = 2) -> List[str]:
        """Get synonyms using WordNet and free APIs - ENHANCEMENT 5"""
        # Check cache first
        if word in self.synonym_cache:
            return self.synonym_cache[word]

        synonyms = set()

        # Method 1: WordNet (free, offline)
        wordnet_synonyms = self._get_wordnet_synonyms(word)
        synonyms.update(wordnet_synonyms)

        # Method 2: DataMuse API (free, online)
        if len(synonyms) < max_synonyms:
            datamuse_synonyms = self._get_datamuse_synonyms(word)
            synonyms.update(datamuse_synonyms)

        # Filter and limit
        result = [s for s in synonyms if s != word and len(s) > 2][:max_synonyms]

        # Cache result
        self.synonym_cache[word] = result
        return result

    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet - ENHANCEMENT 5"""
        synonyms = set()

        try:
            # Get synsets (synonym sets) for the word
            synsets = wordnet.synsets(word)

            for synset in synsets[:2]:  # Limit to first 2 synsets
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
        except Exception as e:
            print(f"WordNet error for '{word}': {e}")

        return list(synonyms)[:3]  # Return top 3

    def _get_datamuse_synonyms(self, word: str) -> List[str]:
        """Get synonyms from DataMuse API (free) - ENHANCEMENT 5"""
        try:
            url = f"https://api.datamuse.com/words?rel_syn={word}&max=3"
            response = requests.get(url, timeout=2)  # Short timeout

            if response.status_code == 200:
                data = response.json()
                return [item['word'] for item in data if 'word' in item]

        except Exception as e:
            print(f"DataMuse API error for '{word}': {e}")

        return []

    def enhanced_retrieve_context(self, query: str, vector_store, k: int = 5) -> List[Dict]:
        """Enhanced context retrieval with query expansion - ENHANCEMENT 5"""
        # Get expanded queries
        expanded_queries = self.expand_query_with_synonyms(query, max_expansions=2)

        all_results = []
        seen_texts = set()

        # Search with each expanded query
        for i, exp_query in enumerate(expanded_queries):
            try:
                # Adjust k based on query position (give more weight to original)
                query_k = k if i == 0 else max(2, k // 2)
                results = vector_store.similarity_search(exp_query, k=query_k)

                # Add expansion info to results
                for result in results:
                    result['expansion_query'] = exp_query
                    result['is_original_query'] = (i == 0)

                    # Avoid duplicates
                    if result['text'] not in seen_texts:
                        all_results.append(result)
                        seen_texts.add(result['text'])

            except Exception as e:
                print(f"Error searching with expanded query '{exp_query}': {e}")

        # Sort by relevance (prioritize original query results)
        all_results.sort(key=lambda x: (
            -1 if x.get('is_original_query', False) else 0,  # Original query first
            -x.get('similarity_score', 0)  # Then by similarity
        ))

        return all_results[:k]

    def toggle_expansion(self, enabled: bool):
        """Enable or disable query expansion - ENHANCEMENT 5"""
        self.expansion_enabled = enabled
        print(f"🔧 Query expansion {'enabled' if enabled else 'disabled'}")

    def get_expansion_stats(self) -> Dict:
        """Get query expansion statistics - ENHANCEMENT 5"""
        return {
            'expansion_enabled': self.expansion_enabled,
            'cached_synonyms': len(self.synonym_cache),
            'synonym_cache_keys': list(self.synonym_cache.keys())[:10]  # Show first 10
        }




    def retrieve_context(self, query: str, vector_store, k: int = 5) -> List[Dict]:
        """Process query and retrieve relevant context"""
        query_analysis = self.analyze_query(query)


        if self.expansion_enabled:
            return self.enhanced_retrieve_context(query, vector_store, k)


        # Original retrieval logic (if expansion disabled)
        expanded_queries = [query]  # Just original query

        all_results = []
        for exp_query in expanded_queries:
            results = vector_store.similarity_search(exp_query, k=k // len(expanded_queries) + 1)
            all_results.extend(results)

        seen_texts = set()
        unique_results = []
        for result in all_results:
            if result['text'] not in seen_texts:
                unique_results.append(result)
                seen_texts.add(result['text'])

        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return unique_results[:k]

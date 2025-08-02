import re
import spacy
from typing import Dict, List, Optional
from collections import defaultdict


class TechnicalQueryRouter:
    """
    Professional technical query classification and routing system
    Specializes in programming language queries, method comparisons, and technical concepts
    """

    def __init__(self):
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not available. Using pattern-based classification.")
            self.nlp = None

        # Programming language detection patterns
        self.programming_languages = {
            'java': ['java', 'jvm', 'spring', 'hibernate', 'thread', 'synchronized'],
            'python': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'node', 'react', 'vue', 'angular'],
            'c++': ['c++', 'cpp', 'iostream', 'namespace', 'class'],
            'c#': ['c#', 'csharp', '.net', 'visual studio'],
            'go': ['golang', 'goroutine', 'channel'],
            'rust': ['rust', 'cargo', 'ownership', 'borrowing']
        }

        # Technical query patterns for precise classification
        self.technical_patterns = {
            'method_comparison': {
                'patterns': [
                    r'(\w+\(\))\s*(?:vs|versus|compared?\s+to|difference\s+between)\s*(\w+\(\))',
                    r'(?:difference\s+between|compare)\s*(\w+\(\))\s*(?:and|with)\s*(\w+\(\))',
                    r'(\w+\(\))\s*(?:or)\s*(\w+\(\))',
                ],
                'keywords': ['vs', 'versus', 'compare', 'difference', 'between', 'or'],
                'requires_precision': True
            },

            'method_explanation': {
                'patterns': [
                    r'(?:what\s+is|explain|describe|how\s+does)\s+(\w+\(\))',
                    r'(\w+\(\))\s+(?:method|function)',
                    r'purpose\s+of\s+(\w+\(\))',
                ],
                'keywords': ['what is', 'explain', 'describe', 'how does', 'purpose'],
                'requires_precision': True
            },

            'synchronization_concepts': {
                'patterns': [
                    r'(?:thread|threading|synchronization|concurrency)',
                    r'(?:lock|mutex|semaphore|monitor)',
                    r'(?:race\s+condition|deadlock|blocking)',
                ],
                'keywords': ['thread', 'sync', 'lock', 'concurrent', 'parallel', 'blocking'],
                'requires_precision': True
            },

            'implementation_guide': {
                'patterns': [
                    r'how\s+to\s+(?:implement|use|create)',
                    r'(?:example|sample|code)\s+(?:of|for)',
                    r'(?:syntax|usage)\s+(?:of|for)',
                ],
                'keywords': ['how to', 'implement', 'example', 'syntax', 'usage', 'create'],
                'requires_precision': False
            },

            'debugging_help': {
                'patterns': [
                    r'(?:error|exception|bug|issue|problem)',
                    r'(?:why\s+(?:is|does)|what\s+causes)',
                    r'(?:not\s+working|fails\s+to)',
                ],
                'keywords': ['error', 'exception', 'bug', 'issue', 'problem', 'why', 'fails'],
                'requires_precision': True
            },

            'performance_query': {
                'patterns': [
                    r'(?:performance|speed|optimization|efficiency)',
                    r'(?:faster|slower|optimize|improve)',
                    r'(?:memory|cpu|resource)\s+(?:usage|consumption)',
                ],
                'keywords': ['performance', 'speed', 'optimization', 'faster', 'memory'],
                'requires_precision': False
            }
        }

        # Programming concept mapping
        self.concept_mappings = {
            'threading_concepts': {
                'yield()': ['cooperative scheduling', 'thread yielding', 'voluntary pause', 'scheduler hint'],
                'join()': ['thread synchronization', 'blocking wait', 'thread completion', 'synchronization barrier'],
                'wait()': ['object monitor', 'conditional waiting', 'notify/notifyAll', 'thread parking'],
                'sleep()': ['timed pause', 'thread suspension', 'non-blocking delay'],
                'synchronized': ['mutual exclusion', 'thread safety', 'critical section', 'monitor lock'],
                'notify()': ['thread notification', 'monitor signaling', 'wake waiting threads'],
                'notifyAll()': ['broadcast notification', 'wake all waiting threads']
            },

            'data_structures': {
                'ArrayList': ['dynamic array', 'resizable array', 'indexed access'],
                'LinkedList': ['linked structure', 'sequential access', 'node-based'],
                'HashMap': ['hash table', 'key-value mapping', 'constant time lookup'],
                'TreeMap': ['balanced tree', 'sorted mapping', 'logarithmic access']
            }
        }

        print("✅ Technical Query Router initialized with programming intelligence")

    def classify_technical_query(self, query: str) -> Dict:

        query_lower = query.lower().strip()


        classification = {
            'type': 'general',
            'subtype': None,
            'programming_language': None,
            'methods_mentioned': [],
            'technical_concepts': [],
            'requires_precision': False,
            'confidence': 0.0,
            'routing_strategy': 'standard',
            'specialized_processing': []
        }


        methods = self._extract_methods(query)
        concepts = self._extract_technical_concepts(query)
        language = self._detect_programming_language(query)


        query_type, confidence = self._classify_query_type(query_lower)


        is_technical = self._is_technical_query(query_lower, methods, concepts)

        if is_technical:
            classification.update({
                'type': 'technical',
                'subtype': query_type,
                'programming_language': language,
                'methods_mentioned': methods,
                'technical_concepts': concepts,
                'requires_precision': self._requires_high_precision(query_type),
                'confidence': confidence,
                'routing_strategy': self._get_routing_strategy(query_type),
                'specialized_processing': self._get_specialized_processing(query_type, methods)
            })

        return classification

    def _extract_methods(self, query: str) -> List[str]:


        method_pattern = r'\b(\w+)\s*\(\s*\)'
        methods = re.findall(method_pattern, query, re.IGNORECASE)

        # Clean and deduplicate
        unique_methods = list(set([method.lower() for method in methods]))
        return unique_methods

    def _extract_technical_concepts(self, query: str) -> List[str]:

        concepts = []
        query_lower = query.lower()


        for category, concept_map in self.concept_mappings.items():
            for concept, related_terms in concept_map.items():
                if concept.lower() in query_lower:
                    concepts.append(concept)


                for term in related_terms:
                    if term.lower() in query_lower:
                        concepts.append(term)


        technical_terms = [
            'thread', 'process', 'synchronization', 'lock', 'mutex', 'semaphore',
            'blocking', 'non-blocking', 'asynchronous', 'concurrent', 'parallel',
            'deadlock', 'race condition', 'critical section', 'monitor',
            'algorithm', 'data structure', 'complexity', 'performance'
        ]

        for term in technical_terms:
            if term in query_lower:
                concepts.append(term)

        return list(set(concepts))

    def _detect_programming_language(self, query: str) -> Optional[str]:

        query_lower = query.lower()

        for language, keywords in self.programming_languages.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return language

        return None

    def _classify_query_type(self, query_lower: str) -> tuple:

        best_match = None
        highest_confidence = 0.0

        for query_type, config in self.technical_patterns.items():
            confidence = 0.0


            for pattern in config['patterns']:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    confidence += 0.4


            keyword_matches = sum(1 for keyword in config['keywords']
                                  if keyword in query_lower)
            confidence += (keyword_matches / len(config['keywords'])) * 0.6

            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match = query_type

        return best_match, highest_confidence

    def _is_technical_query(self, query_lower: str, methods: List[str], concepts: List[str]) -> bool:
        """Determine if query is technical/programming related"""

        if methods:
            return True


        if concepts:
            return True


        programming_indicators = [
            'java', 'python', 'javascript', 'code', 'programming', 'method',
            'function', 'class', 'object', 'thread', 'process', 'algorithm',
            'implementation', 'syntax', 'api', 'library', 'framework'
        ]

        if any(indicator in query_lower for indicator in programming_indicators):
            return True

        return False

    def _requires_high_precision(self, query_type: str) -> bool:

        if not query_type:
            return False

        high_precision_types = [
            'method_comparison', 'method_explanation',
            'synchronization_concepts', 'debugging_help'
        ]

        return query_type in high_precision_types

    def _get_routing_strategy(self, query_type: str) -> str:

        routing_map = {
            'method_comparison': 'multi_method_retrieval',
            'method_explanation': 'method_focused_retrieval',
            'synchronization_concepts': 'concept_hierarchical_retrieval',
            'implementation_guide': 'example_focused_retrieval',
            'debugging_help': 'problem_solution_retrieval',
            'performance_query': 'best_practice_retrieval'
        }

        return routing_map.get(query_type, 'standard_retrieval')

    def _get_specialized_processing(self, query_type: str, methods: List[str]) -> List[str]:

        processing = []

        if query_type == 'method_comparison' and len(methods) >= 2:
            processing.extend(['comparative_analysis', 'technical_validation'])

        if query_type == 'method_explanation':
            processing.extend(['method_signature_extraction', 'behavior_analysis'])

        if methods:
            processing.append('method_documentation_retrieval')

        processing.append('technical_accuracy_check')

        return processing

    def enhance_query_for_technical_search(self, query: str, classification: Dict) -> List[str]:
        """Generate enhanced queries for better technical retrieval"""
        enhanced_queries = [query]

        if classification['type'] != 'technical':
            return enhanced_queries

        methods = classification['methods_mentioned']
        concepts = classification['technical_concepts']


        for method in methods:
            if method in self.concept_mappings.get('threading_concepts', {}):
                related_concepts = self.concept_mappings['threading_concepts'][method]
                for concept in related_concepts[:2]:  # Top 2 concepts
                    enhanced_query = f"{query} {concept}"
                    enhanced_queries.append(enhanced_query)


        if classification['programming_language']:
            lang_query = f"{classification['programming_language']} {query}"
            enhanced_queries.append(lang_query)


        if classification['subtype'] == 'method_comparison' and len(methods) >= 2:
            comparison_query = f"difference between {' and '.join(methods)} behavior synchronization blocking"
            enhanced_queries.append(comparison_query)

        return enhanced_queries[:4]
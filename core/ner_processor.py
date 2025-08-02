import spacy
from typing import List, Dict, Set, Tuple
import re
from collections import defaultdict, Counter


class NERProcessor:
    """Named Entity Recognition using spaCy and custom patterns - ENHANCEMENT 8"""

    def __init__(self):
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy English model loaded")
        except OSError:
            print("📥 Installing spaCy English model...")
            import subprocess
            try:
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model installed and loaded")
            except Exception as e:
                print(f"❌ Could not install spaCy model: {e}")
                self.nlp = None

        # Custom entity patterns for domain-specific recognition
        self.custom_patterns = {
            'TECH': {
                'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                                          'swift'],
                'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel'],
                'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'cassandra'],
                'cloud_platforms': ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services']
            },
            'BUSINESS': {
                'metrics': ['roi', 'kpi', 'revenue', 'profit', 'ebitda', 'market cap', 'valuation'],
                'roles': ['ceo', 'cto', 'cfo', 'manager', 'director', 'analyst', 'consultant']
            },
            'SCIENCE': {
                'fields': ['machine learning', 'artificial intelligence', 'data science', 'nlp', 'computer vision'],
                'methods': ['regression', 'classification', 'clustering', 'neural network', 'deep learning']
            }
        }

        # Entity cache for performance
        self.entity_cache = {}
        print("✅ NER processor initialized with custom patterns")

    def extract_entities(self, text: str, include_custom: bool = True) -> Dict:
        """Extract named entities from text - ENHANCEMENT 8"""
        if text in self.entity_cache:
            return self.entity_cache[text]

        if not self.nlp:
            return self._fallback_entity_extraction(text)

        # Process with spaCy
        doc = self.nlp(text)

        # Standard spaCy entities
        standard_entities = []
        for ent in doc.ents:
            standard_entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 0.9),
                'description': spacy.explain(ent.label_) or ent.label_
            })

        # Custom domain entities
        custom_entities = []
        if include_custom:
            custom_entities = self._extract_custom_entities(text)

        # Combine results
        all_entities = standard_entities + custom_entities

        # Extract relationships between entities
        relationships = self._extract_simple_relationships(doc, all_entities)

        result = {
            'standard_entities': standard_entities,
            'custom_entities': custom_entities,
            'all_entities': all_entities,
            'entity_counts': self._count_entity_types(all_entities),
            'relationships': relationships
        }

        # Cache result
        self.entity_cache[text] = result
        return result

    def _extract_custom_entities(self, text: str) -> List[Dict]:
        """Extract domain-specific entities using patterns - ENHANCEMENT 8"""
        custom_entities = []
        text_lower = text.lower()

        for domain, categories in self.custom_patterns.items():
            for category, items in categories.items():
                for item in items:
                    # Use word boundaries for better matching
                    pattern = r'\b' + re.escape(item.lower()) + r'\b'
                    matches = re.finditer(pattern, text_lower)

                    for match in matches:
                        # Get original text (preserve case)
                        original_text = text[match.start():match.end()]

                        custom_entities.append({
                            'text': original_text,
                            'label': f'CUSTOM_{domain}_{category.upper()}',
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 0.8,
                            'description': f'{domain.lower()} {category.replace("_", " ")}',
                            'domain': domain,
                            'category': category
                        })

        return custom_entities

    def _extract_simple_relationships(self, doc, entities: List[Dict]) -> List[Dict]:
        """Extract simple relationships between entities - ENHANCEMENT 8"""
        relationships = []

        if not doc:
            return relationships

        # Simple subject-verb-object relationships
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp']:
                # Find subject and object
                subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]

                for subj in subjects:
                    for obj in objects:
                        # Check if subject and object are entities
                        subj_entities = [e for e in entities if e['start'] <= subj.idx < e['end']]
                        obj_entities = [e for e in entities if e['start'] <= obj.idx < e['end']]

                        if subj_entities and obj_entities:
                            relationships.append({
                                'subject': subj_entities[0]['text'],
                                'subject_type': subj_entities[0]['label'],
                                'relation': token.lemma_,
                                'object': obj_entities[0]['text'],
                                'object_type': obj_entities[0]['label'],
                                'confidence': 0.7
                            })

        return relationships

    def _fallback_entity_extraction(self, text: str) -> Dict:
        """Fallback entity extraction when spaCy not available - ENHANCEMENT 8"""
        print("⚠️ Using fallback entity extraction (spaCy not available)")

        # Extract custom entities only
        custom_entities = self._extract_custom_entities(text)

        # Simple pattern-based extraction for common entities
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'URL': r'https?://[^\s]+',
            'PHONE': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'DATE': r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
        }

        pattern_entities = []
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                pattern_entities.append({
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'description': entity_type.lower()
                })

        all_entities = custom_entities + pattern_entities

        return {
            'standard_entities': pattern_entities,
            'custom_entities': custom_entities,
            'all_entities': all_entities,
            'entity_counts': self._count_entity_types(all_entities),
            'relationships': []
        }

    def _count_entity_types(self, entities: List[Dict]) -> Dict:
        """Count entities by type - ENHANCEMENT 8"""
        counts = Counter([ent['label'] for ent in entities])
        return dict(counts)

    def create_entity_index(self, documents: List[Dict]) -> Dict:
        """Create searchable entity index across documents - ENHANCEMENT 8"""
        print(f"🔨 Building entity index for {len(documents)} documents...")

        entity_index = defaultdict(lambda: defaultdict(list))

        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            source = doc.get('metadata', {}).get('source', f'doc_{doc_idx}')

            try:
                entities_data = self.extract_entities(text)

                for entity in entities_data['all_entities']:
                    entity_text = entity['text'].lower()
                    entity_label = entity['label']

                    entity_index[entity_label][entity_text].append({
                        'document_index': doc_idx,
                        'source': source,
                        'start_pos': entity['start'],
                        'end_pos': entity['end'],
                        'confidence': entity['confidence'],
                        'context': text[max(0, entity['start'] - 50):entity['end'] + 50]
                    })
            except Exception as e:
                print(f"⚠️ Error processing document {doc_idx}: {e}")

        print(f"✅ Entity index created with {len(entity_index)} entity types")
        return dict(entity_index)

    def search_by_entity(self, entity_index: Dict, entity_type: str = None,
                         entity_text: str = None) -> List[Dict]:
        """Search documents by entity type or specific entity - ENHANCEMENT 8"""
        results = []

        if entity_type and entity_type in entity_index:
            if entity_text:
                # Search for specific entity text
                entity_text_lower = entity_text.lower()
                if entity_text_lower in entity_index[entity_type]:
                    results.extend(entity_index[entity_type][entity_text_lower])
            else:
                # Get all entities of this type
                for entities in entity_index[entity_type].values():
                    results.extend(entities)
        elif entity_text:
            # Search for entity text across all types
            entity_text_lower = entity_text.lower()
            for entity_type_data in entity_index.values():
                if entity_text_lower in entity_type_data:
                    results.extend(entity_type_data[entity_text_lower])

        return results

    def get_entity_summary(self, documents: List[Dict]) -> Dict:
        """Get entity summary across all documents - ENHANCEMENT 8"""
        print(f"📊 Generating entity summary for {len(documents)} documents...")

        all_entity_counts = defaultdict(int)
        entity_frequencies = defaultdict(int)

        for doc in documents:
            text = doc.get('text', '')
            try:
                entities_data = self.extract_entities(text)

                for entity in entities_data['all_entities']:
                    entity_text = entity['text'].lower()
                    entity_label = entity['label']

                    all_entity_counts[entity_label] += 1
                    entity_frequencies[entity_text] += 1

            except Exception as e:
                print(f"⚠️ Error in entity summary for document: {e}")

        # Get most common entities
        top_entities = sorted(entity_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            'total_unique_entities': len(entity_frequencies),
            'entity_type_counts': dict(all_entity_counts),
            'most_common_entities': top_entities,
            'custom_patterns_available': len(self.custom_patterns),
            'spacy_available': self.nlp is not None
        }


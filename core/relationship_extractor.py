
import spacy
from typing import List, Dict, Tuple, Set
import re
from collections import defaultdict


class RelationshipExtractor:
    """Extract relationships between entities using spaCy and patterns - ENHANCEMENT 10"""

    def __init__(self):
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded for relationship extraction")
        except OSError:
            print("⚠️ spaCy model not available, using pattern-based extraction only")
            self.nlp = None

        # Define relationship patterns
        self.relationship_patterns = {
            'IS_A': [
                r'(\w+(?:\s+\w+)*)\s+is\s+a\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+are\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+was\s+a\s+(\w+(?:\s+\w+)*)'
            ],
            'HAS': [
                r'(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+contains\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+includes\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+owns\s+(\w+(?:\s+\w+)*)'
            ],
            'USES': [
                r'(\w+(?:\s+\w+)*)\s+uses\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+utilizes\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+employs\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+applies\s+(\w+(?:\s+\w+)*)'
            ],
            'LOCATED_IN': [
                r'(\w+(?:\s+\w+)*)\s+in\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+at\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+located\s+in\s+(\w+(?:\s+\w+)*)'
            ],
            'CAUSES': [
                r'(\w+(?:\s+\w+)*)\s+causes\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+leads\s+to\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+results\s+in\s+(\w+(?:\s+\w+)*)'
            ],
            'WORKS_FOR': [
                r'(\w+(?:\s+\w+)*)\s+works\s+for\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+employed\s+by\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+at\s+(\w+(?:\s+\w+)*)'
            ]
        }

        print("✅ Relationship extractor initialized")

    def extract_relationships(self, text: str, include_entities: bool = True) -> Dict:
        """Extract relationships from text - ENHANCEMENT 10"""
        print(f"🔗 Extracting relationships from text ({len(text.split())} words)")

        # Pattern-based extraction (always available)
        pattern_relationships = self._extract_pattern_relationships(text)

        # Dependency-based extraction (if spaCy available)
        dependency_relationships = []
        entity_relationships = []

        if self.nlp:
            doc = self.nlp(text)
            dependency_relationships = self._extract_dependency_relationships(doc)

            if include_entities:
                entity_relationships = self._extract_entity_relationships(doc)

        # Combine all relationships
        all_relationships = pattern_relationships + dependency_relationships + entity_relationships

        # Remove duplicates and create relationship summary
        unique_relationships = self._deduplicate_relationships(all_relationships)

        return {
            'relationships': unique_relationships,
            'relationship_count': len(unique_relationships),
            'pattern_based': len(pattern_relationships),
            'dependency_based': len(dependency_relationships),
            'entity_based': len(entity_relationships),
            'relationship_types': self._categorize_relationships(unique_relationships)
        }

    def _extract_pattern_relationships(self, text: str) -> List[Dict]:
        """Extract relationships using regex patterns - ENHANCEMENT 10"""
        relationships = []

        for relation_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    if len(match.groups()) >= 2:
                        subject = match.group(1).strip()
                        obj = match.group(2).strip()

                        # Filter out very short or common words
                        if len(subject) > 2 and len(obj) > 2:
                            relationships.append({
                                'subject': subject,
                                'relation': relation_type,
                                'object': obj,
                                'confidence': 0.8,
                                'method': 'pattern',
                                'start_pos': match.start(),
                                'end_pos': match.end(),
                                'context': text[max(0, match.start() - 50):match.end() + 50]
                            })

        return relationships

    def _extract_dependency_relationships(self, doc) -> List[Dict]:
        """Extract relationships using dependency parsing - ENHANCEMENT 10"""
        relationships = []

        for token in doc:
            # Subject-Verb-Object relationships
            if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                verb = token.head
                subject = token.text

                # Find direct objects
                for child in verb.children:
                    if child.dep_ == 'dobj':
                        relationships.append({
                            'subject': subject,
                            'relation': verb.lemma_,
                            'object': child.text,
                            'confidence': 0.7,
                            'method': 'dependency',
                            'pos_pattern': 'nsubj-verb-dobj'
                        })

                    # Find prepositional objects
                    elif child.dep_ == 'prep':
                        for grandchild in child.children:
                            if grandchild.dep_ == 'pobj':
                                relationships.append({
                                    'subject': subject,
                                    'relation': f"{verb.lemma_}_{child.text}",
                                    'object': grandchild.text,
                                    'confidence': 0.6,
                                    'method': 'dependency',
                                    'pos_pattern': 'nsubj-verb-prep-pobj'
                                })

            # Compound relationships (adjective-noun, noun-noun)
            elif token.dep_ == 'compound':
                relationships.append({
                    'subject': token.text,
                    'relation': 'MODIFIES',
                    'object': token.head.text,
                    'confidence': 0.5,
                    'method': 'dependency',
                    'pos_pattern': 'compound'
                })

            # Possession relationships
            elif token.dep_ == 'poss':
                relationships.append({
                    'subject': token.text,
                    'relation': 'POSSESSES',
                    'object': token.head.text,
                    'confidence': 0.6,
                    'method': 'dependency',
                    'pos_pattern': 'possessive'
                })

        return relationships

    def _extract_entity_relationships(self, doc) -> List[Dict]:
        """Extract relationships between named entities - ENHANCEMENT 10"""
        entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
        relationships = []

        # Find relationships between nearby entities
        for i, (ent1_text, ent1_label, ent1_start, ent1_end) in enumerate(entities):
            for j, (ent2_text, ent2_label, ent2_start, ent2_end) in enumerate(entities[i + 1:], i + 1):

                # Only consider entities close to each other (within 15 tokens)
                distance = abs(ent1_start - ent2_end) if ent1_start > ent2_end else abs(ent2_start - ent1_end)

                if distance <= 15:
                    # Extract connecting words between entities
                    start_pos = min(ent1_start, ent2_start)
                    end_pos = max(ent1_end, ent2_end)
                    between_tokens = doc[start_pos:end_pos]

                    # Find verbs or prepositions between entities
                    connecting_words = []
                    for token in between_tokens:
                        if token.pos_ in ['VERB', 'ADP'] and token.text.lower() not in [ent1_text.lower(),
                                                                                        ent2_text.lower()]:
                            connecting_words.append(token.text)

                    relation = '_'.join(
                        connecting_words) if connecting_words else f'RELATED_TO_{ent1_label}_{ent2_label}'

                    relationships.append({
                        'subject': ent1_text,
                        'subject_type': ent1_label,
                        'relation': relation,
                        'object': ent2_text,
                        'object_type': ent2_label,
                        'confidence': max(0.3, 0.8 - distance * 0.05),  # Confidence decreases with distance
                        'method': 'entity',
                        'distance': distance
                    })

        return relationships

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships - ENHANCEMENT 10"""
        seen = set()
        unique_relationships = []

        for rel in relationships:
            # Create a key for deduplication
            subject = rel.get('subject', '').lower().strip()
            relation = rel.get('relation', '').lower().strip()
            obj = rel.get('object', '').lower().strip()

            key = (subject, relation, obj)

            if key not in seen and subject and obj and relation:
                seen.add(key)
                unique_relationships.append(rel)

        return unique_relationships

    def _categorize_relationships(self, relationships: List[Dict]) -> Dict:
        """Categorize relationships by type - ENHANCEMENT 10"""
        categories = defaultdict(int)

        for rel in relationships:
            relation_type = rel.get('relation', 'UNKNOWN')
            categories[relation_type] += 1

        return dict(categories)

    def create_relationship_graph(self, relationships: List[Dict]) -> Dict:
        """Create a graph representation of relationships - ENHANCEMENT 10"""
        print("🕸️ Creating relationship graph...")

        nodes = set()
        edges = []

        for rel in relationships:
            subject = rel.get('subject', '')
            obj = rel.get('object', '')
            relation = rel.get('relation', '')

            if subject and obj:
                nodes.add(subject)
                nodes.add(obj)

                edges.append({
                    'from': subject,
                    'to': obj,
                    'label': relation,
                    'confidence': rel.get('confidence', 0.5),
                    'method': rel.get('method', 'unknown')
                })

        # Calculate node statistics
        node_connections = defaultdict(int)
        for edge in edges:
            node_connections[edge['from']] += 1
            node_connections[edge['to']] += 1

        # Create node list with metadata
        node_list = []
        for node in nodes:
            node_list.append({
                'id': node,
                'label': node,
                'connections': node_connections[node],
                'size': min(node_connections[node] * 2 + 5, 20)  # Visual size based on connections
            })

        return {
            'nodes': node_list,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges),
            'most_connected': sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def extract_relationship_patterns(self, documents: List[Dict]) -> Dict:
        """Extract common relationship patterns across documents - ENHANCEMENT 10"""
        print(f"🔍 Analyzing relationship patterns across {len(documents)} documents...")

        all_relationships = []
        pattern_frequency = defaultdict(int)

        for doc in documents:
            text = doc.get('text', '')
            try:
                rel_data = self.extract_relationships(text)
                doc_relationships = rel_data['relationships']
                all_relationships.extend(doc_relationships)

                # Count relationship patterns
                for rel in doc_relationships:
                    pattern = f"{rel.get('relation', 'UNKNOWN')}"
                    pattern_frequency[pattern] += 1

            except Exception as e:
                print(f"⚠️ Error processing document: {e}")

        # Find most common patterns
        common_patterns = sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            'total_relationships': len(all_relationships),
            'unique_patterns': len(pattern_frequency),
            'most_common_patterns': common_patterns,
            'relationship_methods': Counter([rel.get('method', 'unknown') for rel in all_relationships]),
            'average_confidence': np.mean(
                [rel.get('confidence', 0) for rel in all_relationships]) if all_relationships else 0
        }

    def get_extraction_stats(self) -> Dict:
        """Get relationship extraction capabilities - ENHANCEMENT 10"""
        return {
            'spacy_available': self.nlp is not None,
            'pattern_types': len(self.relationship_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.relationship_patterns.values()),
            'supported_methods': ['pattern'] + (['dependency', 'entity'] if self.nlp else [])
        }


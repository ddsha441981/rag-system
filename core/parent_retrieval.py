from typing import Dict, List, Optional, Tuple
import uuid
from collections import defaultdict
import json
from datetime import datetime


class ParentDocumentRetriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store


        self.parent_documents = {}  # parent_id -> full document
        self.child_parent_map = {}  # child_id -> parent_info
        self.parent_child_map = defaultdict(list)  # parent_id -> [child_ids]

        print("✅ Parent document retriever initialized")

    def index_document_with_hierarchy(self, file_path: str, document_text: str,
                                      small_chunk_size: int = 250,
                                      large_chunk_size: int = 1000) -> str:
        parent_id = str(uuid.uuid4())

        print(f"📚 Indexing document with hierarchy: {file_path}")

        self.parent_documents[parent_id] = {
            'text': document_text,
            'source': file_path,
            'metadata': {
                'file_path': file_path,
                'word_count': len(document_text.split()),
                'created_at': datetime.now().isoformat(),
                'document_type': self._detect_document_type(document_text)
            },
            'child_count': 0
        }

        small_chunks = self._create_small_chunks(document_text, parent_id, small_chunk_size)


        for chunk in small_chunks:
            child_id = chunk['metadata']['chunk_id']
            self.child_parent_map[child_id] = {
                'parent_id': parent_id,
                'chunk_index': chunk['metadata']['chunk_index'],
                'source': file_path
            }
            self.parent_child_map[parent_id].append(child_id)


        self.parent_documents[parent_id]['child_count'] = len(small_chunks)


        self.vector_store.add_documents(small_chunks)

        print(f"✅ Created {len(small_chunks)} child chunks for parent document")
        return parent_id

    def _create_small_chunks(self, text: str, parent_id: str, chunk_size: int) -> List[Dict]:

        words = text.split()
        chunks = []
        overlap = chunk_size // 4  #

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text.strip()) > 10:
                chunk_id = f"{parent_id}_chunk_{len(chunks)}"

                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_id': chunk_id,
                        'parent_id': parent_id,
                        'chunk_index': len(chunks),
                        'start_word': i,
                        'end_word': min(i + chunk_size, len(words)),
                        'word_count': len(chunk_words),
                        'source': self.parent_documents[parent_id]['source']
                    },
                    'word_count': len(chunk_words)
                })

        return chunks

    def retrieve_with_parent_context(self, query: str, k: int = 5,
                                     context_window: int = 500) -> List[Dict]:
        print(f"🔍 Retrieving with parent context for: {query}")

        initial_results = self.vector_store.similarity_search(query, k=k * 2)

        enhanced_results = []
        seen_parents = set()

        for result in initial_results:
            if len(enhanced_results) >= k:
                break

            chunk_id = result['metadata'].get('chunk_id')
            if not chunk_id or chunk_id not in self.child_parent_map:
                enhanced_results.append(result)
                continue

            parent_info = self.child_parent_map[chunk_id]
            parent_id = parent_info['parent_id']

            parent_context = self._get_parent_context(
                parent_id,
                parent_info['chunk_index'],
                context_window
            )

            enhanced_result = {
                'text': result['text'],
                'parent_context': parent_context,
                'metadata': {
                    **result['metadata'],
                    'has_parent_context': True,
                    'parent_id': parent_id,
                    'context_window_size': len(parent_context.split())
                },
                'similarity_score': result.get('similarity_score', 0),
                'retrieval_method': 'parent_enhanced'
            }

            enhanced_results.append(enhanced_result)
            seen_parents.add(parent_id)

        print(f"✅ Enhanced {len(enhanced_results)} results with parent context")
        return enhanced_results

    def _get_parent_context(self, parent_id: str, chunk_index: int, context_window: int) -> str:
        if parent_id not in self.parent_documents:
            return ""

        parent_doc = self.parent_documents[parent_id]
        parent_words = parent_doc['text'].split()

        chunk_size = 250
        chunk_start = chunk_index * chunk_size

        context_start = max(0, chunk_start - context_window // 2)
        context_end = min(len(parent_words), chunk_start + chunk_size + context_window // 2)

        context_words = parent_words[context_start:context_end]
        return ' '.join(context_words)

    def _detect_document_type(self, text: str) -> str:
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ['class ', 'def ', 'import ', 'function', 'algorithm']):
            return 'technical'


        elif any(keyword in text_lower for keyword in ['abstract', 'introduction', 'methodology', 'references']):
            return 'academic'

        elif any(keyword in text_lower for keyword in ['whereas', 'pursuant', 'contract', 'agreement']):
            return 'legal'

        elif any(keyword in text_lower for keyword in ['revenue', 'profit', 'business', 'market']):
            return 'business'

        else:
            return 'general'

    def get_parent_document(self, parent_id: str) -> Optional[Dict]:

        return self.parent_documents.get(parent_id)

    def get_hierarchy_stats(self) -> Dict:
        total_parents = len(self.parent_documents)
        total_children = len(self.child_parent_map)

        if total_parents == 0:
            return {
                'total_parent_documents': 0,
                'total_child_chunks': 0,
                'average_chunks_per_document': 0
            }

        doc_types = defaultdict(int)
        for parent_doc in self.parent_documents.values():
            doc_type = parent_doc['metadata'].get('document_type', 'unknown')
            doc_types[doc_type] += 1

        return {
            'total_parent_documents': total_parents,
            'total_child_chunks': total_children,
            'average_chunks_per_document': total_children / total_parents,
            'document_types': dict(doc_types),
            'largest_document': max([doc['metadata']['word_count']
                                     for doc in self.parent_documents.values()]) if total_parents > 0 else 0
        }


from typing import Dict, List, Optional
from .document_processor import UniversalDocumentProcessor
from .chunking import SmartChunker
from .vector_store import VectorStore, ShardedVectorStore
from .retrieval import HybridRetriever, MultiVectorRetriever
from .query_processor import QueryProcessor
from .prompt_builder import PromptBuilder
from .llm_handler import GroqLLMHandler
from .cache import MultiLevelCache
from .memory import ConversationMemory


from .contextual_compression import ContextualCompressor
from .parent_retrieval import ParentDocumentRetriever
from .ner_processor import NERProcessor
from .summarization import DocumentSummarizer
from .relationship_extractor import RelationshipExtractor


from .technical_query_router import TechnicalQueryRouter
from .technical_validator import TechnicalAccuracyValidator

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import gc
import psutil


class EnhancedRAGSystem:

    def __init__(self, groq_api_key: str, collection_name: str = "enhanced_knowledge_base",
                 redis_host: str = "localhost", redis_port: int = 6379,
                 use_sharding: bool = False, num_shards: int = 4,
                 enable_advanced_nlp: bool = True, enable_technical_intelligence: bool = True):

        print("🚀 Initializing Enhanced RAG System...")


        self.doc_processor = UniversalDocumentProcessor()
        self.chunker = SmartChunker()

        if use_sharding:
            self.vector_store = ShardedVectorStore(num_shards=num_shards)
            print(f"✅ Using sharded vector store with {num_shards} shards")
        else:
            self.vector_store = VectorStore()
            print("✅ Using standard vector store")

        self.collection = self.vector_store.create_collection(collection_name)


        self.query_processor = QueryProcessor()
        self.prompt_builder = PromptBuilder()
        self.llm_handler = GroqLLMHandler(groq_api_key)


        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.retriever = MultiVectorRetriever(self.vector_store)
        print("✅ Using multi-vector retriever (dense + sparse)")


        self.query_cache = MultiLevelCache(redis_host=redis_host, redis_port=redis_port)


        self.conversation_memory = ConversationMemory()


        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.performance_stats = {
            'total_queries': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'total_documents_processed': 0
        }


        if enable_advanced_nlp:
            print("🧠 Initializing advanced NLP components...")
            self.contextual_compressor = ContextualCompressor()
            self.parent_retriever = ParentDocumentRetriever(self.vector_store)
            self.ner_processor = NERProcessor()
            self.document_summarizer = DocumentSummarizer()
            self.relationship_extractor = RelationshipExtractor()
            print("✅ Advanced NLP components initialized")
        else:
            self.contextual_compressor = None
            self.parent_retriever = None
            self.ner_processor = None
            self.document_summarizer = None
            self.relationship_extractor = None


        if enable_technical_intelligence:
            print("🔬 Initializing Technical Intelligence Components...")
            self.technical_router = TechnicalQueryRouter()
            self.technical_validator = TechnicalAccuracyValidator()
            print("✅ Technical Intelligence components initialized")
        else:
            self.technical_router = None
            self.technical_validator = None


        self._initialize_indices()
        print("🎉 Enhanced RAG System fully initialized!")

    def _initialize_indices(self):
        try:
            if hasattr(self.vector_store, 'get_all_documents'):
                results = self.vector_store.get_all_documents()
            else:
                results = self.collection.get() if self.collection else {'documents': [], 'metadatas': []}

            if results and results.get('documents'):
                print("🔨 Building multi-vector indices...")
                self.retriever.build_sparse_indices(results['documents'], results.get('metadatas', []))
                self.hybrid_retriever.build_bm25_index(results['documents'], results.get('metadatas', []))
                print("✅ All indices initialized")
            else:
                print("⚠️ No documents found for index initialization")
        except Exception as e:
            print(f"❌ Index initialization failed: {e}")

    def process_document(self, file_path: str) -> int:
        print(f"📄 Processing document: {file_path}")
        pages = self.doc_processor.process_document(file_path)
        all_chunks = []
        for page_data in pages:
            chunks = self.chunker.semantic_chunking(page_data['text'], page_data)
            all_chunks.extend(chunks)
        self.vector_store.add_documents(all_chunks)
        documents = [chunk['text'] for chunk in all_chunks]
        metadata = [chunk['metadata'] for chunk in all_chunks]
        self.hybrid_retriever.build_bm25_index(documents, metadata)
        self.retriever.build_sparse_indices(documents, metadata)
        print(f"✅ Document processed: {len(all_chunks)} chunks")
        return len(all_chunks)

    async def async_process_document(self, file_path: str) -> int:
        start_time = time.time()
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(self.thread_pool, self.doc_processor.process_document, file_path)
        chunk_tasks = [
            loop.run_in_executor(self.thread_pool, self.chunker.semantic_chunking, page['text'], page) for page in pages
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)
        all_chunks = [c for chunks in chunk_results for c in chunks]
        await loop.run_in_executor(self.thread_pool, self._batch_insert_chunks, all_chunks)
        self.performance_stats['total_documents_processed'] += 1
        print(f"✅ Async processed {len(all_chunks)} chunks in {time.time() - start_time:.2f} seconds")
        return len(all_chunks)

    def _batch_insert_chunks(self, chunks: List[Dict], batch_size: int = 100):
        for i in range(0, len(chunks), batch_size):
            self.vector_store.add_documents(chunks[i:i + batch_size])
            time.sleep(0.01)

    def process_document_with_advanced_features(self, file_path: str,
                                                enable_parent_retrieval: bool = True,
                                                enable_ner: bool = True,
                                                enable_summarization: bool = True,
                                                enable_relationships: bool = True) -> Dict:
        print(f"🔬 Processing document with advanced features: {file_path}")
        pages = self.doc_processor.process_document(file_path)
        full_text = ' '.join(page['text'] for page in pages)
        results = {
            'file_path': file_path,
            'pages_processed': len(pages),
            'chunks_created': 0,
            'parent_id': None,
            'entities_extracted': {},
            'summary': None,
            'relationships': {}
        }
        if enable_parent_retrieval and self.parent_retriever:
            parent_id = self.parent_retriever.index_document_with_hierarchy(file_path, full_text)
            results['parent_id'] = parent_id
            print(f"📚 Created parent document: {parent_id}")
        else:
            all_chunks = []
            for page_data in pages:
                chunks = self.chunker.semantic_chunking(page_data['text'], page_data)
                all_chunks.extend(chunks)
            self.vector_store.add_documents(all_chunks)
            results['chunks_created'] = len(all_chunks)
        if enable_ner and self.ner_processor:
            entities_data = self.ner_processor.extract_entities(full_text)
            results['entities_extracted'] = {
                'total_entities': len(entities_data['all_entities']),
                'entity_types': entities_data['entity_counts'],
                'top_entities': sorted(entities_data['entity_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
            }
        if enable_summarization and self.document_summarizer:
            summary_result = self.document_summarizer.summarize_document(full_text)
            results['summary'] = {
                'text': summary_result['summary'],
                'method': summary_result['summary_type'],
                'compression_ratio': summary_result['compression_ratio'],
                'word_count': summary_result['word_count']
            }
        if enable_relationships and self.relationship_extractor:
            relationships_data = self.relationship_extractor.extract_relationships(full_text)
            results['relationships'] = {
                'total_relationships': relationships_data['relationship_count'],
                'relationship_types': relationships_data['relationship_types'],
                'top_relations': sorted(relationships_data['relationship_types'].items(), key=lambda x: x[1], reverse=True)[:10]
            }
        if hasattr(self.vector_store, 'get_all_documents'):
            docs = self.vector_store.get_all_documents()
            if docs and docs.get('documents'):
                self.retriever.build_sparse_indices(docs['documents'], docs.get('metadatas', []))
        self.performance_stats['total_documents_processed'] += 1
        print(f"✅ Advanced processing complete")
        return results

    def query(self, question: str, k: int = 5, conversation_context: str = "",
              search_mode: str = "hybrid", session_id: Optional[str] = None,
              enable_compression: bool = True, use_parent_context: bool = True,
              enable_technical_validation: bool = True) -> Dict:
        start_time = time.time()
        self.performance_stats['total_queries'] += 1

        print(f"🔍 Processing query: {question}")

        query_classification = {'type': 'general'}
        enhanced_queries = [question]
        if self.technical_router:
            query_classification = self.technical_router.classify_technical_query(question)
            enhanced_queries = query_classification.get('enhanced_queries', [question])
            if query_classification.get('type') == 'technical':
                print(f"🎯 Technical query detected: {query_classification.get('subtype')}")
                print(f"📝 Methods mentioned: {query_classification.get('methods_mentioned')}")

        cached_response = self.query_cache.get_cached_response(question)
        if cached_response:
            print("💾 Retrieved answer from cache")
            self.performance_stats['cache_hits'] += 1
            cached_response['cached'] = True
            cached_response['query_classification'] = query_classification
            return cached_response

        context_chunks = []
        if query_classification.get('type') == 'technical':
            print("🎯 Using technical-specialized retrieval...")
            all_chunks = []
            seen_texts = set()
            for i, enhanced_query in enumerate(enhanced_queries):
                query_k = k if i == 0 else max(2, k // 2)
                if use_parent_context and self.parent_retriever:
                    chunks = self.parent_retriever.retrieve_with_parent_context(enhanced_query, query_k)
                elif search_mode == 'hybrid':
                    chunks = self.retriever.multi_vector_search(enhanced_query, query_k)
                else:
                    chunks = self.query_processor.enhanced_retrieve_context(enhanced_query, self.vector_store, query_k)
                for chunk in chunks:
                    chunk['enhanced_query_used'] = enhanced_query
                    chunk['is_primary_query'] = (i == 0)
                    if chunk['text'] not in seen_texts:
                        all_chunks.append(chunk)
                        seen_texts.add(chunk['text'])
            all_chunks.sort(key=lambda x: (
                -1 if x.get('is_primary_query', False) else 0,
                -1 if x.get('metadata', {}).get('content_type') == 'method_documentation' else 0,
                -x.get('similarity_score', 0)
            ))
            context_chunks = all_chunks[:k]
        else:
            if use_parent_context and self.parent_retriever:
                context_chunks = self.parent_retriever.retrieve_with_parent_context(question, k)
            elif search_mode == 'hybrid':
                context_chunks = self.retriever.multi_vector_search(question, k)
            else:
                context_chunks = self.query_processor.enhanced_retrieve_context(question, self.vector_store, k)

        if enable_compression and self.contextual_compressor and len(context_chunks) > 3:
            print("🗜️ Applying contextual compression...")
            context_chunks = self.contextual_compressor.compress_context(context_chunks, question, max_tokens=2000)

        try:
            if query_classification.get('type') == 'technical' and query_classification.get('requires_precision'):
                print("🔬 Using technical-specialized prompting...")
                prompt = self.prompt_builder.build_technical_prompt(question, context_chunks, query_classification,
                                                                    conversation_context)
            else:
                analysis = self.query_processor.analyze_query(question)
                prompt = self.prompt_builder.build_enhanced(question, context_chunks, analysis.get('query_type'),
                                                            conversation_context)
        except Exception as ex:
            return {'Error : Prompt building failed': str(ex)}

        answer_result = self.llm_handler.generate_answer(prompt)

        if answer_result.get('success', False):
            validation_result = {'is_accurate': True}
            if enable_technical_validation and self.technical_validator and query_classification.get('type') == 'technical' and query_classification.get('requires_precision'):
                print("🔍 Performing technical accuracy validation...")
                validation_result = self.technical_validator.validate_technical_response(question, answer_result['answer'], query_classification, context_chunks)
                if not validation_result['is_accurate']:
                    print("⚠️ Technical inaccuracy detected - attempting correction...")
                    correction_prompt = self.technical_validator.generate_correction_prompt(question, answer_result['answer'], validation_result)
                    corrected_result = self.llm_handler.generate_answer(correction_prompt)
                    if corrected_result.get('success', False):
                        answer_result = corrected_result
                        print("✅ Generated corrected technical response")

            result = {
                'question': question,
                'answer': answer_result['answer'],
                'query_classification': query_classification,
                'enhanced_queries_used': enhanced_queries if query_classification.get('type') == 'technical' else [question],
                'technical_validation': validation_result if enable_technical_validation else None,
                'context_chunks': len(context_chunks),
                'sources': [c['metadata'].get('source', 'unknown') for c in context_chunks],
                'tokens_used': answer_result.get('tokens_used', 0),
                'success': True,
                'cached': False
            }

            self._update_performance_stats(time.time() - start_time)
            print(f"⚡ Query processed in {(time.time() - start_time):.2f} seconds")

            self.query_cache.cache_response(question, result)

            if session_id:
                self.conversation_memory.add_conversation(session_id, question, answer_result['answer'], context_chunks)

            return result

        else:
            return {
                'question': question,
                'answer': answer_result.get('answer'),
                'query_classification': query_classification,
                'success': False,
                'error': answer_result.get('error')
            }

    def _update_performance_stats(self, processing_time: float):
        current_avg = self.performance_stats['average_response_time']
        total_queries = self.performance_stats['total_queries']
        new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        self.performance_stats['average_response_time'] = new_avg

    def get_performance_stats(self) -> Dict:
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
        except Exception:
            memory_usage = 0.0
            cpu_percent = 0.0

        return {
            'memory_usage_mb': round(memory_usage, 2),
            'cpu_usage_percent': cpu_percent,
            'total_queries': self.performance_stats['total_queries'],
            'average_response_time': round(self.performance_stats['average_response_time'], 3),
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_hit_rate': (self.performance_stats['cache_hits'] / max(1, self.performance_stats['total_queries'])),
            'total_documents_processed': self.performance_stats['total_documents_processed'],
            'thread_pool_active': len(self.thread_pool._threads) if self.thread_pool._threads else 0
        }

    def get_stats(self) -> Dict:
        try:
            collection_count = self.collection.count()
        except Exception:
            collection_count = 0
        return {
            'documents_indexed': collection_count,
            'cache_available': getattr(self.query_cache, 'redis_available', False),
            'bm25_index_size': len(getattr(self.hybrid_retriever, 'documents', [])),
            'active_sessions': len(self.conversation_memory.conversations),
            'performance': self.get_performance_stats()
        }

    def get_advanced_stats(self) -> Dict:
        base_stats = self.get_performance_stats()
        advanced_stats = {
            'basic_stats': base_stats,
            'components_available': {
                'contextual_compression': self.contextual_compressor is not None,
                'parent_retrieval': self.parent_retriever is not None,
                'ner_processing': self.ner_processor is not None,
                'document_summarization': self.document_summarizer is not None,
                'relationship_extraction': self.relationship_extractor is not None,
                'multi_vector_retrieval': self.retriever is not None,
                'sharded_vector_store': isinstance(self.vector_store, ShardedVectorStore)
            },
            'technical_intelligence': {
                'enabled': self.technical_router is not None,
                'router_available': self.technical_router is not None,
                'validator_available': self.technical_validator is not None
            }
        }
        return advanced_stats

    def analyze_document_collection(self) -> Dict:
        print("📊 Analyzing entire document collection...")
        documents = []
        try:
            if hasattr(self.vector_store, 'get_all_documents'):
                docs_data = self.vector_store.get_all_documents()
                documents = [{'text': doc, 'metadata': meta} for doc, meta in zip(docs_data.get('documents', []), docs_data.get('metadatas', []))]
            else:
                print("⚠️ Vector store API get_all_documents not implemented")
        except Exception as e:
            print(f"❌ Error fetching documents: {e}")

        analysis = {
            'total_documents': len(documents),
            'entity_analysis': {},
            'relationship_analysis': {},
            'parent_hierarchy_stats': {}
        }
        if self.ner_processor and documents:
            analysis['entity_analysis'] = self.ner_processor.get_entity_summary(documents)
        if self.relationship_extractor and documents:
            analysis['relationship_analysis'] = self.relationship_extractor.extract_relationship_patterns(documents)
        if self.parent_retriever:
            analysis['parent_hierarchy_stats'] = self.parent_retriever.get_hierarchy_stats()
        return analysis

    def optimize_memory(self):
        gc.collect()
        if hasattr(self.query_cache, 'memory_cache') and len(self.query_cache.memory_cache) > 1000:
            self.query_cache.memory_cache.clear()
            self.query_cache.memory_cache_timestamps.clear()
            print("🗑️ Cleared memory cache")
        if hasattr(self.ner_processor, 'entity_cache') and len(self.ner_processor.entity_cache) > 500:
            self.ner_processor.entity_cache.clear()
            print("🗑️ Cleared NER cache")
        print("✅ Memory optimized")

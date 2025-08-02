import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
# === END ENHANCEMENT 3 IMPORTS ===

import chromadb
from chromadb.utils import embedding_functions
import uuid
from typing import List, Dict, Optional


class VectorStore:
    def __init__(self, persist_directory="./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = None

    def create_collection(self, collection_name: str):
        """Create or get collection"""
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

        return self.collection

    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to vector store"""
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            metadatas.append({
                'source': chunk['metadata'].get('source', ''),
                'page_number': chunk['metadata'].get('page_number', 0),
                'chunk_id': chunk['metadata'].get('chunk_id', f'chunk_{i}'),
                'word_count': chunk.get('word_count', 0),
                'format': chunk['metadata'].get('format', 'unknown')
            })
            ids.append(str(uuid.uuid4()))

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return f"Added {len(documents)} chunks to vector store"

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]
            })

        return search_results

    def get_all_documents(self):
        """Get all documents from collection"""
        return self.collection.get()

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        self.client.delete_collection(name=collection_name)


class ShardedVectorStore:
    """Sharded vector store for handling large datasets - ENHANCEMENT 3"""

    def __init__(self, persist_directory="./data/chroma_db", num_shards=4):
        self.persist_directory = persist_directory
        self.num_shards = num_shards
        self.shards = {}
        self.thread_lock = threading.Lock()

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Document-to-shard mapping for consistent routing
        self.doc_shard_map = {}

        # Initialize all shards
        self._initialize_shards()
        print(f"✅ Initialized {num_shards} vector database shards")

    def _initialize_shards(self):
        """Initialize multiple ChromaDB shards - ENHANCEMENT 3"""
        for shard_id in range(self.num_shards):
            shard_dir = os.path.join(self.persist_directory, f"shard_{shard_id}")
            os.makedirs(shard_dir, exist_ok=True)

            try:
                client = chromadb.PersistentClient(path=shard_dir)
                self.shards[shard_id] = {
                    'client': client,
                    'collection': None,
                    'document_count': 0
                }
                print(f"✅ Shard {shard_id} initialized")
            except Exception as e:
                print(f"❌ Error initializing shard {shard_id}: {e}")

    def _get_shard_for_document(self, source: str) -> int:
        """Determine which shard to use for a document - ENHANCEMENT 3"""
        # Use consistent hashing based on document source
        hash_value = int(hashlib.md5(source.encode()).hexdigest(), 16)
        shard_id = hash_value % self.num_shards

        # Store mapping for future reference
        self.doc_shard_map[source] = shard_id
        return shard_id

    def create_collection(self, collection_name: str):
        """Create collection across all shards - ENHANCEMENT 3"""
        created_collections = []

        for shard_id, shard_data in self.shards.items():
            try:
                # Try to create collection
                collection = shard_data['client'].create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"✅ Created collection '{collection_name}' in shard {shard_id}")
            except Exception:
                # Collection might already exist, get it
                try:
                    collection = shard_data['client'].get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    print(f"✅ Retrieved existing collection '{collection_name}' from shard {shard_id}")
                except Exception as e:
                    print(f"❌ Error with collection in shard {shard_id}: {e}")
                    continue

            self.shards[shard_id]['collection'] = collection
            created_collections.append(collection)

        return f"Collection '{collection_name}' created/accessed across {len(created_collections)} shards"

    def add_documents(self, chunks: List[Dict]):
        """Add documents with sharding - ENHANCEMENT 3"""
        if not chunks:
            return "No chunks to add"

        # Group chunks by shard
        shard_chunks = {i: [] for i in range(self.num_shards)}

        for chunk in chunks:
            source = chunk['metadata'].get('source', 'unknown')
            shard_id = self._get_shard_for_document(source)
            shard_chunks[shard_id].append(chunk)

        # Add chunks to their respective shards
        total_added = 0
        for shard_id, chunks_for_shard in shard_chunks.items():
            if not chunks_for_shard:
                continue

            collection = self.shards[shard_id]['collection']
            if not collection:
                print(f"❌ No collection available for shard {shard_id}")
                continue

            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []

            for chunk in chunks_for_shard:
                documents.append(chunk['text'])
                metadata = chunk['metadata'].copy()
                metadata['shard_id'] = shard_id  # Add shard info to metadata
                metadatas.append(metadata)
                ids.append(f"shard_{shard_id}_{chunk['metadata'].get('chunk_id', str(uuid.uuid4()))}")

            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

                self.shards[shard_id]['document_count'] += len(documents)
                total_added += len(documents)
                print(f"✅ Added {len(documents)} chunks to shard {shard_id}")

            except Exception as e:
                print(f"❌ Error adding documents to shard {shard_id}: {e}")

        return f"Added {total_added} chunks across {sum(1 for chunks in shard_chunks.values() if chunks)} shards"

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search across all shards and merge results - ENHANCEMENT 3"""
        all_results = []

        def search_shard(shard_id):
            """Search individual shard"""
            collection = self.shards[shard_id]['collection']
            if not collection:
                return []

            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(k * 2, 50),  # Get more results per shard
                    include=['documents', 'metadatas', 'distances']
                )

                shard_results = []
                if results['documents'] and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        shard_results.append({
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': 1 - results['distances'][0][i],
                            'shard_id': shard_id
                        })

                return shard_results

            except Exception as e:
                print(f"❌ Error searching shard {shard_id}: {e}")
                return []

        # Search all shards in parallel
        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            shard_futures = []
            for shard_id in range(self.num_shards):
                future = executor.submit(search_shard, shard_id)
                shard_futures.append(future)

            # Collect results from all shards
            for future in shard_futures:
                try:
                    shard_results = future.result(timeout=30)  # 30 second timeout
                    all_results.extend(shard_results)
                except Exception as e:
                    print(f"❌ Shard search failed: {e}")

        # Sort all results by similarity score and return top k
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_results[:k]

    def get_shard_stats(self) -> Dict:
        """Get statistics for each shard - ENHANCEMENT 3"""
        stats = {
            'total_shards': self.num_shards,
            'shard_details': {},
            'total_documents': 0
        }

        for shard_id, shard_data in self.shards.items():
            try:
                doc_count = shard_data.get('document_count', 0)
                stats['shard_details'][f'shard_{shard_id}'] = {
                    'document_count': doc_count,
                    'has_collection': shard_data['collection'] is not None
                }
                stats['total_documents'] += doc_count
            except Exception as e:
                stats['shard_details'][f'shard_{shard_id}'] = {
                    'error': str(e)
                }

        return stats



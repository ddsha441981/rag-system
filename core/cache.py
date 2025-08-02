import redis
import pickle
import hashlib
import sqlite3
import time
import os
from typing import Optional, Dict
from datetime import datetime, timedelta

try:
    import diskcache

    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    print("⚠️ diskcache not installed. Install with: pip install diskcache")


class MultiLevelCache:
    """
    Multi-level cache implementation - ENHANCEMENT 2
    Level 1: Memory (fastest)
    Level 2: Redis (fast, shared)
    Level 3: Disk (persistent, large capacity)
    Level 4: SQLite (structured, persistent)
    """

    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, ttl_hours=24):
        self.ttl_seconds = ttl_hours * 3600

        # Level 1: Memory cache (fastest)
        self.memory_cache = {}
        self.memory_cache_timestamps = {}
        self.memory_cache_max_size = 1000

        # Level 2: Redis cache (fast, shared)
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
            self.redis_client.ping()
            self.redis_available = True
            print("✅ Redis cache connected")
        except Exception as e:
            self.redis_available = False
            print(f"⚠️ Redis not available: {e}. Using local cache only.")

        # Level 3: Disk cache (persistent, large capacity)
        if DISKCACHE_AVAILABLE:
            try:
                os.makedirs('./data/cache', exist_ok=True)
                self.disk_cache = diskcache.Cache('./data/cache', size_limit=1024 * 1024 * 500)  # 500MB
                self.disk_available = True
                print("✅ Disk cache initialized")
            except Exception as e:
                self.disk_available = False
                print(f"⚠️ Disk cache error: {e}")
        else:
            self.disk_available = False
            self.disk_cache = None

        # Level 4: SQLite cache (structured, persistent)
        self.sqlite_available = self._init_sqlite_cache()

    def _init_sqlite_cache(self) -> bool:
        """Initialize SQLite cache - ENHANCEMENT 2"""
        try:
            os.makedirs('./data/cache', exist_ok=True)
            self.sqlite_path = './data/cache/query_cache.db'

            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS query_cache
                           (
                               key
                               TEXT
                               PRIMARY
                               KEY,
                               value
                               BLOB,
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               expires_at
                               TIMESTAMP
                           )
                           ''')
            conn.commit()
            conn.close()
            print("✅ SQLite cache initialized")
            return True
        except Exception as e:
            print(f"⚠️ SQLite cache error: {e}")
            return False

    def _generate_cache_key(self, query: str, context_hash: str = None) -> str:
        """Generate cache key from query and context - ENHANCEMENT 2"""
        cache_input = f"{query.lower().strip()}"
        if context_hash:
            cache_input += f"_{context_hash}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def get_cached_response(self, query: str, context_hash: str = None) -> Optional[Dict]:
        """Multi-level cache retrieval - ENHANCEMENT 2"""
        cache_key = self._generate_cache_key(query, context_hash)

        # Level 1: Memory cache
        if cache_key in self.memory_cache:
            if time.time() - self.memory_cache_timestamps[cache_key] < 300:  # 5 minutes
                return self.memory_cache[cache_key]
            else:
                del self.memory_cache[cache_key]
                del self.memory_cache_timestamps[cache_key]

        # Level 2: Redis cache
        if self.redis_available:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    self._store_in_memory(cache_key, data)  # Promote to Level 1
                    return data
            except Exception as e:
                print(f"Redis get error: {e}")

        # Level 3: Disk cache
        if self.disk_available and cache_key in self.disk_cache:
            try:
                data = self.disk_cache[cache_key]
                self._store_in_memory(cache_key, data)  # Promote to Level 1
                return data
            except Exception as e:
                print(f"Disk cache get error: {e}")

        # Level 4: SQLite cache
        if self.sqlite_available:
            data = self._get_from_sqlite(cache_key)
            if data:
                self._store_in_memory(cache_key, data)  # Promote to Level 1
                return data

        return None

    def cache_response(self, query: str, response: Dict, context_hash: str = None):
        """Multi-level cache storage - ENHANCEMENT 2"""
        cache_key = self._generate_cache_key(query, context_hash)

        # Store in all available levels
        self._store_in_memory(cache_key, response)

        if self.redis_available:
            try:
                self.redis_client.setex(cache_key, self.ttl_seconds, pickle.dumps(response))
            except Exception as e:
                print(f"Redis storage error: {e}")

        if self.disk_available:
            try:
                self.disk_cache[cache_key] = response
            except Exception as e:
                print(f"Disk cache storage error: {e}")

        if self.sqlite_available:
            self._store_in_sqlite(cache_key, response)

    def _store_in_memory(self, key: str, value: Dict):
        """Store in memory with size management - ENHANCEMENT 2"""
        if len(self.memory_cache) >= self.memory_cache_max_size:
            # Remove oldest item
            oldest_key = min(self.memory_cache_timestamps, key=self.memory_cache_timestamps.get)
            del self.memory_cache[oldest_key]
            del self.memory_cache_timestamps[oldest_key]

        self.memory_cache[key] = value
        self.memory_cache_timestamps[key] = time.time()

    def _get_from_sqlite(self, key: str) -> Optional[Dict]:
        """Get from SQLite cache - ENHANCEMENT 2"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT value FROM query_cache WHERE key = ? AND expires_at > datetime("now")',
                (key,)
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                return pickle.loads(result[0])
            return None
        except Exception as e:
            print(f"SQLite get error: {e}")
            return None

    def _store_in_sqlite(self, key: str, value: Dict):
        """Store in SQLite cache - ENHANCEMENT 2"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            expires_at = datetime.now() + timedelta(seconds=self.ttl_seconds)
            cursor.execute(
                '''INSERT OR REPLACE INTO query_cache (key, value, expires_at) 
                   VALUES (?, ?, ?)''',
                (key, pickle.dumps(value), expires_at)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"SQLite store error: {e}")

    def clear_cache(self):
        """Clear all cache levels - ENHANCEMENT 2"""
        # Clear memory
        self.memory_cache.clear()
        self.memory_cache_timestamps.clear()

        # Clear Redis
        if self.redis_available:
            try:
                self.redis_client.flushdb()
                print("✅ Redis cache cleared")
            except Exception as e:
                print(f"Redis clear error: {e}")

        # Clear disk
        if self.disk_available:
            try:
                self.disk_cache.clear()
                print("✅ Disk cache cleared")
            except Exception as e:
                print(f"Disk cache clear error: {e}")

        # Clear SQLite
        if self.sqlite_available:
            try:
                conn = sqlite3.connect(self.sqlite_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM query_cache')
                conn.commit()
                conn.close()
                print("✅ SQLite cache cleared")
            except Exception as e:
                print(f"SQLite clear error: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics - ENHANCEMENT 2"""
        return {
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_available,
            'disk_available': self.disk_available,
            'sqlite_available': self.sqlite_available,
            'total_cache_levels': sum([
                1,  # Memory always available
                1 if self.redis_available else 0,
                1 if self.disk_available else 0,
                1 if self.sqlite_available else 0
            ])
        }


QueryCache = MultiLevelCache

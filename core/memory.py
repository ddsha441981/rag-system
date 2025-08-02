from datetime import datetime
from typing import Dict, List


class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.conversations = {}
        self.max_history = max_history

    def add_conversation(self, session_id: str, query: str, response: str, context: List[Dict]):
        """Add conversation to memory"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'context': context
        }

        self.conversations[session_id].append(conversation_entry)

        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]

    def get_conversation_context(self, session_id: str, include_last_n: int = 3) -> str:
        """Get recent conversation context"""
        if session_id not in self.conversations:
            return ""

        recent_conversations = self.conversations[session_id][-include_last_n:]
        context_text = ""

        for conv in recent_conversations:
            context_text += f"Previous Q: {conv['query']}\nPrevious A: {conv['response']}\n\n"

        return context_text

    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]

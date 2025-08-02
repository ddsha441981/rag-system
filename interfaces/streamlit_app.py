import streamlit as st
import requests
import tempfile
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List


class StreamlitRAGInterface:
    def __init__(self, api_base_url: str = "http://localhost:8000/api/v1"):
        self.api_base_url = api_base_url

    def run(self):
        st.set_page_config(
            page_title="Enhanced RAG System",
            page_icon="🤖",
            layout="wide"
        )

        st.title("🤖 Enhanced RAG System")
        st.markdown("Upload documents and ask questions with AI-powered answers")

        # Initialize session state
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")

            # Model selection
            search_mode = st.selectbox(
                "Search Mode",
                ["hybrid", "semantic", "keyword"],
                help="Hybrid combines semantic and keyword search with re-ranking"
            )

            # Search parameters
            num_results = st.slider("Number of results", 1, 20, 5)

            # System stats
            self._display_system_stats()

            # Clear cache button
            if st.button("🗑️ Clear Cache"):
                self._clear_cache()

            # Clear session
            if st.button("🧹 Clear Session"):
                self._clear_session()

        # Main content
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📄 Document Upload")
            self._document_upload_section()

        with col2:
            st.header("❓ Ask Questions")
            self._query_section(search_mode, num_results)

        # Conversation History
        if st.session_state.conversation_history:
            st.header("💬 Conversation History")
            self._display_conversation_history()

    def _document_upload_section(self):
        """Document upload interface"""
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'html', 'json'],
            help="Supported formats: PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON"
        )

        if uploaded_files:
            if st.button("📚 Process Documents", type="primary"):
                self._process_documents(uploaded_files)

    def _process_documents(self, uploaded_files):
        """Process uploaded documents"""
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        files_data = []
        for file in uploaded_files:
            files_data.append(('files', (file.name, file.getvalue(), file.type)))

        try:
            response = requests.post(
                f"{self.api_base_url}/upload",
                files=files_data,
                timeout=300
            )

            if response.status_code == 200:
                results = response.json()

                success_count = 0
                total_chunks = 0

                for i, result in enumerate(results):
                    progress_bar.progress((i + 1) / len(results))

                    if result['success']:
                        success_count += 1
                        total_chunks += result['chunks_processed']
                        st.success(f"✅ {result['filename']}: {result['chunks_processed']} chunks")
                    else:
                        st.error(f"❌ {result['filename']}: {result['error']}")

                status_placeholder.success(
                    f"🎉 Processed {success_count}/{len(results)} files successfully. Total chunks: {total_chunks}"
                )
            else:
                st.error(f"Upload failed: {response.text}")

        except Exception as e:
            st.error(f"Error uploading files: {str(e)}")

    def _query_section(self, search_mode: str, num_results: int):
        """Query interface"""
        query = st.text_area(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            height=100
        )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🔍 Get Answer", type="primary") and query:
                self._process_query(query, search_mode, num_results)

        with col_btn2:
            if st.button("🎲 Example Questions"):
                self._show_example_questions()

    def _process_query(self, query: str, search_mode: str, num_results: int):
        """Process user query"""
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                response = requests.post(
                    f"{self.api_base_url}/query",
                    json={
                        "query": query,
                        "k": num_results,
                        "session_id": st.session_state.session_id,
                        "search_mode": search_mode
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    self._display_query_result(result)

                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'query': query,
                        'answer': result['answer'],
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'cached': result.get('cached', False),
                        'sources': result.get('sources', [])
                    })
                else:
                    st.error(f"Query failed: {response.text}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

    def _display_query_result(self, result: dict):
        """Display query results"""
        # Cache indicator
        if result.get('cached', False):
            st.info("📋 Answer retrieved from cache")

        # Main answer
        st.markdown("### 🎯 Answer")
        st.markdown(result['answer'])

        # Expandable sections
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("🔍 Answer Verification"):
                st.markdown(result.get('verification', 'No verification available'))

        with col2:
            with st.expander("📚 Sources"):
                sources = result.get('sources', [])
                if sources:
                    for source in set(sources):
                        st.markdown(f"- {source}")
                else:
                    st.markdown("No sources available")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Context Chunks", result.get('context_chunks', 0))
        with col2:
            st.metric("Tokens Used", result.get('tokens_used', 0))
        with col3:
            st.metric("Sources", len(set(sources)) if sources else 0)
        with col4:
            st.metric("Search Mode", result.get('search_mode', 'Unknown'))

    def _display_conversation_history(self):
        """Display conversation history"""
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
            cache_indicator = "📋" if conv.get('cached', False) else "🔍"

            with st.expander(
                    f"{cache_indicator} Q{len(st.session_state.conversation_history) - i}: {conv['query'][:50]}... ({conv['timestamp']})"):
                st.markdown(f"**Question:** {conv['query']}")
                st.markdown(f"**Answer:** {conv['answer']}")

                if conv.get('sources'):
                    st.markdown("**Sources:**")
                    for source in set(conv['sources']):
                        st.markdown(f"- {source}")

    def _display_system_stats(self):
        """Display system statistics"""
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()

                st.subheader("📊 System Stats")
                st.metric("Documents Indexed", stats.get('documents_indexed', 0))
                st.metric("Cache Available", "✅" if stats.get('cache_available', False) else "❌")
                st.metric("BM25 Index Size", stats.get('bm25_index_size', 0))
                st.metric("Active Sessions", stats.get('active_sessions', 0))
        except:
            st.warning("Could not load system stats")

    def _clear_cache(self):
        """Clear system cache"""
        try:
            response = requests.post(f"{self.api_base_url}/clear-cache", timeout=10)
            if response.status_code == 200:
                st.success("✅ Cache cleared successfully!")
            else:
                st.error("Failed to clear cache")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")

    def _clear_session(self):
        """Clear current session"""
        try:
            response = requests.delete(
                f"{self.api_base_url}/session/{st.session_state.session_id}",
                timeout=10
            )
            if response.status_code == 200:
                st.session_state.conversation_history = []
                st.success("✅ Session cleared successfully!")
            else:
                st.error("Failed to clear session")
        except Exception as e:
            st.error(f"Error clearing session: {str(e)}")

    def _show_example_questions(self):
        """Show example questions"""
        examples = [
            "What are the main topics discussed in the documents?",
            "Can you summarize the key findings?",
            "How do the different concepts relate to each other?",
            "What are the important dates and events mentioned?",
            "Compare the different approaches described."
        ]

        st.markdown("### 💡 Example Questions:")
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                st.text_area("Selected question:", value=example, key="selected_example")


def main():
    """Main Streamlit application"""
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

    interface = StreamlitRAGInterface(api_base_url)
    interface.run()


if __name__ == "__main__":
    main()

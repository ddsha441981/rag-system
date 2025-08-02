from typing import List, Dict


def _format_technical_context(context_chunks: List[Dict]) -> str:
    formatted = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk['metadata'].get('source', 'Unknown')
        content_type = chunk['metadata'].get('content_type', 'general')
        if content_type == 'method_documentation':
            header = f"[High Priority: Method {chunk['metadata'].get('method_name', 'unknown')}]"
        else:
            header = "[Supporting Information]"
        formatted += f"""
    --- Chunk {i} ---
    Source: {source}
    {header}
    
    {chunk['text']}
    """
    return formatted


def _build_method_explanation_prompt(query: str, context_chunks: List[Dict], methods: List[str],
                                     conversation_context: str) -> str:
    context_text = _format_technical_context(context_chunks)
    method = methods[0] if methods else "the method"

    prompt = f"""You are an AI specialized in technical documentation.

    Context:
    {context_text}
    
    User query: {query}
    
    Please provide a detailed explanation of {method} including:
    
    - Formal signature (if available).
    - Purpose and behavior (blocking vs. non-blocking).
    - Category type (synchronization, timing, etc.).
    - Requirements or preconditions.
    - Typical usage and side effects.
    
    {conversation_context}
    
    Answer:"""

    return prompt


def _build_method_comparison_prompt(query: str, context_chunks: List[Dict], methods: List[str],
                                    conversation_context: str) -> str:
    context_text = _format_technical_context(context_chunks)
    methods_str = ', '.join(methods) if methods else "the methods"

    prompt = f"""You are a senior software engineer providing a highly accurate technical comparison.

    Context:
    {context_text}
    
    User query: {query}
    
    Please provide a detailed comparison of {methods_str}:
    
    For each method:
    
    - Describe the purpose.
    - Explain the behavior (blocking vs. non-blocking, synchronization vs. scheduling).
    - Describe typical use cases.
    - Identify its category (e.g., synchronization primitive, scheduling hint, etc.).
    
    Finally, summarize the key differences in behavior and usage.
    
    {conversation_context}
    
    Answer:"""

    return prompt


class PromptBuilder:
    def __init__(self):
        self.system_prompts = {
            'question': """You are a helpful AI assistant that answers questions based on provided context. 
            Use only the information from the provided context. 
            If the context doesn't have the answer, say so clearly.""",

            'summary': """You are a helpful AI assistant for summarization. 
            Create concise summaries preserving key information.""",

            'comparison': """You are a helpful AI assistant that compares concepts or items 
            based on provided information."""
        }

    def build_prompt(self, query: str, context_chunks: List[Dict], query_type: str = "question") -> str:
        system_prompt = self.system_prompts.get(query_type, self.system_prompts['question'])
        context_text = self._format_context(context_chunks)

        prompt = f"""System: {system_prompt}

        Context Information:
        {context_text}

        User Question:
        {query}

        Instructions:
        - Answer using ONLY information above.
        - If info insufficient, state that clearly.
        - Cite parts of the context where possible.

        Answer:"""
        return prompt

    def build_enhanced(self, query: str, context_chunks: List[Dict], query_type: str = "question",
                       conversation_context: str = "") -> str:
        system_prompt = self.system_prompts.get(query_type, self.system_prompts['question'])
        context_text = self._format_context(context_chunks)

        conv_section = f"\nPrevious conversation:\n{conversation_context}" if conversation_context else ""

        prompt = f"""System: {system_prompt}{conv_section}

        Context Information:
        {context_text}

        User Question:
        {query}

        Instructions:
        - Answer with info above and conversation history.
        - Cite relevant context.
        - If unable, state clearly.

        Answer:"""
        return prompt

    def _format_context(self, context_chunks: List[Dict]) -> str:
        formatted = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page_number', 'Unknown')
            similarity = chunk.get('similarity_score', 0)
            formatted += f"""
        --- Chunk {i} ---
        Source: {source} (Page: {page})
        Similarity Score: {similarity:.4f}

        {chunk['text']}
        """
        return formatted

    def build_verification_prompt(self, query: str, answer: str, context: str) -> str:
        prompt = f"""Please verify the accuracy of the answer below, based only on the context provided.

        Question:
        {query}
        
        Answer:
        {answer}
        
        Context:
        {context}
        
        Instructions:
        - Confirm accuracy or specify incorrect parts.
        - Rate accuracy on 1-10.
        - Suggest improvements if needed.
        
        Verification:"""
        return prompt



    def build_technical_prompt(self, query: str, context_chunks: List[Dict], query_classification: Dict,
                               conversation_context: str = "") -> str:
        subtype = query_classification.get('subtype')
        methods = query_classification.get('methods', []) or query_classification.get('methods_mentioned', [])
        requires_precision = query_classification.get('requires_precision', False)

        if subtype == 'method_comparison':
            return _build_method_comparison_prompt(query, context_chunks, methods, conversation_context)
        elif subtype == 'method_explanation':
            return _build_method_explanation_prompt(query, context_chunks, methods, conversation_context)
        elif subtype == 'synchronization':
            return self._build_synchronization_prompts(query, context_chunks, conversation_context)
        elif requires_precision:
            return self._build_high_precision_prompt(query, context_chunks, query_classification, conversation_context)
        else:
            return self._build_general_prompt(query, context_chunks, conversation_context)

    def _build_synchronization_prompts(self, query: str, context_chunks: List[Dict], conversation_context: str) -> str:
        context_text = _format_technical_context(context_chunks)

        prompt = f"""You are an expert in concurrency and parallel programming.

        Context:
        {context_text}
        
        User query: {query}
        
        Please provide a detailed explanation covering:
        
        - Core concepts of synchronization.
        - Blocking vs non-blocking behavior.
        - Typical patterns and usage.
        - Relations to other concurrency concepts.
        
        {conversation_context}
        
        Answer:"""

        return prompt

    def _build_high_precision_prompt(self, query: str, context_chunks: List[Dict], query_classification: Dict,
                                     conversation_context: str) -> str:
        context_text = _format_technical_context(context_chunks)
        lang = query_classification.get('programming_language') or 'the relevant programming language'

        prompt = f"""You are an expert technical assistant.

        Context:
        {context_text}
        
        User query: {query}
        
        Please provide a precise and accurate response, emphasizing factual correctness, correct terminology, and referencing documentation as applicable.
        
        Programming language context: {lang}
        
        {conversation_context}
        
        Answer:"""

        return prompt

    def _build_general_prompt(self, query: str, context_chunks: List[Dict], conversation_context: str) -> str:
        context_text = self._format_context(context_chunks)

        prompt = f"""System: Please answer the user's query with attention to technical detail based on the context.

        Context:
        {context_text}
        
        User query:
        {query}
        
        {conversation_context}
        
        Answer:"""

        return prompt


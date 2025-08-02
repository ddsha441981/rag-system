import re
import nltk
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np



def ensure_nltk_data():
    required_packages = ['punkt', 'punkt_tab', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=True)



ensure_nltk_data()


class SmartChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None



    def semantic_chunking(self, text: str, metadata: Dict) -> List[Dict]:

        try:

            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            print(f"NLTK tokenization failed: {e}, using simple sentence splitting")

            sentences = self._simple_sentence_split(text)

        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in sentences:
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'sentences': current_sentences,
                    'metadata': metadata,
                    'word_count': len(current_chunk.split())
                }
                chunks.append(chunk_data)

                overlap_text = ' '.join(current_sentences[-2:]) if len(current_sentences) > 2 else ""
                current_chunk = overlap_text + " " + sentence
                current_sentences = current_sentences[-2:] + [sentence] if len(current_sentences) > 2 else [sentence]
            else:
                current_chunk += " " + sentence
                current_sentences.append(sentence)

        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'sentences': current_sentences,
                'metadata': metadata,
                'word_count': len(current_chunk.split())
            })

        return chunks

    def _simple_sentence_split(self, text: str) -> List[str]:


        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def hierarchical_chunking(self, text: str) -> List[str]:

        patterns = [
            r'\n\s*#{1,6}\s+.+\n',
            r'\n\s*\d+\.\s+',
            r'\n\s*[A-Z][A-Z\s]+\n',
            r'\n\s*\n\s*\n'
        ]

        chunks = [text]
        for pattern in patterns:
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(re.split(pattern, chunk))
            chunks = [c.strip() for c in new_chunks if c.strip()]

        return chunks



    def technical_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        if self._is_technical_content(text):
            return self._technical_semantic_chunking(text, metadata)
        else:
            return self.semantic_chunking(text, metadata)

    def _is_technical_content(self, text: str) -> bool:

        technical_indicators = [

            r'\w+\s*\([^)]*\)\s*[{:]',

            r'\b(?:class|method|function|synchronized|thread|lock|mutex)\b',

            r'(?:public|private|static|void|return)\s+\w+',

            r'(?:algorithm|implementation|performance|complexity)'
        ]

        for pattern in technical_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _technical_semantic_chunking(self, text: str, metadata: Dict) -> List[Dict]:



        sections = self._split_by_technical_boundaries(text)

        chunks = []
        for section in sections:

            if self._contains_method_signature(section):

                method_chunk = self._create_method_documentation_chunk(section, metadata)
                chunks.append(method_chunk)
            else:

                standard_chunks = self.semantic_chunking(section, metadata)
                chunks.extend(standard_chunks)

        return chunks

    def _split_by_technical_boundaries(self, text: str) -> List[str]:



        boundaries = [
            r'\n\s*(?:public|private|protected|static)\s+\w+.*?\n(?=\s*[a-zA-Z])',
            r'\n\s*\w+\(\)\s*[:;]\s*\n',
            r'\n\s*(?:Example|Code|Implementation):\s*\n',
            r'\n\s*(?:\d+\.|\*|-)\s+',
        ]

        sections = [text]

        for boundary_pattern in boundaries:
            new_sections = []
            for section in sections:

                parts = re.split(f'({boundary_pattern})', section, flags=re.IGNORECASE)
                current_section = ""

                for i, part in enumerate(parts):
                    if re.match(boundary_pattern, part, re.IGNORECASE):
                        if current_section.strip():
                            new_sections.append(current_section.strip())
                        current_section = part
                    else:
                        current_section += part

                if current_section.strip():
                    new_sections.append(current_section.strip())

            sections = new_sections

        return [section for section in sections if len(section.strip()) > 20]

    def _contains_method_signature(self, text: str) -> bool:

        patterns = [
            r'\w+\s*\([^)]*\)\s*[{:]',
            r'\w+\(\)\s*[:;]',
            r'(?:public|private|static)\s+\w+',
            r'def\s+\w+\s*\([^)]*\)',
        ]

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _create_method_documentation_chunk(self, text: str, metadata: Dict) -> Dict:



        method_info = self._extract_method_info(text)

        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'content_type': 'method_documentation',
            'method_name': method_info.get('name'),
            'method_signature': method_info.get('signature'),
            'parameters': method_info.get('parameters', []),
            'return_type': method_info.get('return_type'),
            'is_technical': True,
            'programming_language': method_info.get('language', 'unknown')
        })

        return {
            'text': text,
            'metadata': enhanced_metadata,
            'word_count': len(text.split()),
            'technical_entities': method_info
        }

    def _extract_method_info(self, text: str) -> Dict:

        method_info = {
            'name': None,
            'signature': None,
            'parameters': [],
            'return_type': None,
            'language': 'unknown'
        }


        java_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)'
        java_match = re.search(java_pattern, text)

        if java_match:
            method_info.update({
                'return_type': java_match.group(1),
                'name': java_match.group(2),
                'signature': java_match.group(0),
                'parameters': [param.strip() for param in java_match.group(3).split(',') if param.strip()],
                'language': 'java'
            })
        else:
            method_pattern = r'(\w+)\s*\([^)]*\)'
            method_match = re.search(method_pattern, text)

            if method_match:
                method_info.update({
                    'name': method_match.group(1),
                    'signature': method_match.group(0)
                })

        return method_info


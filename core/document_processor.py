import json
from pathlib import Path
from typing import Dict, List

import docx
import fitz  # PyMuPDF
import markdown
import pandas as pd
from bs4 import BeautifulSoup


class UniversalDocumentProcessor:
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_txt,
            '.md': self._process_markdown,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.html': self._process_html,
            '.json': self._process_json
        }

    def process_document(self, file_path: str) -> List[Dict]:
        """Process any supported document format"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")

        return self.supported_formats[file_ext](file_path)

    def _process_pdf(self, file_path: str) -> List[Dict]:
        """Process PDF files using PyMuPDF"""
        doc = fitz.open(file_path)
        chunks = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                chunks.append({
                    'text': text,
                    'page_number': page_num + 1,
                    'source': str(file_path),
                    'chunk_id': f"{Path(file_path).stem}_page_{page_num + 1}",
                    'format': 'pdf'
                })

        doc.close()
        return chunks

    def _process_docx(self, file_path: str) -> List[Dict]:
        """Process DOCX files"""
        doc = docx.Document(file_path)
        chunks = []

        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                chunks.append({
                    'text': para.text,
                    'paragraph_number': i + 1,
                    'source': str(file_path),
                    'chunk_id': f"{Path(file_path).stem}_para_{i + 1}",
                    'format': 'docx'
                })

        return chunks

    def _process_txt(self, file_path: str) -> List[Dict]:
        """Process text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        return [{
            'text': text,
            'source': str(file_path),
            'chunk_id': f"{Path(file_path).stem}_full",
            'format': 'txt'
        }]

    def _process_markdown(self, file_path: str) -> List[Dict]:
        """Process Markdown files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()

        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()

        return [{
            'text': text,
            'source': str(file_path),
            'chunk_id': f"{Path(file_path).stem}_md",
            'format': 'markdown',
            'original_markdown': md_content
        }]

    def _process_csv(self, file_path: str) -> List[Dict]:
        """Process CSV files"""
        df = pd.read_csv(file_path)

        text_content = f"CSV Data Summary:\nColumns: {', '.join(df.columns)}\nRows: {len(df)}\n\n"
        text_content += df.to_string(index=False)

        return [{
            'text': text_content,
            'source': str(file_path),
            'chunk_id': f"{Path(file_path).stem}_csv",
            'format': 'csv',
            'rows': len(df),
            'columns': list(df.columns)
        }]

    def _process_excel(self, file_path: str) -> List[Dict]:
        """Process Excel files"""
        excel_file = pd.ExcelFile(file_path)
        all_content = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text_content = f"Sheet: {sheet_name}\nColumns: {', '.join(df.columns)}\nRows: {len(df)}\n\n"
            text_content += df.to_string(index=False)

            all_content.append({
                'text': text_content,
                'source': str(file_path),
                'chunk_id': f"{Path(file_path).stem}_{sheet_name}",
                'format': 'excel',
                'sheet_name': sheet_name,
                'rows': len(df),
                'columns': list(df.columns)
            })

        return all_content

    def _process_html(self, file_path: str) -> List[Dict]:
        """Process HTML files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()

        return [{
            'text': text,
            'source': str(file_path),
            'chunk_id': f"{Path(file_path).stem}_html",
            'format': 'html'
        }]

    def _process_json(self, file_path: str) -> List[Dict]:
        """Process JSON files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        text_content = f"JSON Data:\n{json.dumps(json_data, indent=2)}"

        return [{
            'text': text_content,  # Fixed: text_content instead of text_
            'source': str(file_path),  # Fixed: Added missing comma above
            'chunk_id': f"{Path(file_path).stem}_json",
            'format': 'json',
            'original_data': json_data
        }]

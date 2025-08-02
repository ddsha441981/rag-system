import os
from groq import Groq
from typing import Dict


class GroqLLMHandler:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_answer(self, prompt: str, max_tokens: int = 1000) -> Dict:
        """Generate answer using Groq LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9
            )

            answer = response.choices[0].message.content

            return {
                'answer': answer,
                'model': self.model,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'success': True
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'success': False,
                'error': str(e)
            }

    def verify_answer(self, verification_prompt: str) -> Dict:
        """Verify answer accuracy"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": verification_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            verification = response.choices[0].message.content

            return {
                'verification': verification,
                'success': True
            }

        except Exception as e:
            return {
                'verification': f"Error in verification: {str(e)}",
                'success': False
            }

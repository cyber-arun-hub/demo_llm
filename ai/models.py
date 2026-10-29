import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class LLM:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    async def generate_answer(self, query: str, context: str) -> str:
        if not context:
            return "No context found to answer the query."
        if not self.client:
            return context  # Fallback to file chunk if no API key

        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer concisely based on the context:"
        try:
            response = self.client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return context  # Fallback to file chunk if API call fails

llm = LLM()

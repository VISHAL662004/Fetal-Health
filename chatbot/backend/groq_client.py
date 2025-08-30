from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

client = Groq(api_key=GROQ_API_KEY)

def ask_groq(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated to a supported model
            messages=[
                {"role": "system", "content": "You are a helpful assistant on fetal health. Answer questions based on the provided knowledge base with crisp and concise answers."},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API error: {e}"

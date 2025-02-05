import os
import time
from groq import Groq

class LLMClient_groq:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)

    def llm(self, text: str):
        
        try:
            chat_completion = self.client.chat.completions.create(
            max_tokens=5000,

            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
            model="llama3-8b-8192",
            temperature=0.01,
        )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(600)
            response=self.llm(text)
        
        return response


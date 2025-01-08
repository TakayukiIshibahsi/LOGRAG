import requests
import os
import time

class LLMClient:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        self.api_token = os.getenv("HF_API_TOKEN")  # 環境変数からAPIトークンを取得
        if not self.api_token:
            raise ValueError("Hugging Face API トークンが設定されていません。")
        
    def llm(self,query):
        parameters = {
            "max_new_tokens": 5000,
            "temperature": 0.01,
            "top_k": 50,
            "top_p": 0.95,
            "return_full_text": False
            }
        
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful and smart assistant. You accurately provide answer to the provided user query.<|eot_id|><|start_header_id|>user<|end_header_id|> Here is the query: ```{query}```.
            Provide precise and concise answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        prompt = prompt.replace("{query}", query)
        
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        while (response.status_code == 429):
            retry_after = int(response.headers.get("Retry-After", 100))  # デフォルトは1秒
            print(f"Rate limit hit. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            response = requests.post(self.api_url, headers=headers, json=payload)
        response_text = response.json()[0]['generated_text'].strip()

        return response_text
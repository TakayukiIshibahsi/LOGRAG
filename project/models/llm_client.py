import requests
import os

class LLMClient:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/bert-base-cased"
        self.api_token = os.getenv("HF_API_TOKEN")  # 環境変数からAPIトークンを取得

    def query(self, inputs: str):
        """
        モデルにクエリを送信し、結果を取得する
        """
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            response = requests.post(self.api_url, headers=headers, json={"inputs": inputs})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying the model: {e}")
            return None
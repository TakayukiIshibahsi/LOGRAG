from .llm_client import LLMClient

class Classifier:
    def __init__(self):
        self.client = LLMClient()

    def classify(self, text: str):
        inputs = f"分類してください: {text}"
        response = self.client.query(inputs)
        return response
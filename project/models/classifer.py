from .llm_client import LLMClient

class Classifier:
    def __init__(self):
        self.client = LLMClient()

    def predict(self, text: str):
        inputs = f"""
            Categories:
            1- World
            2- Sports
            3- Business
            4- Sci/Tech

            Please decide only one category for the following text:
            {text}
            """
        response = self.client.llm(inputs)
        return response
    

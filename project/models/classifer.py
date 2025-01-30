from .llm_client import LLMClient

class Classifier:
    def __init__(self):
        self.client = LLMClient()

    def predict(self, text: str,auto=True):
        if not text or len(text) == 0:
            raise ValueError("Input text is empty.")

        inputs = f"""
            Categories:
            1- World
            2- Sports
            3- Business
            4- Sci/Tech

            Please decide only one category for the following text and respond with the format:
            "Category: <number> - <name>"

            Text: {text}
            """
        if auto:
            response = self.client.llm(inputs)
        else:
            response = self.client.llm(text)
        return response
    

import re

class Extract_categories:
    def __init__(self):
        self.pattern = r'Category: (\d) - (\w+)'

    def extract(self,llm_response):
        if not llm_response or len(llm_response) == 0:
            return "0"
        match = re.search(self.pattern, llm_response)
        if not match:
            print(llm_response)
            return "0"
            raise ValueError(f"No category found:{llm_response}")
        num = match.group(1)
        
        return num
import re

class Extract_categories:
    def __init__(self):
        self.pattern = r'Category: (\d) - (\w+)'
        self.pattern2 = r"Category: (\d)"

    def extract(self,llm_response):
        if not llm_response or len(llm_response) == 0:
            return "0"
        match = re.search(self.pattern, llm_response)
        match2 = re.search(self.pattern2, llm_response)
        if not match and not match2:
            print(llm_response)
            return "0"
            raise ValueError(f"No category found:{llm_response}")
        elif match:
            num = match.group(1)
        else:
            num = match2.group(1)
        
        return num
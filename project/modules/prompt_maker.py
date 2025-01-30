class Prompt_maker:
    def __init__(self,category,text):
        self.category = category
        self.text = text
        self.class_dict = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

    def get_prompt(self,target_article):
        prompt = f"""
        Categories:
            1- World
            2- Sports
            3- Business
            4- Sci/Tech

        Example:
        Article: {self.text}
        Category: {self.category}- {self.class_dict[self.category]}

        Please decide only one category for the following text and respond with the format:
            "Category: <number> - <name>"
            
        Article: {target_article}
        """

        return prompt
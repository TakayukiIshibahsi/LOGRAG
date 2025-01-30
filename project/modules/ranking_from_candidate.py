from modules.updater import Updater

class Ranking_from_candidate:
    def __init__(self, candidates, num_classes):
        self.leraning_rate = 0.1
        self.candidates = candidates
        self.num_classes = num_classes
        self.ranking_dict = {i: 0 for i in range(1, num_classes + 1)}

    def weight_rank(self):
        for k, candidate in enumerate(self.candidates):
            metadata = candidate["metadata"]
            class_index = metadata["Class Index"]
            self.ranking_dict[class_index] += 1 * 1 / (1 + k) * candidate["distance"]*metadata["reliability"]
        
        top_class = max(self.ranking_dict, key=self.ranking_dict.get)
        return top_class

    def rank(self):
        for candidate in self.candidates:
            class_index = candidate["metadata"]["Class Index"]
            self.ranking_dict[class_index] += candidate["distance"]
        
        top_class = max(self.ranking_dict, key=self.ranking_dict.get)
        return top_class
    
    def weight_modify(self,true_class):
        for k, candidate in enumerate(self.candidates):
            metadata = candidate["metadata"]
            class_index = metadata["Class Index"]
            if class_index == true_class:
                metadata["reliability"] += 1 * self.leraning_rate * candidate["distance"]
            else:
                metadata["reliability"] -= 1 * self.leraning_rate * candidate["distance"]
        
        top_class = max(self.ranking_dict, key=self.ranking_dict.get)
        return top_class
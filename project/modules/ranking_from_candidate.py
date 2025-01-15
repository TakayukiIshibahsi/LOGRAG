class Ranking_from_candidate:
    def __init__(self, candidates, num_classes):
        self.candidates = candidates
        self.num_classes = num_classes
        # クラス数に基づいて辞書を動的に初期化
        self.ranking_dict = {i: 0 for i in range(1, num_classes + 1)}

    def weight_rank(self):
        for k, candidate in enumerate(self.candidates):
            metadata = candidate["metadata"]
            class_index = metadata["Class Index"]
            self.ranking_dict[class_index] += 1 * 1 / (1 + k) * candidate["distance"]
        
        # 最大スコアを持つクラスを決定
        top_class = max(self.ranking_dict, key=self.ranking_dict.get)
        return top_class

    def rank(self):
        for candidate in self.candidates:
            class_index = candidate["metadata"]["Class Index"]
            self.ranking_dict[class_index] += candidate["distance"]
        
        # 最大スコアを持つクラスを決定
        top_class = max(self.ranking_dict, key=self.ranking_dict.get)
        return top_class
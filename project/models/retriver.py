from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self):
        # 埋め込み生成モデル（例：BERTベース）
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def retrieve(self, query: str, candidates: list):
        """
        クエリと候補データを埋め込み空間にマップし、特徴ベクトルを返す
        """
        # クエリと候補文の埋め込みを取得
        query_embedding = self.model.encode([query])
        candidate_embeddings = self.model.encode(candidates)

        # 類似度を計算して、特徴量ベクトルを返す
        similarities = self.calculate_similarity(query_embedding, candidate_embeddings)
        return similarities, candidate_embeddings

    def calculate_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray):
        """
        コサイン類似度を計算
        """
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(query_embedding, candidate_embeddings)
        return similarity.flatten()  # 類似度スコアの1次元リストを返す

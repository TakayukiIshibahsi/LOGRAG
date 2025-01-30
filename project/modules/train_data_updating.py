import pandas as pd
from vectordb import Memory
from modules.updater import Updater
import pickle
import os

class Train_data_updating:
    def __init__(self, file_path: str = "data/normal_vectordb.pkl", rows_to_load: int = None):
        """
        :param file_path: 保存するVectorDBファイルのパス
        :param rows_to_load: CSVファイルから読み込む行数（Noneの場合は全行を読み込む）
        """
        self.file_path = file_path
        self.rows_to_load = rows_to_load
        
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        if not os.path.exists(self.file_path):
            dummy_data = [{"memory": [], "metadata": [{}]}]
            with open(self.file_path, "wb") as file:
                pickle.dump(dummy_data, file)
            print(f"Dummy pickle file created at: {self.file_path}")

        self.df = pd.read_csv('data/raw/train.csv', nrows=self.rows_to_load)
        self.memory = Memory(embeddings='best', memory_file=self.file_path)
        self.updater = Updater(self.memory)

    def update(self, threshold: int = 50):
        """
        データをVectorDBにアップデート
        :param threshold: 1回のアップデートで処理するデータ数
        """
        metadatas = []
        texts = []
        count = 0

        for text, title, expected in zip(self.df['Description'], self.df['Title'], self.df['Class Index']):
            metadata = {"Title": title, "Class Index": expected,"reliability": 1}
            metadatas.append(metadata)
            texts.append(text)
            count += 1

            # 閾値に達した場合はアップデートを実行
            if count == threshold:
                self.updater.update(texts, metadatas)
                count = 0
                texts = []
                metadatas = []

        # 残ったデータもアップデート
        if texts and metadatas:
            self.updater.update(texts, metadatas)


if __name__ == "__main__":
    # 読み込む行数を動的に指定
    rows_to_update = 60000
    file_path = "data/custom_vectordb_half.pkl"

    updater = Train_data_updating()
    updater.update(threshold=50)

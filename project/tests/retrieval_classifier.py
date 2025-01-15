from vectordb import Memory
import pandas as pd 
import csv,os
import sys
print(sys.path)
from modules.ranking_from_candidate import Ranking_from_candidate

df = pd.read_csv('data/raw/test.csv')
log_file = "retrieval_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "pred", "expected"])
        writer.writeheader()  # ヘッダー行を追加
        print(f"Initialized new CSV file: {log_file}")

class Retrieval_classifier:
    def __init__(self):
        self.memory = Memory(embeddings='best',memory_file='data/normal_vectordb.pkl')
    
    def classify(self, rank=1):
      true_num = 0
      false_num = 0
      log_df = pd.read_csv(log_file)
      start=len(log_df)-1
      for k,(data, expected) in enumerate(zip(df['Description'], df['Class Index'])):
          if(k>start):
              candidates=self.memory.search(data,rank)
              rfm = Ranking_from_candidate(candidates, 4)
              num = rfm.rank()
              log_df = pd.DataFrame([[k+1, num, expected]], columns=["id", "pred", "expected"])
              log_df.to_csv(log_file, mode="a", header=False, index=False)
              # 予測結果を比較
              pred = str(num)  # 予測結果を文字列に変換
              print(f"pred:{pred} expected:{expected}")
              print(f"True: {true_num}, False: {false_num}")
              if pred == str(expected):
                  true_num += 1
              else:
                  false_num += 1

      print(f"True: {true_num}, False: {false_num}")


if __name__ == "__main__":
    retrieval_classifier=Retrieval_classifier()
    retrieval_classifier.classify()
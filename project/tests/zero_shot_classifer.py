import pandas as pd
from models.classifer import Classifier
from modules.extract_categories import Extract_categories
import csv,os

df = pd.read_csv('data/raw/test.csv')
log_file = "log_groq.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "pred", "expected"])
        writer.writeheader()  # ヘッダー行を追加
        print(f"Initialized new CSV file: {log_file}")

class ZeroShotClassifier:
    def __init__(self):
      self.classifier = Classifier()
      self.extract_categories = Extract_categories()

    def classify(self):
      true_num = 0
      false_num = 0
      log_df = pd.read_csv(log_file)
      start=len(log_df)-1
      for k,(data, expected) in enumerate(zip(df['Description'], df['Class Index'])):
          if(k>start):
              # 分類予測とカテゴリ抽出
              num = self.extract_categories.extract(self.classifier.predict(data))
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
    classifier = ZeroShotClassifier()
    classifier.classify()
        
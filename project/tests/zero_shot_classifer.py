import pandas as pd
from models.classifer import Classifier
from modules.extract_categories import Extract_categories

df = pd.read_csv('data/raw/test.csv')

class ZeroShotClassifier:
    def __init__(self):
      self.classifier = Classifier()
      self.extract_categories = Extract_categories()

    def classify(self):
      true_num = 0
      false_num = 0

      for (data, expected) in zip(df['Description'], df['Class Index']):
          # 分類予測とカテゴリ抽出
          num = self.extract_categories.extract(self.classifier.predict(data))
          print("log:", num)

          # 予測結果を比較
          pred = str(num)  # 予測結果を文字列に変換
          print(f"pred:{pred} expected:{expected}")
          print(f"True: {true_num}, False: {false_num}")
          if pred == expected:
              true_num += 1
          else:
              false_num += 1

      print(f"True: {true_num}, False: {false_num}")


if __name__ == "__main__":
    classifier = ZeroShotClassifier()
    classifier.classify()
        
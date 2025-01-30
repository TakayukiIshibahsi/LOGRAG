from vectordb import Memory
import pandas as pd
from models.classifer import Classifier
from modules.extract_categories import Extract_categories
from modules.prompt_maker import Prompt_maker
import csv,os


df = pd.read_csv('data/raw/test.csv')
log_file = "log/vdbllm_detail_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "rank","retrieve_class","pred", "expected"])
        writer.writeheader()  # ヘッダー行を追加
        print(f"Initialized new CSV file: {log_file}")


class vdbllm:
    def __init__(self):
        self.memory = Memory(embeddings='best',memory_file='data/normal_vectordb.pkl')
        self.extract_categories = Extract_categories()
        self.classifier = Classifier()
    
    def classify(self):
      rank = 8
      true_num = 0
      false_num = 0
      log_df = pd.read_csv(log_file)
      start=len(log_df)-1
      for k,(data, expected) in enumerate(zip(df['Description'], df['Class Index'])):
          if(k>start):
              
              pred_dict={'1':0,'2':0,'3':0,'4':0}
              candidates=self.memory.search(data,rank)
              for l,candidate in enumerate(candidates):
                  candidate_class = candidate['metadata']['Class Index']
                  prompt = Prompt_maker(candidate_class,candidate['chunk']).get_prompt(data)
                  num = self.extract_categories.extract(self.classifier.predict(prompt,False))
                  log_df = pd.DataFrame([[k+1,l+1,candidate_class,num, expected]], columns=["id", "pred", "expected"])
                  log_df.to_csv(log_file, mode="a", header=False, index=False)
                  if num in pred_dict.keys():
                      pred_dict[num]+=candidate['distance']
                  
              max_key = max(pred_dict, key=pred_dict.get)
              
              # 予測結果を比較
              pred = str(max_key)  # 予測結果を文字列に変換
              print(f"pred:{pred} expected:{expected}")
              print(f"True: {true_num}, False: {false_num}")
              if pred == str(expected):
                  true_num += 1
              else:
                  false_num += 1

      print(f"True: {true_num}, False: {false_num}")

if __name__ == "__main__":
    vdbllm = vdbllm()
    vdbllm.classify()
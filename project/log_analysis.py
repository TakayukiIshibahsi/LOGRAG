import pandas as pd

def log_analysis(log_file):
    df = pd.read_csv(log_file)
    true_num = 0
    false_num = 0
    for pred,expected in zip(df['pred'],df['expected']):
        if pred == expected:
            true_num += 1
        else:
            false_num += 1
    print(f"True: {true_num}, False: {false_num}, Accuracy: {true_num/(true_num+false_num)}")

log_analysis("log.csv")
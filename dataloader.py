import pandas as pd

def load_data_from_txt_to_pd(pos_path, neg_path):
    with open(pos_path, 'r') as f:
        pos_lines = f.readlines()
        pos_lines = [line.split for line in pos_lines]

    with open(pos_path, 'r') as f:
        neg_lines = f.readlines()
        pos_lines = [line.split for line in neg_lines]


    data = pd.DataFrame({
        "acids": pos_lines,
        "isPositive": [True] * len(pos_lines)
    })
    print(data)

if __name__ == "__main__":
    load_data_from_txt_to_pd("neg_A0201.txt", "data")
import pandas as pd

def load_data_from_txt_to_pd(file_path, data_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    data = pd.DataFrame(lines)
    print(data)

if __name__ == "__main__":
    load_data_from_txt_to_pd("neg_A0201.txt", "data")
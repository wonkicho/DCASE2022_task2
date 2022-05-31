import pandas as pd
import os

root_path = "D:\dev_data2\ToyTrain"
csv_list = ["attributes_00.csv", "attributes_01.csv", "attributes_02.csv"]

attr = []
file_path_list = []
for c in csv_list:
    csv_path = os.path.join(root_path, c)
    df = pd.read_csv(csv_path)
    
    for v in df["d1v"].unique():
        attr.append(v)
    
    for p in df["file_name"].values():
        fp = os.path.join(root_path, p)
        file_path_list.append(fp)
    
print([i for i in range(len(attr))])
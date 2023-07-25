import os
import pickle

import faiss

path = "crs"
indexs_file_path = f"checkpoints/{path}/feature_and_index.pkl"
indexs_out_dir = f"checkpoints/{path}/"

with open("feature_and_index.pkl",mode="rb") as f:
    indexs = pickle.load(f)

for k in indexs:
    print(f"Save {k} index")
    faiss.write_index(
        indexs[k],
        os.path.join(indexs_out_dir,f"Index-{k}.index")
    )

print("Saved all index")
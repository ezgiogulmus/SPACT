import argparse
import numpy as np
import pickle
import os
import h5py
import sys
import pandas as pd
from sklearn.cluster import KMeans


args = argparse.ArgumentParser()
args.add_argument("data_name")
args.add_argument("--dataset_dir", type=str, default="./datasets_csv/")
args.add_argument("--target_size", type=int, default=10)
args.add_argument("--seed", type=int, default=0)

args = args.parse_args()
if os.path.isfile(os.path.join(args.dataset_dir, args.data_name+"_cluster_ids", f"k{args.target_size}_labels.pkl")):
	print("Cluster IDs already saved")
	sys.exit(0)
df = pd.read_csv(os.path.join(args.dataset_dir, args.data_name+".csv"))
cancer_type = args.data_name.rsplit("_", 1)[0].upper()
patch_dir = f"/media/nfs/SURV/{cancer_type}/SP1024/patches/"
cluster_ids = {}
for slide in df["slide_id"].values:
	with h5py.File(os.path.join(patch_dir, slide+".h5"), 'r') as f:
		coords = np.array(f['coords'])
	k = max(1, len(coords) // args.target_size)
	kmeans = KMeans(n_clusters=k, random_state=args.seed).fit(coords)
	cluster_ids[slide] = kmeans.labels_
os.makedirs(os.path.join(args.dataset_dir, args.data_name+"_cluster_ids"), exist_ok=True)
with open(os.path.join(args.dataset_dir, args.data_name+"_cluster_ids", f"k{args.target_size}_labels.pkl"), "wb") as f:
    pickle.dump(cluster_ids, f)
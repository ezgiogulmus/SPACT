import os
import pdb
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('data_name', type=str)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--dataset_dir', type=str, default="./datasets_csv")
parser.add_argument('--split_dir', type=str, default='./splits')
parser.add_argument('--k', type=int, default=5, help='number of splits (default: 5)')
parser.add_argument('--val_frac', type=float, default= 0.2, help='fraction of labels for validation (default: 0.2)')
parser.add_argument('--test_frac', type=float, default= 0.2, help='fraction of labels for test (default: 0.2)')
parser.add_argument('--stratify_col', default="time")
parser.add_argument('--time_col', type=str, default="survival_months")
parser.add_argument('--event_col', type=str, default="event")
parser.add_argument('--seed', type=int, default=58, help='random seed (default: 58)')
parser.add_argument('--verbose', action="store_true", default=False)

args = parser.parse_args()

split_dir = os.path.join(args.split_dir, args.data_name)
os.makedirs(split_dir)

args.csv_path = os.path.join(args.dataset_dir, args.data_name + ".csv") if args.csv_path is None else args.csv_path
df = pd.read_csv(args.csv_path, compression="zip" if ".zip" in args.csv_path else None)
case_df = df.drop_duplicates("case_id").reset_index(drop=True)

print("Survival Times:")
print(case_df[args.time_col].describe())

_, time_breaks = pd.qcut(case_df[args.time_col], q=4, retbins=True, labels=False)
time_breaks[0] = 0
time_breaks[-1] += 1
disc_labels, _ = pd.cut(case_df[args.time_col], bins=time_breaks, retbins=True, labels=False, right=False, include_lowest=True)
case_df.insert(2, 'time', disc_labels.values.astype(int))
df = pd.merge(case_df.drop("slide_id", axis=1), df[["slide_id", "case_id"]], on="case_id", how="left")

if args.verbose:
    print("Time intervals: ", time_breaks)
    print("Number of samples per time interval: ")
    print(case_df["time"].value_counts().sort_index())
    print("Number of slides per time interval: ")
    print(df["time"].value_counts().sort_index())
    print()

strats = []
if args.stratify_col not in ["None", "none", None]:
    for i in args.stratify_col.split(","):
        strats.append(i)
    strats.append(args.event_col)
# pdb.set_trace()
print("Stratification columns: ", strats)

len(case_df)
val_size = int(len(case_df) * args.val_frac)
test_size = int(len(case_df) * args.test_frac)
train_size = len(case_df) - val_size - test_size
print("Train size: {} ({:.2f}%) | Val size: {} ({:.2f}%) | Test size: {} ({:.2f}%)" .format(train_size, 100*train_size/len(case_df), val_size, 100*val_size/len(case_df), test_size, 100*test_size/len(case_df)))

if test_size > 0:
    try:
        dev_set, test_set = train_test_split(case_df, test_size=test_size, stratify=None if strats == [] else case_df[strats], random_state=args.seed)
    except ValueError:
        print("Not enough samples for test set. Using all samples for test set.")
        strats.remove(args.event_col)
        dev_set, test_set = train_test_split(case_df, test_size=test_size, stratify=None if strats == [] else case_df[strats], random_state=args.seed)
    dev_set.reset_index(inplace=True, drop=True)
else:
    dev_set = case_df

if args.verbose:
    print("############################################")
for k in range(args.k):
    train_set, val_set = train_test_split(dev_set, test_size=val_size, stratify=None if strats == [] else dev_set[strats], random_state=args.seed+k)
    if test_size > 0:
        split_df = pd.concat([
            train_set[["case_id"]].rename(columns={"case_id": "train"}).reset_index(drop=True),
            val_set[["case_id"]].rename(columns={"case_id": "val"}).reset_index(drop=True),
            test_set[["case_id"]].rename(columns={"case_id": "test"}).reset_index(drop=True)
        ], axis=1)
    else:
        split_df = pd.concat([
            train_set[["case_id"]].rename(columns={"case_id": "train"}).reset_index(drop=True),
            val_set[["case_id"]].rename(columns={"case_id": "val"}).reset_index(drop=True)
        ], axis=1)

    train_slides = df[df["case_id"].isin(train_set["case_id"].values)]
    val_slides = df[df["case_id"].isin(val_set["case_id"].values)]
    if test_size > 0:
        test_slides = df[df["case_id"].isin(test_set["case_id"].values)]
    if args.verbose:
        print(f"\nFold: {k}")
        print("\nSurvival time:")
        print("Train:")
        for i in sorted(pd.unique(case_df["time"])):
            print("\tClass: {} | cases: {}({:.2f}%) | slides {}({:.2f}%)" .format(i, (train_set["time"]==i).sum(), 100*(train_set["time"]==i).sum()/len(train_set), (train_slides["time"]==i).sum(), 100*(train_slides["time"]==i).sum()/len(train_slides)))
        print("\nVal:")
        for i in sorted(pd.unique(case_df["time"])):
            print("\tClass: {} | cases: {}({:.2f}%) | slides {}({:.2f}%)" .format(i, (val_set["time"]==i).sum(), 100*(val_set["time"]==i).sum()/len(val_set), (val_slides["time"]==i).sum(), 100*(val_slides["time"]==i).sum()/len(val_slides)))
        if test_size > 0:
            print("\nTest:")
            for i in sorted(pd.unique(case_df["time"])):
                print("\tClass: {} | cases: {}({:.2f}%) | slides {}({:.2f}%)" .format(i, (test_set["time"]==i).sum(), 100*(test_set["time"]==i).sum()/len(test_set), (test_slides["time"]==i).sum(), 100*(test_slides["time"]==i).sum()/len(test_slides)))
        
        print("\nEvent:")
        print("Train:")
        for i in sorted(pd.unique(case_df[args.event_col])):
            print("\tClass: {} | cases: {}({:.2f}%) | slides {}({:.2f}%)" .format(i, (train_set[args.event_col]==i).sum(), 100*(train_set[args.event_col]==i).sum()/len(train_set), (train_slides[args.event_col]==i).sum(), 100*(train_slides[args.event_col]==i).sum()/len(train_slides)))
        print("\nVal:")
        for i in sorted(pd.unique(case_df[args.event_col])):
            print("\tClass: {} | cases: {}({:.2f}%) | slides {}({:.2f}%)" .format(i, (val_set[args.event_col]==i).sum(), 100*(val_set[args.event_col]==i).sum()/len(val_set), (val_slides[args.event_col]==i).sum(), 100*(val_slides[args.event_col]==i).sum()/len(val_slides)))
        if test_size > 0:
            print("\nTest:")
            for i in sorted(pd.unique(case_df[args.event_col])):
                print("\tClass: {} | cases: {}({:.2f}%) | slides {}({:.2f}%)" .format(i, (test_set[args.event_col]==i).sum(), 100*(test_set[args.event_col]==i).sum()/len(test_set), (test_slides[args.event_col]==i).sum(), 100*(test_slides[args.event_col]==i).sum()/len(test_slides)))
    split_df.to_csv(os.path.join(split_dir, "splits_{}.csv".format(k)))

print("############################################\n\n")
base_train = None
for i in range(5):
    df = pd.read_csv(f"{split_dir}/splits_{i}.csv")
    if base_train is None:
        base_train = df["train"].dropna().values
        base_val = df["val"].dropna().values
        if "test" in df.columns:
            base_test = df["test"].dropna().values
        else:
            base_test = None
    else:
        assert not np.array_equal(base_train, df["train"].dropna().values), "equal splits.."
    if base_test is not None:
        assert np.array_equal(base_test, df["test"].dropna().values), "incorrect test splits.."
            
    train_length = len(df["train"].dropna())
    val_length = len(df["val"].dropna())
    test_length = len(df["test"].dropna()) if "test" in df.columns else 0
    total_length = train_length + val_length + test_length
    print("Split: {} | Train: {:.2f} | Validation: {:.2f} | Test: {:.2f}" .format(i, train_length/total_length, val_length/total_length, test_length/total_length))

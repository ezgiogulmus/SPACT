import sys
import os
import pandas as pd
import numpy as np
import argparse
import warnings
from lifelines.exceptions import ConvergenceWarning
from fs_utils import *
warnings.filterwarnings('ignore', category=ConvergenceWarning)


parser = argparse.ArgumentParser(description='Feature selection')
parser.add_argument('data_name', type=str)
parser.add_argument('target_data', type=str)
parser.add_argument('--dataset_dir', type=str, default="./datasets_csv/fs_csv")
parser.add_argument('--split_dir', type=str, default='./splits')
parser.add_argument('--results_dir', type=str, default='./results/fs')
parser.add_argument('--seed', type=int, default=58, help='random seed (default: 58)')
parser.add_argument('--only_filtering', action="store_true", default=False)
parser.add_argument('--verbose', action="store_true", default=False)
args = parser.parse_args()

if __name__ == "__main__":
    args.results_dir = os.path.join(args.results_dir, args.data_name)
    os.makedirs(args.results_dir, exist_ok=True)
    if os.path.isfile(os.path.join(args.results_dir, f'fs_{args.target_data}.xlsx')):
        print(f"Results already exist for {args.target_data}.")
        sys.exit(0)
    np.random.seed(args.seed)
    saving_dict = {}

    # Prepare the data
    X_train, y_train_time, y_train_event = load_data(args)
    y_train = pd.concat([y_train_time, y_train_event], axis=1)
    X_train = fill_missing(X_train)
    print("\t**Initial shape: ", X_train.shape)

    # Low Variance Filtering
    X_train, removed_cols = var_filter(X_train, thresh=0.01)

    # Normalization
    X_train = norm(X_train)

    # Multi-collinearity Filtering
    X_train, removed_cols2 = multicol_filter(X_train, y_train_time)

    # Univariate Feature Selection
    X_train, logrank_results, removed_cols3 = logrank(X_train, y_train)
    saving_dict["LogRank"] = logrank_results
    logrank_results.to_csv(os.path.join(args.results_dir, f"tmp_logrank_{args.target_data}.csv"))

    # Save the removed feature lists
    print("Removed {}+{}+{}={} features. Remaining: {}" .format(len(removed_cols), len(removed_cols2), len(removed_cols3), len(removed_cols)+len(removed_cols2)+len(removed_cols3), X_train.shape[1]))
    max_length = max(len(removed_cols), len(removed_cols2), len(removed_cols3))

    removed_cols_padded = np.pad(removed_cols, (0, max_length - len(removed_cols)), constant_values=np.nan)
    removed_cols2_padded = np.pad(removed_cols2, (0, max_length - len(removed_cols2)), constant_values=np.nan)
    removed_cols3_padded = np.pad(removed_cols3, (0, max_length - len(removed_cols3)), constant_values=np.nan)

    removed_df = pd.DataFrame({
        "VarThresh": removed_cols_padded,
        "CollinReduced": removed_cols2_padded,
        "Univariate": removed_cols3_padded
    })
    saving_dict["Filtering"] = removed_df
    removed_df.to_csv(os.path.join(args.results_dir, f"tmp_filtering_{args.target_data}.csv"))

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(args.results_dir, f"tmp_data_{args.target_data}.csv"))
    if args.only_filtering or X_train.shape[1] == 0:
        sys.exit(0)
    
    # Calculate SHAP scores
    feature_importance_df = feature_importance(X_train, y_train)
    saving_dict["SHAP"] = feature_importance_df
    feature_importance_df.to_csv(os.path.join(args.results_dir, f"tmp_feats_{args.target_data}.csv"))
    
    # CoxPH evaluation
    results_df = cross_validate_survival_model(X_train, y_train, feature_importance_df, save_path=os.path.join(args.results_dir, f"tmp_cv_{args.target_data}.csv"))
    saving_dict["CV"] = results_df
    
    # Save all results
    with pd.ExcelWriter(os.path.join(args.results_dir, f'fs_{args.target_data}.xlsx')) as writer:
        for k, v in saving_dict.items():
            v.to_excel(writer, sheet_name=k)
    
    # Remove temporary files
    # for f in os.listdir(args.results_dir):
    #     if "tmp_" in f:
    #         os.remove(os.path.join(args.results_dir, f))
    
    

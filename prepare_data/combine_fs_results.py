import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Feature selection')
parser.add_argument('data_name', type=str)
parser.add_argument('--dataset_dir', type=str, default="./datasets_csv")
parser.add_argument('--results_dir', type=str, default='./results/fs')
# parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--verbose', action="store_true", default=False)
args = parser.parse_args()

if __name__ == "__main__":
    csv_path = os.path.join(args.dataset_dir, f"{args.data_name}.csv")
    assert os.path.isfile(csv_path), f"Data file does not exist > {csv_path}"
    df = pd.read_csv(csv_path)[["case_id", "slide_id", "survival_months", "event", "censorship"]]

    overall_results = {}
    selected_features = []
    for target_data in ["pro", "rna", "dna", "cnv", "mut"]:
        if not os.path.isfile(os.path.join(args.dataset_dir, f"{args.data_name}_{target_data}.csv.zip")):
            print(f"File {args.data_name}_{target_data}.csv.zip does not exist")
            continue
        genetic_df = pd.read_csv(os.path.join(args.dataset_dir, f"{args.data_name}_{target_data}.csv.zip"), compression="zip")
        overall_results[target_data] = {"Total": len(genetic_df.columns[1:])}

        excel_file = os.path.join(args.results_dir, args.data_name, f'fs_{target_data}.xlsx')
        if not os.path.isfile(excel_file): 
            print(f"File {excel_file} does not exist")
            continue
        
        removed_df = pd.read_excel(excel_file, sheet_name="Filtering", engine="openpyxl")
        overall_results[target_data].update({col: len(removed_df[col].dropna().values) for col in removed_df.columns[1:]})
        
        feature_list = pd.read_excel(excel_file, sheet_name="SHAP", engine="openpyxl")["feature"].values
        overall_results[target_data].update({"Remaining": len(feature_list)})

        cv_results = pd.read_excel(excel_file, sheet_name="CV", engine="openpyxl")

        max_cv = cv_results[cv_results["Mean C-index"] == cv_results["Mean C-index"].max()]
        selected_nb_of_feats = max_cv["Number of features"].values[0]
        selected_features = feature_list[:selected_nb_of_feats]

        overall_results[target_data].update({
            "Selected features": selected_nb_of_feats,
            "Mean C-index": max_cv["Mean C-index"].values[0],
            "Std C-index": max_cv["Std C-index"].values[0]
        })

        df = pd.merge(df, genetic_df[["case_id"] + list(selected_features)], on="case_id", how="outer")

    print("Final data shape: ", df.shape)
    pd.DataFrame(overall_results).to_csv(os.path.join(args.results_dir, args.data_name, "feature_selection.csv"))
    df.to_csv(os.path.join(args.dataset_dir, f"{args.data_name}_selected.csv"), index=False)
    
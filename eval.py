from __future__ import print_function
import argparse

import os
import sys
import json
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch

### Internal Imports
from utils.data_utils import MIL_Survival_Dataset
from utils.core_utils import eval_model
from utils.file_utils import get_data

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(eval_args): 
	
	feats_dir = eval_args.feats_dir
	coords_dir = eval_args.coords_dir
	data_name = eval_args.data_name
	load_from = eval_args.load_from
	test_all = eval_args.test_all
	
	with open(os.path.join(load_from, "experiment.json"), "r") as f:
		config = json.load(f)

	parser = argparse.ArgumentParser()
	parser.add_argument('--load_from', default=load_from)
	parser.add_argument('--test_all', default=test_all, action="store_false")
	for k, v in config.items():
		parser.add_argument('--' + k, default=v, type=type(v))
	args = parser.parse_args()
	
	args.load_from = load_from
	args.data_name = data_name
	args.feats_dir = feats_dir
	args.coords_dir = coords_dir

	args.split_dir = os.path.join("./splits", args.data_name)
	print("split_dir", args.split_dir)
	assert os.path.isdir(args.split_dir), "Incorrect the split directory: " + args.split_dir

	if args.test_all:
		args.csv_path = os.path.join(args.dataset_dir, args.data_name+".csv")
	print("csv_path", args.csv_path)
	assert os.path.isfile(args.csv_path), "Incorrect csv file path: " + args.csv_path
	
	print("Experiment Name:", args.run_name)
	
	seed_torch(args.seed)
	
	if f'{args.data_name}_eval_summary.csv' in os.listdir(args.results_dir):
		print("Exp Code <%s> already exists! Exiting script." % args.run_name)
		sys.exit()

	settings = vars(args)
	print('\nLoad train survival times: ', config["csv_path"])
	survival_time_list = pd.read_csv(config["csv_path"])["survival_months"].values
	
	print('\nLoad Dataset: ', args.csv_path)
	args, df, indep_vars = get_data(args)
	surv_dataset = MIL_Survival_Dataset(
		df=df,
		data_dir= args.feats_dir,
		coords_dir=args.coords_dir,
		separate_branches=args.separate_branches,
		mode= args.mode,
		print_info=True,
		n_bins=args.n_classes,
		indep_vars=indep_vars,
		survival_time_list=survival_time_list
	)

	print("################# Settings ###################")
	for key, val in settings.items():
		print("{}:  {}".format(key, val))  

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end
	folds = np.arange(start, end)
	for cv in folds:
		
		if test_all:
			dataset = surv_dataset.return_splits(return_all=True, stats_path=os.path.join(args.results_dir, f'train_stats_{cv}.csv'))
		else:
			assert args.data_name in args.split_dir, "Testing is only possible for the same dataset."
			datasets, train_stats = dataset.return_splits(os.path.join(args.split_dir, f"splits_{cv}.csv"))
			dataset = datasets[-1]
		
		_, result_latest = eval_model(dataset, args, cv)
	
		if os.path.isfile(os.path.join(args.results_dir, f'{args.data_name}_eval_summary.csv')):
			results_df = pd.read_csv(os.path.join(args.results_dir, f'{args.data_name}_eval_summary.csv'))
			results_df = results_df.append(result_latest, ignore_index=True)
			results_df.to_csv(os.path.join(args.results_dir, f'{args.data_name}_eval_summary.csv'), index=False)
		else:
			pd.DataFrame(result_latest, index=[0], dtype=float).to_csv(os.path.join(args.results_dir, f'{args.data_name}_eval_summary.csv'), index=False)


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	 

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_from',      type=str, default=None)
	parser.add_argument('--data_name',   type=str, default=None)
	parser.add_argument('--feats_dir',   type=str, default=None)
	parser.add_argument('--coords_dir',   type=str, default=None)
	parser.add_argument('--test_all',   action='store_false', default=True)
	eval_args = parser.parse_args()
	
	start = timer()
	results = main(eval_args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))
	

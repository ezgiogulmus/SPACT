from __future__ import print_function
import argparse

import os
import sys
import json
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import wandb

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_experiment(args=None):
	from utils.data_utils import MIL_Survival_Dataset
	from utils.file_utils import save_pkl, check_directories, get_data
	from utils.core_utils import train
	
	if args is None:
		args = setup_argparse()
		args.wandb = True

	seed_torch(args.seed)
	if args.wandb:
		args.k = 5
		args.k_start = 4
		wandb.init()
		config = wandb.config
		for key, value in config.items():
			print(key, value)
			setattr(args, key, value)
		args.run_name = wandb.run.name
		
	args = check_directories(args)
	os.makedirs(args.results_dir, exist_ok=True)
	if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
		print("Exp Code <%s> already exists! Exiting script." % args.run_name)
		sys.exit()
	args.n_classes = args.n_classes if args.surv_model == "discrete" else 1
	settings = vars(args)
	print("Saving to ", args.results_dir)
	with open(args.results_dir + '/experiment.json', 'w') as f:
		json.dump(settings, f, indent=4)
	
	print("################# Settings ###################")
	for key, val in settings.items():
		print("{}:  {}".format(key, val)) 
	
	args, df, indep_vars = get_data(args)
	
	dataset = MIL_Survival_Dataset(
		df=df,
		data_dir=args.feats_dir,
		coords_dir=args.coords_dir,
		cluster_id_path=os.path.join(args.dataset_dir, f'{args.data_name}_cluster_ids'),
		separate_branches=args.separate_branches,
		mode= args.mode,
		print_info=True,
		n_bins=args.n_classes,
		indep_vars=indep_vars,
		target_nb_patches=args.target_nb_patches,
		slide_level=args.slide_aggregation!="early",
		pooling=args.pooling,
		seed=args.seed
	)

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	results = None
	folds = np.arange(start, end)

	### Start 5-Fold CV Evaluation.
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		val_results_pkl_path = os.path.join(args.results_dir, 'latest_val_results_split{}.pkl'.format(i))
		test_results_pkl_path = os.path.join(args.results_dir, 'latest_test_results_split{}.pkl'.format(i))
		if os.path.isfile(test_results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		datasets, train_stats = dataset.return_splits(os.path.join(args.split_dir, f"splits_{i}.csv"))
		train_stats.to_csv(os.path.join(args.results_dir, f'train_stats_{i}.csv'))
		
		log, val_latest, test_latest = train(datasets, i, args)
		
		if results is None:
			results = {k: [] for k in log.keys()}
		
		for k in log.keys():
			results[k].append(log[k])
		
		if args.wandb:
			wandb.log(log)

		save_pkl(val_results_pkl_path, val_latest)
		if test_latest != None:
			save_pkl(test_results_pkl_path, test_latest)
		pd.DataFrame(results).to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))
		print("Mean c-index: ", np.mean(results["test_cindex"]))
		
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))
	
		



def setup_argparse():
	
	parser = argparse.ArgumentParser(description='Configurations for Multi-Modal Survival Analysis.')

	### Data 
	parser.add_argument('--data_name',   type=str, default=None)
	parser.add_argument('--feats_dir',   type=str, default=None)
	parser.add_argument('--coords_dir',   type=str, default=None)
	parser.add_argument('--dataset_dir', type=str, default="./datasets_csv")
	parser.add_argument('--results_dir', type=str, default='./results', help='Results directory (Default: ./results)')
	parser.add_argument('--split_dir', type=str, default="./splits", help='Split directory (Default: ./splits)')

	### Experiment
	parser.add_argument('--run_name',      type=str, default='run')
	parser.add_argument('--run_config_file',      type=str, default=None)
	parser.add_argument('--seed', 			 type=int, default=58, help='Random seed for reproducible experiment (default: 58)')
	parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
	parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
	parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
	parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
	parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')
	parser.add_argument('--wandb',     	 action='store_true', default=False)

	### Model Parameters
	parser.add_argument('--model_type', type=str, choices=["spact", 'mil', 'snn', 'deepset', "deepattnmisl", 'amil', 'mcat', "motcat", "porpoise"], default='spact',  help='name of model')
	parser.add_argument('--drop_out',        default=.25, type=float, help='Enable dropout (p=0.25)')
	parser.add_argument('--n_classes', type=int, default=4)
	parser.add_argument('--surv_model', default="discrete", choices=["cont", "discrete"])

	############ Multi-modal Parameters
	parser.add_argument('--path_input_dim', type=int, default=768)
	parser.add_argument('--omics', default=None)
	parser.add_argument('--separate_branches',     	 action='store_true', default=False)
	parser.add_argument('--selected_features',     	 action='store_true', default=False)
	parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
	parser.add_argument('--fusion',        type=str, choices=["concat", "bilinear", None], default=None)
	parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
	parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
	parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

	# MOTCAT Parameters
	parser.add_argument('--bs_micro', type=int, default=16384, help='The Size of Micro-batch (Default: 16384)')  # new
	parser.add_argument('--ot_impl', type=str, default='pot-uot-l2', help='impl of ot (default: pot-uot-l2)')  # new
	parser.add_argument('--ot_reg', type=float, default=0.1, help='epsilon of OT (default: 0.1)')
	parser.add_argument('--ot_tau', type=float, default=0.5, help='tau of UOT (default: 0.5)')

	# PORPOISE Parameters
	parser.add_argument('--gate_path', action='store_true', default=False)
	parser.add_argument('--gate_omic', action='store_true', default=False)
	parser.add_argument('--scale_dim1', type=int, default=8)
	parser.add_argument('--scale_dim2', type=int, default=8)
	parser.add_argument('--skip', action='store_true', default=False)
	parser.add_argument('--dropinput', type=float, default=0.0)
	parser.add_argument('--use_mlp', action='store_true', default=False)

	# ViT params
	parser.add_argument("--embedding_dim", default=128, type=int)
	parser.add_argument("--mha_heads", default=4, type=int)
	parser.add_argument("--dim_head", default=16, type=int, help="inner_dim = dim_head * heads")

	# MLP params
	parser.add_argument("--activation", default="relu", choices=["relu", "leakyrelu", "gelu"])
	parser.add_argument("--mlp_depth", default=4, type=int)
	parser.add_argument("--mlp_type", default="small", choices=["small", "big"])
	parser.add_argument("--mlp_skip", default=True, action="store_false")

	# SPACT params
	parser.add_argument('--slide_aggregation', choices=["early", "mid", "late"], default="early")
	parser.add_argument("--target_nb_patches", default=None, action='store', type=int, nargs="+")
	parser.add_argument("--pooling", choices=["maxpooltomin", "avgpooltomin", "maxpoolto1", "avgpoolto1", "padortruncate"], default="avgpoolto1")
	parser.add_argument("--ff", choices=[1, 2], default=2, type=int, help="1: ff only before mha, 2: ff both before and after")
	
	### Training Parameters
	parser.add_argument('--accumulation_steps',      type=int, default=8, help="For continuous surv models.")
	parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
	parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
	parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
	parser.add_argument('--grad_norm', action='store_false', default=True, help='Normalize gradients during training.')
	parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
	parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
	parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
	parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
	parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-5)')
	parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
	parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')

	parser.add_argument('--weighted_sample', action='store_false', default=True, help='Enable weighted sampling')
	parser.add_argument('--early_stopping',  type=int, default=15, help='Enable early stopping')
	parser.add_argument('--lr_patience',  type=int, default=10, help='Enable early stopping')


	args = parser.parse_args()
	return args


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
	args = setup_argparse()
	if args.run_config_file:
		new_run_name = args.run_name
		results_dir = args.results_dir
		split_dir = args.split_dir
		wandb_flag = args.wandb
		cv_fold = args.k
		data_name = args.data_name
		feats_dir = args.feats_dir
		coords_dir = args.coords_dir
		
		with open(args.run_config_file, "r") as f:
			config = json.load(f)
		
		parser = argparse.ArgumentParser()
		parser.add_argument("--run_config_file")
		for k, v in config.items():
			if k != "run_config_file":
				parser.add_argument('--' + k, default=v, type=type(v))
		args = parser.parse_args()
		args.run_name = new_run_name
		args.split_dir = split_dir
		args.results_dir = results_dir
		args.wandb = wandb_flag
		args.k = cv_fold
		args.k_start = 0
		if data_name is not None:
			args.data_name = data_name
			args.feats_dir = f"/media/nfs/SURV/{args.data_name.rsplit('_', 1)[0].upper()}/Feats1024/CONCH"
			args.coords_dir = f"/media/nfs/SURV/{args.data_name.rsplit('_', 1)[0].upper()}/SP1024/patches"

		start = timer()
		run_experiment(args)
		end = timer()
		print("Finished!")
		print('Script Time: %f seconds' % (end - start))
	else:
		start = timer()
		if args.model_type in [ 'deepset', 'amil', 'deepattnmisl', 'mcat', "motcat", "porpoise"]:
			print("Initializing MM survival model")
			import mmsurv
			args.weighted_sample = False if args.data_name == "tcga_kich_os" else args.weighted_sample
			mmsurv.run(args)
		else:
			run_experiment(args)
		end = timer()
		print("Finished!")
		print('Script Time: %f seconds' % (end - start))

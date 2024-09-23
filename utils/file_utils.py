import os
import numpy as np
import pandas as pd
from itertools import chain
import pickle
import h5py

def get_data(args):
	df = pd.read_csv(args.csv_path, compression="zip" if ".zip" in args.csv_path else None)
	indep_vars = []
	args.omics_input_dim = 0
	if args.omics not in ["None", "none", None]:
		print("Selected omics variables:")
		if args.selected_features:
			omics_cols = {k: [col for col in df.columns if col[-3:]==k] for k in args.omics.split(",")}
			indep_vars = list(chain(*omics_cols.values()))
			for k, v in omics_cols.items():
				print("\t", k, len(v))
		else:
			remove_cols = {k: [col for col in df.columns if col[-3:]==k] for k in ["cli", "cnv", "rna", "pro", "mut", "dna"]}
			if "cli" in args.omics:
				cli_cols = remove_cols.pop("cli")
				print("\tcli", len(cli_cols))
				indep_vars.extend(cli_cols)
			df = df[[i for i in df.columns if i not in list(chain(*remove_cols.values()))]]
			print(df.shape)
			for g in args.omics.split(","):
				if g != "cli":
					gen_df = pd.read_csv(f"{args.dataset_dir}/{args.data_name}_{g}.csv.zip", compression="zip")
					indep_vars.extend(gen_df.columns[1:])
					print("\t", g, gen_df.shape[1]-1)
					df = pd.merge(df, gen_df, on='case_id', how="outer")
			df = df.reset_index(drop=True).drop(df.index[df["event"].isna()]).reset_index(drop=True)
		args.omics_input_dim = len(indep_vars)
		if args.separate_branches:
			args.omics_input_dim = []
			gen_types = np.unique([i[-3:] for i in indep_vars])
			for g in gen_types:
				args.omics_input_dim.append(len([col for col in indep_vars if col[-3:]==g]))
	print("Total number of cases: {} | slides: {}" .format(len(df["case_id"].unique()), len(df)))
	return args, df, indep_vars

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def check_directories(args):
	r"""
	Updates the argparse.NameSpace with a custom experiment code.

	Args:
		- args (NameSpace)

	Returns:
		- args (NameSpace)
	"""
	
	feat_extractor = None
	if args.feats_dir:
		feat_extractor = args.feats_dir.split('/')[-1] if len(args.feats_dir.split('/')[-1]) > 0 else args.feats_dir.split('/')[-2]
		if feat_extractor == "RESNET50":
			args.path_input_dim = 2048 
		elif feat_extractor in ["PLIP", "CONCH"]:
			args.path_input_dim = 512 
		elif feat_extractor == "UNI":
			args.path_input_dim = 1024
		else:
			args.path_input_dim = 768

	args.split_dir = os.path.join('./splits', args.data_name)
	print("split_dir", args.split_dir)
	assert os.path.isdir(args.split_dir)

	param_code = args.model_type.upper()
	inputs = []
	if feat_extractor:
		param_code += "_" + feat_extractor
		if args.model_type == "spact" and args.target_nb_patches is not None:
			inputs.append("cpath")
		else:
			inputs.append("path")
	if args.omics:
		inputs.append("tab")
	
	args.mode = ("+").join(inputs)
	if args.mode == "tab" and not args.separate_branches:
		args.fusion = None
	
	param_code += '_' + args.mode

	if args.omics not in ["None", "none", None]:
		suffix = ""
		if args.fusion is not None:
			suffix += "_"+args.fusion
		if args.separate_branches:
			suffix += "_mb"
		suffix += "_"+args.omics
		if not args.selected_features:
			suffix += "_all"
		args.run_name += suffix
	
	args.results_dir = os.path.join(args.results_dir, param_code, args.run_name)
	args.csv_path = f"{args.dataset_dir}/"+args.data_name+".csv" if not args.selected_features else f"{args.dataset_dir}/"+args.data_name+"_selected.csv"
	print("Loading the data from ", args.csv_path)
	assert os.path.isfile(args.csv_path), f"Data file does not exist > {args.csv_path}"
	return args

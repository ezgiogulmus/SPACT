from __future__ import print_function, division
import os
import h5py
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Generic_WSI_Survival_Dataset(Dataset):
	def __init__(self,
		df, print_info=False, n_bins=4, 
		indep_vars=[],  mode="path", survival_time_list=None, 
		target_nb_patches=[10, 50, 100], seed=0, pooling="maxpoolto1",
		slide_level=False
		):
		"""
		Args:
			print_info (bool): Flag to print dataset information.
			n_bins (int): Number of bins to split the survival time.
			indep_vars (list): List of independent variables to use.
			mode (str): Mode for data types (path, tab or path+tab).
			survival_time_list (list or None): List of survival times of the training data (only needed for evaluation).
			target_nb_patches (list): List of desired number of patches in each cluster group.
		"""
		self.print_info = print_info
		self.data_dir = None
		self.coords_dir = None
		self.cluster_id_path = None
		self.separate_branches = None
		self.num_intervals = n_bins
		self.mode = mode
		self.target_nb_patches = target_nb_patches
		self.pooling = pooling
		self.slide_level = slide_level
		self.seed = seed
		
		self.indep_vars = indep_vars
		if self.print_info:
			print("Number of selected tabular data: ", len(self.indep_vars))
		
		slide_data = df[["case_id", "slide_id", "survival_months", "event"]+self.indep_vars]
		
		patients_df = slide_data.drop_duplicates(['case_id']).copy()

		survival_time_list = survival_time_list if survival_time_list is not None else patients_df["survival_months"]
		_, time_breaks = pd.qcut(survival_time_list, q=self.num_intervals, retbins=True, labels=False)
		time_breaks[0] = 0
		time_breaks[-1] = max(survival_time_list.max(), patients_df["survival_months"].max()) + 1
		self.time_breaks = time_breaks
		if self.print_info:
			print("Time intervals: ", self.time_breaks)

		self.patient_dict = {
			case: slide_data["slide_id"][slide_data["case_id"] == case].values \
			for case in slide_data["case_id"].unique()
			}
		
		disc_labels, _ = pd.cut(patients_df["survival_months"], bins=self.time_breaks, retbins=True, labels=False, right=False, include_lowest=True)
		patients_df.insert(2, 'label', disc_labels.values.astype(int))
		
		print(patients_df["label"].value_counts())

		slide_data = patients_df
		slide_data.reset_index(drop=True, inplace=True)
		slide_data = slide_data.assign(slide_id=slide_data['case_id'])

		label_dict = {}
		key_count = 0
		for i in range(len(self.time_breaks)-1):
			for c in [0, 1]:
				label_dict.update({(i, c):key_count})
				key_count+=1

		self.label_dict = label_dict
		
		for i in slide_data.index:
			key = slide_data.loc[i, 'label']
			slide_data.at[i, 'disc_label'] = key
			event = slide_data.loc[i, 'event']
			key = (key, int(event))
			slide_data.at[i, 'label'] = label_dict[key]

		self.num_classes=len(self.label_dict)
		
		new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])
		slide_data = slide_data[new_cols]
		
		self.slide_data = slide_data.reset_index(drop=True)
		
		if print_info:
			self.summarize()
			
	def getlabel(self, ids):
		return self.slide_data['label'][ids]
	
	def summarize(self):
		
		print("label column: {}".format("survival_months"))
		print("number of classes: {}".format(self.num_classes))
		for i in range(self.num_classes):
			cases = self.slide_data["case_id"][self.slide_data["label"]==i].values
			nb_cases = len(cases)
			nb_slides = sum([len(self.patient_dict[v]) for v in cases])
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, nb_cases))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, nb_slides))
		
	def __len__(self):
		return len(self.slide_data)

	def get_split_from_df(self, all_splits=None, split_key='train', scaler=None):
		if split_key == 'all':
			return Generic_Split(self.slide_data, self.time_breaks, self.indep_vars, self.mode, self.data_dir, self.coords_dir, self.cluster_id_path, self.separate_branches, patient_dict=self.patient_dict, print_info=self.print_info, num_classes=self.num_classes, target_nb_patches=self.target_nb_patches, pooling=self.pooling, slide_level=self.slide_level, seed=self.seed)
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, self.time_breaks, self.indep_vars, self.mode, self.data_dir, self.coords_dir, self.cluster_id_path, self.separate_branches, patient_dict=self.patient_dict, print_info=self.print_info, num_classes=self.num_classes, target_nb_patches=self.target_nb_patches, pooling=self.pooling, slide_level=self.slide_level, seed=self.seed)
		else:
			split = None
		
		return split

	def return_splits(self, csv_path=None, return_all=False, stats_path=None):
		
		if return_all:
			test_split = self.get_split_from_df(split_key='all')
			if len(self.indep_vars) > 0:
				train_stats = pd.read_csv(stats_path)
				train_stats.set_index("Unnamed: 0", inplace=True)
				assert "mean" in train_stats.columns and "std" in train_stats.columns
				test_split.preprocess(train_stats, use_csv=True)
			return test_split
		all_splits = pd.read_csv(csv_path)
		train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
		val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
		test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')

		train_stats = train_split.get_stats()
	
		sc = train_split.preprocess(train_stats)
		val_split.preprocess(train_stats, sc=sc)
		test_split.preprocess(train_stats, sc=sc)
		return (train_split, val_split, test_split), train_stats

	def __getitem__(self, idx):
		return None

	def apply_preprocessing(self, slide_data, stats):
		if slide_data.isna().any().any():
			print("Filling missing values with train medians:")
			for col_idx, col in enumerate(self.indep_vars):
				if col_idx % 10000 == 0:
					print("\tProcessing:", col_idx, "/", len(self.indep_vars))
				if slide_data[col].isna().any():
					slide_data[col] = slide_data[col].fillna(stats["median"].loc[col])

		print("Z-score normalization with train mean and std")
		print("\tBefore: {:.2f} - {:.2f}" .format(slide_data[self.indep_vars].min().min(), slide_data[self.indep_vars].max().max()))
		for col_idx, col in enumerate(self.indep_vars):
			slide_data[col] = (slide_data[col] - stats["mean"].loc[col]) / stats["std"].loc[col]
			
		print("\tAfter: {:.2f} - {:.2f}" .format(slide_data[self.indep_vars].min().min(), slide_data[self.indep_vars].max().max()))
		assert slide_data.isna().sum().sum() == 0, "There are still NaN values in the data."
		return slide_data


class MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
	def __init__(self, data_dir, coords_dir, cluster_id_path=None, separate_branches=False, **kwargs):
		super(MIL_Survival_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.coords_dir = coords_dir
		self.cluster_id_path = cluster_id_path
		self.separate_branches = separate_branches

	def __getitem__(self, idx):
		
		case_id = self.slide_data['case_id'].iloc[idx]
		
		event_time = self.slide_data["survival_months"].iloc[idx]
		event = self.slide_data['event'].iloc[idx]
		label = torch.Tensor([self.slide_data['disc_label'][idx]])
		slide_ids = self.patient_dict[case_id]
		
		if len(self.indep_vars) == 0:
			tab_tensor = torch.tensor(np.zeros((1, 1)))
		elif self.separate_branches:
			gen_types = np.unique([i[-3:] for i in self.indep_vars])
			tab_data = []
			for g in gen_types:
				tab_data.append(self.slide_data[[col for col in self.indep_vars if col[-3:]==g]].iloc[idx].values)
			tab_tensor = [torch.tensor(i[np.newaxis, :]) for i in tab_data]
		else:
			tabular_data = self.slide_data[self.indep_vars].iloc[idx].values
			tab_tensor = torch.tensor(tabular_data[np.newaxis, :]) if len(tabular_data) > 0 else torch.tensor(np.zeros((1, 1)))
		
		if "path" in self.mode:
			if "cpath" in self.mode:
				slide_patch_list = []
				for slide_id in slide_ids:
					patch_clusters = {}
					wsi_path = os.path.join(self.data_dir, f'{slide_id.rstrip(".svs")}.pt')
					coords_path = os.path.join(self.coords_dir, f'{slide_id.rstrip(".svs")}.h5')
					wsi_bag = torch.load(wsi_path, weights_only=True)
					with h5py.File(coords_path, "r") as hf:
						coords = np.array(hf["coords"])
					
					for k in self.target_nb_patches:
						n_clusters = max(1, len(coords) // k)
						
						# kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed).fit(coords)
						# kmeans_labels = kmeans.labels_
						kmeans_labels = self.kmeans_labels_dict[k][slide_id.rstrip(".svs")]
						clustered_patches = []
						for i in range(n_clusters):
							
							cluster_indices = np.where(kmeans_labels == i)[0]
							if len(cluster_indices) > 0:
								clustered_patches.append(wsi_bag[cluster_indices])
								
						patch_clusters[k] = clustered_patches
					slide_patch_list.append(patch_clusters)

				if not self.slide_level:
					patch_clusters = {k: [] for k in self.target_nb_patches}
					for p_clusters in slide_patch_list:
						for k in patch_clusters.keys():
							patch_clusters[k].extend(p_clusters[k])
					slide_patch_list = [patch_clusters]
				
				slide_path_features = []
				for slide_patch_clusters in slide_patch_list:
					path_features = [[] for _ in self.target_nb_patches]
					for i, k in enumerate(self.target_nb_patches):
						for cluster in slide_patch_clusters[k]:
							if self.pooling == "padortruncate":
								t, m = self.pad_or_truncate(cluster, k)
								path_features[i].append(t)
							else:
								min_patch_size = min([x.shape[0] for x in slide_patch_clusters[k]])
								embed_dim = slide_patch_clusters[k][0].shape[1]
								
								if self.pooling == "maxpooltomin":
									pooled_cluster, _ = F.adaptive_max_pool2d_with_indices(cluster.unsqueeze(0), (min_patch_size, embed_dim), return_indices=True)
								elif self.pooling == "avgpooltomin":
									pooled_cluster = F.adaptive_avg_pool2d(cluster.unsqueeze(0), (min_patch_size, embed_dim))
								elif self.pooling == "maxpoolto1":
									pooled_cluster, _ = F.adaptive_max_pool2d_with_indices(cluster.unsqueeze(0), (1, embed_dim))
								elif self.pooling == "avgpoolto1":
									pooled_cluster = F.adaptive_avg_pool2d(cluster.unsqueeze(0), (1, embed_dim))
								path_features[i].append(pooled_cluster.squeeze(0))
					slide_path_features.append([torch.stack(tensor_list, dim=0) for tensor_list in path_features])
					
			else:
				slide_path_features = []
				for slide_id in slide_ids:
					wsi_path = os.path.join(self.data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
					wsi_bag = torch.load(wsi_path, weights_only=True)
					slide_path_features.append(wsi_bag)
				slide_path_features = torch.cat(slide_path_features, dim=0)
		else:
			slide_path_features = torch.zeros((1,1))
		
		return (slide_path_features, label, event_time, event, tab_tensor, case_id)

	def pad_or_truncate(self, tensor, size):
		mask = torch.ones(tensor.shape[0])
		if tensor.shape[0] < size:
			padding = torch.zeros(size - tensor.shape[0], tensor.shape[1])
			padded_tensor = torch.cat([tensor, padding], dim=0)
			mask = torch.cat([mask, padding[:, 0]], dim=0)
			return padded_tensor, mask
		return tensor[:size], mask


class Generic_Split(MIL_Survival_Dataset):
	def __init__(
			self, slide_data, time_breaks, indep_vars,
			mode, data_dir=None, coords_dir=None, cluster_id_path=None,
			separate_branches=False, patient_dict=None, 
			print_info=False, num_classes=4, target_nb_patches=None,
			seed=0, pooling="maxpoolto1", slide_level=True
		):
		"""
		Args:
			slide_data (DataFrame): Data for the current split.
			time_breaks (list): Time intervals for survival analysis.
			data_dir (string): Directory where the slide features are located.
			coords_dir (string): Directory where the coordinates are located.
			separate_branches (bool): Whether to use separate branches for different genomic types.
			patient_dict (dict): Dictionary mapping patient IDs to slide data.
		"""
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.coords_dir = coords_dir
		self.cluster_id_path = cluster_id_path
		self.separate_branches = separate_branches
		self.patient_dict = patient_dict
		self.time_breaks = time_breaks
		self.print_info = print_info
		self.target_nb_patches = target_nb_patches
		self.pooling = pooling
		self.slide_level = slide_level
		self.seed = seed
		self.slide_cls_ids = [[] for i in range(num_classes)]
		for i in range(num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
		
		self.mode = mode
		self.indep_vars = indep_vars

		if "cpath" in mode:
			self.kmeans_labels_dict = {}
			for k in self.target_nb_patches:
				with open(os.path.join(self.cluster_id_path, f"k{k}_labels.pkl"), "rb") as pf:
					self.kmeans_labels_dict[k] = pickle.load(pf)
		
	def __len__(self):
		return len(self.slide_data)

	def get_stats(self):
		median_vals = self.slide_data[self.indep_vars].median()
		mean_vals = self.slide_data[self.indep_vars].mean()
		std_vals = self.slide_data[self.indep_vars].std()
		std_vals[std_vals == 0] = 1
		assert 0 not in std_vals.values, "There are still 0 values in the standard deviation."
		stats = pd.concat([median_vals, mean_vals, std_vals], axis=1)
		stats.columns = ['median', 'mean', 'std']
		return stats

	def preprocess(self, stats, sc=None, use_csv=False):
		if len(self.indep_vars) > 0:
			print("Filling missing values with train medians:")
			for col_idx, col in enumerate(self.indep_vars):
				if col_idx % 10000 == 0:
					print("\tProcessing:", col_idx, "/", len(self.indep_vars))
				if self.slide_data[col].isna().any():
					self.slide_data[col] = self.slide_data[col].fillna(stats["median"].loc[col])
			print("Z-score normalization with train mean and std")
			if sc == None and not use_csv:
				sc = StandardScaler()
				self.slide_data[self.indep_vars] = sc.fit_transform(self.slide_data[self.indep_vars])
				print(self.slide_data[self.indep_vars].max().max(), self.slide_data[self.indep_vars].min().min())
				return sc
			elif sc == None and use_csv:
				for col_idx, col in enumerate(self.indep_vars):
					mean_val = float(stats["mean"].loc[col])
					std_val = float(stats["std"].loc[col])
					self.slide_data[col] = (self.slide_data[col] - mean_val) / std_val
			else:
				self.slide_data[self.indep_vars] = sc.transform(self.slide_data[self.indep_vars])
			print(self.slide_data[self.indep_vars].max().max(), self.slide_data[self.indep_vars].min().min())
		assert self.slide_data.isna().sum().sum() == 0, "There are still NaN values in the data."
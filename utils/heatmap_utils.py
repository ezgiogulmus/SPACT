import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from utils.wsi_utils import WholeSlideImage


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
initiate a pandas df describing a list of slides to process
args:
	slides (df or array-like): 
		array-like structure containing list of slide ids, if df, these ids assumed to be
		stored under the 'slide_id' column
	seg_params (dict): segmentation paramters 
	filter_params (dict): filter parameters
	vis_params (dict): visualization paramters
	patch_params (dict): patching paramters
	use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
'''
def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
	use_heatmap_args=False, save_patches=False):

	total = len(slides)
	if isinstance(slides, pd.DataFrame):
		slide_ids = slides.slide_id.values
	else:
		slide_ids = slides
	default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)}

	# initiate empty labels in case not provided
	if use_heatmap_args:
		default_df_dict.update({'label': np.full((total), -1)})
	
	default_df_dict.update({
		'status': np.full((total), 'tbp'),
		# seg params
		'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
		'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
		'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
		'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
		'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
		'keep_ids': np.full((total), seg_params['keep_ids']),
		'exclude_ids': np.full((total), seg_params['exclude_ids']),
		
		# filter params
		'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
		'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
		'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

		# vis params
		'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
		'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

		# patching params
		'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
		'contour_fn': np.full((total), patch_params['contour_fn'])
		})

	if save_patches:
		default_df_dict.update({
			'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
			'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

	if use_heatmap_args:
		# initiate empty x,y coordinates in case not provided
		default_df_dict.update({'x1': np.empty((total)).fill(np.NaN), 
			'x2': np.empty((total)).fill(np.NaN), 
			'y1': np.empty((total)).fill(np.NaN), 
			'y2': np.empty((total)).fill(np.NaN)})


	if isinstance(slides, pd.DataFrame):
		temp_copy = pd.DataFrame(default_df_dict) # temporary dataframe w/ default params
		# find key in provided df
		# if exist, fill empty fields w/ default values, else, insert the default values as a new column
		for key in default_df_dict.keys(): 
			if key in slides.columns:
				mask = slides[key].isna()
				slides.loc[mask, key] = temp_copy.loc[mask, key]
			else:
				slides.insert(len(slides.columns), key, default_df_dict[key])
	else:
		slides = pd.DataFrame(default_df_dict)
	
	return slides

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
	return params

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
	wsi_object = WholeSlideImage(wsi_path)
	target_downsample = None
	if seg_params['seg_level'] < 0:
		seg_params['seg_level'] = wsi_object.wsi.get_best_level_for_downsample(32)

	wsi_object.segmentTissue(**seg_params, filter_params=filter_params, target_downsample=target_downsample)
	wsi_object.saveSegmentation(seg_mask_path)
	return wsi_object


# def get_patch_clusters(features, coords, target_nb_patches):
#     def pad_or_truncate(tensor1, tensor2, size):
#         assert tensor2.shape[0] == tensor1.shape[0], "Tensors must have the same number of elements"
#         if tensor1.shape[0] < size:
#             padded_tensor1 = torch.cat([tensor1, torch.zeros(size - tensor1.shape[0], tensor1.shape[1])], dim=0)
#             padded_tensor2 = np.concatenate([tensor2, torch.zeros(size - tensor2.shape[0], tensor2.shape[1])], axis=0)
#             return padded_tensor1, padded_tensor2
#         return tensor1[:size], tensor2[:size]
#     patch_clusters = {k: [] for k in target_nb_patches}
#     coords_clusters = {k: [] for k in target_nb_patches}
#     for k in target_nb_patches:
#         n_clusters = max(1, len(coords) // k)
#         kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
#         kmeans_labels = kmeans.labels_
            
#         clustered_patches, clustered_coords = [], []
#         for i in range(n_clusters):
#             cl_ids = np.where(kmeans_labels == i)[0]
#             if len(cl_ids) > 0:
#                 clustered_patches.append(features[cl_ids])
#                 clustered_coords.append(coords[cl_ids])
                
#         patch_clusters[k].extend(clustered_patches)
#         coords_clusters[k].extend(clustered_coords)
    
#     path_features = [[] for _ in target_nb_patches]
#     coordinates = [[] for _ in target_nb_patches]
#     for i, k in enumerate(target_nb_patches):
#         for m in range(len(patch_clusters[k])):
#             t, c = pad_or_truncate(patch_clusters[k][m], coords_clusters[k][m], k)
#             path_features[i].append(t)
#             coordinates[i].append(c)
#     path_features = [torch.stack(tensor_list, dim=0) for tensor_list in path_features]
#     coordinates = [np.stack(tensor_list, axis=0) for tensor_list in coordinates]
#     return path_features, coordinates

# def combine_attention_weights(cluster_att, mha_att, masks, coords):
# 	attention_scores = []
# 	for i in range(len(cluster_att)):
# 		c_mask = masks[i].reshape(-1, 2)

# 		c_att = cluster_att[i][None, None, None, :, :, 0].detach().cpu()
# 		m_att = mha_att[i][:, :, :, :, None].detach().cpu()
# 		# print(c_att.shape, m_att.shape)
# 		combined_att = c_att * m_att
# 		# print(combined_att.shape)
# 		combined_att = combined_att.sum(dim=1)
# 		# print(combined_att.shape)
# 		omics_att = combined_att.chunk(5, dim=1)
# 		ordered_omics_att = []
# 		for att_tensor in omics_att:
# 			# print(att_tensor.shape, att_tensor.max(), att_tensor.min())
# 			att_tensor /= att_tensor.max()
# 			# print(att_tensor.shape, att_tensor.max(), att_tensor.min())
			
# 			coords_dict = {tuple(c): idx for idx, c in enumerate(coords)}
# 			wsi_attention_map = np.zeros(len(coords))
# 			for c, att in zip(c_mask, att_tensor.reshape(-1)):
# 				coord_tuple = tuple(c)
# 				if coord_tuple in coords_dict:
# 					idx = coords_dict[coord_tuple]
# 					if wsi_attention_map[idx] != 0:
# 						print("Double coords: ", coord_tuple, wsi_attention_map[idx], att.item())
# 					wsi_attention_map[idx] = att.item()
# 			ordered_omics_att.append(wsi_attention_map)
# 		attention_scores.append(np.stack(ordered_omics_att, axis=0))
# 	return np.stack(attention_scores, axis=0)

# def get_attention_scores(model, img_features, tab_features, masks, coords):
# 	if isinstance(img_features, list):
# 		img_features = [i.to(device) for i in img_features]
# 	else:
# 		img_features = img_features.to(device)
	
# 	if isinstance(tab_features, list):
# 		tab_features = [i.to(device) for i in tab_features]
# 	else:
# 		tab_features = tab_features.to(device) if tab_features is not None else None
	
# 	with torch.no_grad():
# 		hazards, _, A = model(img_features, tab_features, return_weights=True)
# 		A_scores = combine_attention_weights(A["cluster_gates"], A["mha"], masks, coords)
		
# 		S = torch.cumprod(1 - hazards, dim=1)[0].cpu().numpy()
# 		print(S)
# 		ids = np.argwhere(S < .5)
# 		y_pred  = ids[0][0] if len(ids) > 0 else len(S)
# 		print(ids, y_pred)
# 	return A_scores, y_pred, hazards[0].cpu().numpy(), S

# ##### FOR AMBER ####
# import torch.nn.functional as F
# def get_patch_clusters(features, coords, target_nb_patches, pooling):
#     def pad_or_truncate(tensor1, tensor2, size):
#         assert tensor2.shape[0] == tensor1.shape[0], "Tensors must have the same number of elements"
#         if tensor1.shape[0] < size:
#             padded_tensor1 = torch.cat([tensor1, torch.zeros(size - tensor1.shape[0], tensor1.shape[1])], dim=0)
#             padded_tensor2 = np.concatenate([tensor2, torch.zeros(size - tensor2.shape[0], tensor2.shape[1])], axis=0)
#             return padded_tensor1, padded_tensor2
#         return tensor1[:size], tensor2[:size]
#     patch_clusters = {k: [] for k in target_nb_patches}
#     coords_clusters = {k: [] for k in target_nb_patches}
#     for k in target_nb_patches:
#         n_clusters = max(1, len(coords) // k)
#         kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
#         kmeans_labels = kmeans.labels_
            
#         clustered_patches, clustered_coords = [], []
#         for i in range(n_clusters):
#             cl_ids = np.where(kmeans_labels == i)[0]
#             if len(cl_ids) > 0:
#                 clustered_patches.append(features[cl_ids])
#                 clustered_coords.append(coords[cl_ids])
                
#         patch_clusters[k].extend(clustered_patches)
#         coords_clusters[k].extend(clustered_coords)
    
#     path_features = [[] for _ in target_nb_patches]
#     coordinates = [[] for _ in target_nb_patches]
#     for i, k in enumerate(target_nb_patches):
#         for m in range(len(patch_clusters[k])):
#             if pooling == "padortruncate":
#                 t, c = pad_or_truncate(patch_clusters[k][m], coords_clusters[k][m], k)
#                 path_features[i].append(t)
#                 coordinates[i].append(c)
#             else:
#                 min_patch_size = min([x.shape[0] for x in patch_clusters[k]])
#                 embed_dim = patch_clusters[k][0].shape[1]
                
#                 if pooling == "maxpooltomin":
#                     t, c = F.adaptive_max_pool2d_with_indices(patch_clusters[k][m].unsqueeze(0), (min_patch_size, embed_dim), return_indices=True)
#                 elif pooling == "avgpooltomin":
#                     t = F.adaptive_avg_pool2d(patch_clusters[k][m].unsqueeze(0), (min_patch_size, embed_dim))
#                     c = coords_clusters[k][m] 
#                 elif pooling == "maxpoolto1":
#                     t, c = F.adaptive_max_pool2d_with_indices(patch_clusters[k][m].unsqueeze(0), (1, embed_dim))
#                 elif pooling == "avgpoolto1":
#                     t = F.adaptive_avg_pool2d(patch_clusters[k][m].unsqueeze(0), (1, embed_dim))
#                     c = coords_clusters[k][m]
#                 path_features[i].append(t.squeeze(0))
#                 # coordinates[i].append(c)
#     path_features = [torch.stack(tensor_list, dim=0) for tensor_list in path_features]
#     # coordinates = [np.stack(tensor_list, axis=0) for tensor_list in coordinates]

#     return path_features, list(coords_clusters.values()) 

# def combine_attention_weights_sb(mha_att, cluster_att=None):
# 	attention_scores = []
# 	for i in range(len(mha_att)):
# 		if cluster_att is not None:
# 			c_att = cluster_att[i][None, None, None, :, :, 0].detach().cpu()
# 			m_att = mha_att[i][:, :, :, :, None].detach().cpu()
# 			combined_att = c_att * m_att
# 			combined_att = combined_att.sum(dim=1)
# 		else:
# 			combined_att = mha_att[i].detach().cpu().sum(dim=1)
# 		attention_scores.append(combined_att)
# 	return attention_scores

# def extend_attention_scores_to_wsi(A, masks, coords):
# 	A_scores = []
# 	for cluster_masks, cluster_A in list(zip(masks, A)):
# 		extended_attention_scores = np.zeros((coords.shape[0]))
# 		for i, mask in enumerate(cluster_masks):
# 			for c in mask:
# 				idx = np.where((coords == c).all(axis=1))[0][0]
# 				extended_attention_scores[idx] = cluster_A[0, 0, i]
# 		A_scores.append(extended_attention_scores)
# 	return np.stack(A_scores, axis=0)

# def get_attention_scores(model, img_features, tab_features):
# 	if isinstance(img_features, list):
# 		img_features = [[i.to(device) for i in j] for j in img_features]
# 	else:
# 		img_features = img_features.to(device)
	
# 	if isinstance(tab_features, list):
# 		tab_features = [i.to(device) for i in tab_features]
# 	else:
# 		tab_features = tab_features.to(device) if tab_features is not None else None
	
# 	with torch.no_grad():
# 		hazards, A = model(img_features, tab_features, return_weights=True)
# 		A = A[0]
# 		A_scores = combine_attention_weights_sb(mha_att=A["mha"], cluster_att = A["cluster_gates"] if len(A["cluster_gates"]) > 0 else None)
# 		# normalize A_scores
# 		A_scores = [a/a.max() for a in A_scores]
# 		S = torch.cumprod(1 - hazards, dim=1)[0].cpu().numpy()
# 		# print(S)
# 		ids = np.argwhere(S < .5)
# 		y_pred  = ids[0][0] if len(ids) > 0 else len(S)
# 		# print(ids, y_pred)
# 	return A_scores, y_pred, hazards[0].cpu().numpy(), S

from sklearn.cluster import KMeans
import torch.nn.functional as F
def get_patch_clusters(features, coords, slide_name, kmeans_labels_dict, run_args):
	def pad_or_truncate(tensor, size):
		mask = torch.ones(tensor.shape[0])
		if tensor.shape[0] < size:
			padding = torch.zeros(size - tensor.shape[0], tensor.shape[1])
			padded_tensor = torch.cat([tensor, padding], dim=0)
			mask = torch.cat([mask, padding[:, 0]], dim=0)
			return padded_tensor, mask
		return tensor[:size], mask
	patch_clusters = {}
	coords_clusters = {}
	for k in run_args.target_nb_patches:
		n_clusters = max(1, len(coords) // k)
		kmeans_labels = kmeans_labels_dict[k][slide_name.rstrip(".svs")]
		clustered_patches, clustered_coords = [], []
		for i in range(n_clusters):
			cl_ids = np.where(kmeans_labels == i)[0]
			if len(cl_ids) > 0:
				clustered_patches.append(features[cl_ids])
				clustered_coords.append(coords[cl_ids])
				
		patch_clusters[k] = clustered_patches
		coords_clusters[k] = clustered_coords

	path_features = [[] for _ in run_args.target_nb_patches]
	for i, k in enumerate(run_args.target_nb_patches):
		for cluster in patch_clusters[k]:
			if run_args.pooling == "padortruncate":
				t, _ = pad_or_truncate(cluster, k)
				path_features[i].append(t)
			else:
				min_patch_size = min([x.shape[0] for x in patch_clusters[k]])
				embed_dim = patch_clusters[k][0].shape[1]
				
				if run_args.pooling == "maxpooltomin":
					pooled_cluster, _ = F.adaptive_max_pool2d_with_indices(cluster.unsqueeze(0), (min_patch_size, embed_dim), return_indices=True)
				elif run_args.pooling == "avgpooltomin":
					pooled_cluster = F.adaptive_avg_pool2d(cluster.unsqueeze(0), (min_patch_size, embed_dim))
				elif run_args.pooling == "maxpoolto1":
					pooled_cluster, _ = F.adaptive_max_pool2d_with_indices(cluster.unsqueeze(0), (1, embed_dim))
				elif run_args.pooling == "avgpoolto1":
					pooled_cluster = F.adaptive_avg_pool2d(cluster.unsqueeze(0), (1, embed_dim))
				path_features[i].append(pooled_cluster.squeeze(0))
	path_features = [torch.stack(tensor_list, dim=0) for tensor_list in path_features]
	return path_features, list(coords_clusters.values()) 

def combine_attention_weights_w_upscale(cluster_att, mha_att, masks, coords):
	def upscale_array(pooled_array, original_size):
		upscaled_array = np.repeat(pooled_array, np.ceil(original_size / pooled_array.shape[0]), axis=0)
		extra_rows = upscaled_array.shape[0] - original_size
		if extra_rows > 0:
			upscaled_array = upscaled_array[extra_rows//2:original_size+extra_rows//2]
		return upscaled_array
	attention_scores = []
	for i in range(len(mha_att)):
		m_att = mha_att[i][:, :, :, :, None].detach().cpu()
		if len(cluster_att) > 0:
			c_att = cluster_att[i][None, None, None, :, :, 0].detach().cpu()
			combined_att = c_att * m_att
		else:
			combined_att = m_att
		combined_att = combined_att.sum(dim=1)
		# print(combined_att.shape)
		if combined_att.shape[1] == 1:
			omics_att = [combined_att]
		else:
			omics_att = combined_att.chunk(combined_att.shape[1], dim=1)

		ordered_omics_att = []
		c_mask = masks[i]
		for att_tensor in omics_att:
			# print(att_tensor.shape, att_tensor.max(), att_tensor.min())
			att_tensor /= att_tensor.max()
			# print(att_tensor.shape, att_tensor.max(), att_tensor.min())
			
			coords_dict = {tuple(c): idx for idx, c in enumerate(coords)}
			wsi_attention_map = np.zeros(len(coords))
			for c, att in zip(c_mask, att_tensor.squeeze(0).squeeze(0)):
				upscaled_att =  upscale_array(att, len(c))
				for coord_tuple, u_att in list(zip(c, upscaled_att)):
					if tuple(coord_tuple) in coords_dict:
						idx = coords_dict[tuple(coord_tuple)]
						if wsi_attention_map[idx] != 0:
							print("Double coords: ", coord_tuple, wsi_attention_map[idx], att.item())
						wsi_attention_map[idx] = u_att.item()
			ordered_omics_att.append(wsi_attention_map)
		attention_scores.append(np.stack(ordered_omics_att, axis=0))
	return np.stack(attention_scores, axis=0)

def get_attention_scores(model, img_features, tab_features, masks, coords):
	if isinstance(img_features, list):
		img_features = [[i.to(device) for i in j] for j in img_features]
	else:
		img_features = img_features.to(device)
	
	if isinstance(tab_features, list):
		tab_features = [i.to(device) for i in tab_features]
	else:
		tab_features = tab_features.to(device) if tab_features is not None else None
	
	with torch.no_grad():
		hazards, A = model(img_features, tab_features, return_weights=True)
		A = A[0]
		A_scores = combine_attention_weights_w_upscale(mha_att=A["mha"], cluster_att = A["cluster_gates"], masks=masks, coords=coords)
		# normalize A_scores
		# A_scores = [a/a.max() for a in A_scores]
		S = torch.cumprod(1 - hazards, dim=1)[0].cpu().numpy()
		# print(S)
		ids = np.argwhere(S < .5)
		y_pred  = ids[0][0] if len(ids) > 0 else len(S)
		# print(ids, y_pred)
	return A_scores, y_pred, hazards[0].cpu().numpy(), S

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
	if wsi_object is None:
		wsi_object = WholeSlideImage(slide_path)
		print(wsi_object.name)
	
	wsi = wsi_object.wsi
	if vis_level < 0:
		vis_level = wsi.get_best_level_for_downsample(32)
	
	heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
	return heatmap
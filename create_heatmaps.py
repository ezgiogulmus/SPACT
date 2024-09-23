import argparse
import os
import h5py
import yaml
import json
import numpy as np
import pandas as pd
import pickle
import torch
from utils.heatmap_utils import initialize_df, load_params, initialize_wsi, get_patch_clusters, get_attention_scores, drawHeatmap
from utils.file_utils import get_data
from utils.data_utils import Generic_WSI_Survival_Dataset
from utils.train_utils import model_builder

parser = argparse.ArgumentParser()

exp_args = parser.add_argument_group("experiment")
exp_args.add_argument("--save_exp_code", default="HEATMAP_OUTPUT")
exp_args.add_argument("--save_dir", default="heatmaps/")

data_args = parser.add_argument_group("data")
data_args.add_argument("--data_dir", default="/media/nfs/SURV/TCGA_OV/Slides")
data_args.add_argument("--process_list", default="tcga_ov_os_slides.csv")
data_args.add_argument("--slide_ext", default=".svs")

patch_args = parser.add_argument_group("patching")
patch_args.add_argument("--patch_size", default=1024, type=int)
patch_args.add_argument("--overlap", default=0.5, type=float)
patch_args.add_argument("--patch_level", default=0, type=int)

model_args = parser.add_argument_group("model")
model_args.add_argument("--load_dir", default=None)
model_args.add_argument("--ckpt_name", default="s_0_checkpoint.pt")
model_args.add_argument("--cluster_id_path", default="./datasets_csv/tcga_ov_os_cluster_ids")

heatmap_args = parser.add_argument_group("heatmaps")
heatmap_args.add_argument("--vis_level", default=1, type=int)
heatmap_args.add_argument("--alpha", default=0.4, type=float)
heatmap_args.add_argument("--blank_canvas", default=False, action="store_true")
heatmap_args.add_argument("--save_orig", default=True, action="store_false")
heatmap_args.add_argument("--save_ext", default="jpg")
heatmap_args.add_argument("--use_ref_scores", default=True, action="store_false")
heatmap_args.add_argument("--blur", default=False, action="store_true")
heatmap_args.add_argument("--use_center_shift", default=True, action="store_false")
heatmap_args.add_argument("--use_roi", default=False, action="store_true")
heatmap_args.add_argument("--calc_heatmap", default=True, action="store_false")
heatmap_args.add_argument("--binarize", default=False, action="store_true")
heatmap_args.add_argument("--binary_thresh", default=-1, type=float)
heatmap_args.add_argument("--cmap", default="jet")

heatmap_args.add_argument("--avg_clusters", default=False, action="store_true")
heatmap_args.add_argument("--avg_genomics", default=False, action="store_true")

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    with open(os.path.join(args.load_dir, "experiment.json"), "r") as jf:
        run_config = json.load(jf)
    run_parser = argparse.ArgumentParser()
    for k, v in run_config.items():
        run_parser.add_argument("--" + k, default=v, type=type(v))

    run_args = run_parser.parse_args("")
    run_args.results_dir = args.load_dir
    args.feat_dir = run_args.feats_dir
    args.h5_dir = run_args.coords_dir

    args.avg_genomics = args.avg_genomics if run_args.separate_branches else True
    
    patch_size = tuple([args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1-args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], args.overlap, step_size[0], step_size[1]))

    def_seg_params = {
        'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 
        'close': 2, 'use_otsu': False, 
        'keep_ids': 'none', 'exclude_ids':'none'
    }
    def_filter_params = {
        'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10
    }
    def_vis_params = {
        'vis_level': -1, 'line_thickness': 250
    }
    def_patch_params = {
        'use_padding': True, 'contour_fn': 'four_pt'
    }

    kmeans_labels_dict = {}
    for k in run_args.target_nb_patches:
        with open(os.path.join(args.cluster_id_path, f"k{k}_labels.pkl"), "rb") as pf:
            kmeans_labels_dict[k] = pickle.load(pf)

    if args.process_list is None:
        if isinstance(args.data_dir, list):
            slides = []
            for data_dir in args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(args.data_dir))
        slides = [slide for slide in slides if args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nNb of slides to process: ', len(process_stack))

    args.csv_path = run_args.csv_path
    args.omics = run_args.omics
    args.selected_features = run_args.selected_features
    args.separate_branches = run_args.separate_branches
    args, tab_df, indep_vars = get_data(args)

    process_stack = process_stack[process_stack["slide_id"].isin(tab_df["slide_id"].values)].reset_index(drop=True)
    survival_time_list = tab_df["survival_months"].values
    
    tab_df = tab_df[tab_df["slide_id"].isin(process_stack["slide_id"].values)].reset_index(drop=True)
    
    dataset = Generic_WSI_Survival_Dataset(
        df=tab_df,
		mode= run_args.mode,
		print_info=True,
		n_bins=run_args.n_classes,
		indep_vars=indep_vars,
		target_nb_patches=run_args.target_nb_patches,
        survival_time_list=survival_time_list
	)

    
    print('\ninitializing model from ckpt: {}'.format(os.path.join(args.load_dir, args.ckpt_name)))
    run_args.omics_input_dim = args.omics_input_dim
    model, _, _, _, _, _ =  model_builder(run_args, 1, os.path.join(args.load_dir, args.ckpt_name))
    model.eval()

    label_dict = {}
    for k, v in dataset.label_dict.items():
        label_dict[v] = str(round(dataset.time_breaks[k[0]+1]))+"m"+str(k[1])
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    train_stats = pd.read_csv(os.path.join(run_args.results_dir, f'train_stats_{args.ckpt_name[2]}.csv'))
    train_stats = train_stats.set_index("Unnamed: 0")
    slide_data = dataset.apply_preprocessing(dataset.slide_data, train_stats)
    for i in range(len(process_stack)):
        
        slide_name = process_stack.loc[i, 'slide_id']
        if args.slide_ext not in slide_name:
            slide_name+=args.slide_ext
        print('\nprocessing: ', slide_name)	

        label = slide_data.loc[slide_data['case_id'] == slide_name[:12], 'label'].item()
        time = slide_data.loc[slide_data['case_id'] == slide_name[:12], 'survival_months'].item()
        event = slide_data.loc[slide_data['case_id'] == slide_name[:12], 'event'].item()
        grouping = label_dict[label]
        print('label: ', label, "group: ", grouping, "time: ", time, "event: ", event)
        slide_id = slide_name.replace(args.slide_ext, '')

        slide_save_dir = os.path.join(args.save_dir, args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(slide_save_dir, exist_ok=True)

        if args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        slide_path = os.path.join(args.data_dir, slide_name)
        mask_file = os.path.join(slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[args.patch_level]

        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample)).astype(int))

        block_map_save_path = os.path.join(slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            vis_params['vis_level'] = wsi_object.wsi.get_best_level_for_downsample(32)
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(args.feat_dir, slide_id+'.pt')
        h5_path = os.path.join(args.h5_dir, slide_id+'.h5')

        features = torch.load(features_path)
        print("Nb of patches: ", len(features))
        process_stack.loc[i, 'bag_size'] = len(features)
        with h5py.File(h5_path, "r") as hf:
            coords = hf['coords'][:]
        features, masks = get_patch_clusters(features, coords, slide_name, kmeans_labels_dict, run_args)
        
        wsi_object.saveSegmentation(mask_file)

        if len(indep_vars) == 0:
            tab_tensor = torch.tensor(np.zeros((1, 1)))
        elif run_args.separate_branches:
            gen_types = np.unique([i[-3:] for i in indep_vars])
            tab_data = []
            selected_slide_data = slide_data.loc[slide_data['case_id'] == slide_name[:12], :]
            for g in gen_types:
                tab_data.append(selected_slide_data[[col for col in indep_vars if col[-3:]==g]].values)
            tab_tensor = [torch.tensor(i[np.newaxis, :], dtype=torch.float32) for i in tab_data]
        else:
            tabular_data = slide_data.loc[slide_data['case_id'] == slide_name[:12], indep_vars].values
            tab_tensor = torch.tensor(tabular_data[np.newaxis, :], dtype=torch.float32) 
        
        A, y_pred, hazards, surv_probs = get_attention_scores(model, [features], tab_tensor, masks, coords)
        # A = extend_attention_scores_to_wsi(A, masks, coords)
        
        del features
        if not os.path.isfile(block_map_save_path): 
            with h5py.File(block_map_save_path, 'w') as hf:
                hf.create_dataset('attention_scores', data=A)
                hf.create_dataset('coords', data=coords)
            
        for c in range(run_args.n_classes):
            process_stack.loc[i, 'Hazards_{}'.format(c)] = hazards[c]
        for c in range(run_args.n_classes):
            process_stack.loc[i, 'S_{}'.format(c)] = surv_probs[c]
        process_stack.loc[i, 'y_pred'] = y_pred
        process_stack.loc[i, 'y_true'] = grouping

        os.makedirs('{}/{}/results/' .format(args.save_dir, args.save_exp_code,), exist_ok=True)
        if args.process_list is not None:
            process_stack.to_csv('{}/{}/results/{}.csv'.format(args.save_dir, args.save_exp_code, args.process_list.replace('.csv', '')), index=False)
        else:
            process_stack.to_csv('{}/results/{}.csv'.format(args.save_dir, exp_args.save_exp_code), index=False)

        with h5py.File(block_map_save_path, 'r') as hf:
            scores = np.array(hf['attention_scores'])
            coords = np.array(hf['coords'])

        if args.avg_clusters:
            scores = np.mean(scores, axis=0, keepdims=True)
        if args.avg_genomics:
            scores = np.mean(scores, axis=1, keepdims=True)

        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                target_score = scores[i, j, :]

                cluster_nb = run_args.target_nb_patches[i] if not args.avg_clusters else "avg"
                omx = run_args.omics.split(",")[j] if not args.avg_genomics else "avg"
                heatmap_save_name = '{}_cl{}_{}.png'.format(slide_id, cluster_nb, omx)

                if os.path.isfile(os.path.join(slide_save_dir, heatmap_save_name)):
                    print("Passing..")
                    pass
                else:
                    heatmap = drawHeatmap(
                        target_score, coords, slide_path, wsi_object=wsi_object, cmap=args.cmap, alpha=args.alpha, 
                        use_holes=True, binarize=False, vis_level=-1, blank_canvas=False, thresh=-1, 
                        patch_size = vis_patch_size, convert_to_percentiles=True
                    )

                    heatmap.save(os.path.join(slide_save_dir, heatmap_save_name))
                    del heatmap

    config_dict = vars(args)
    with open(os.path.join(args.save_dir, args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)


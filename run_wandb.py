import wandb
from main import run_experiment


if __name__ == "__main__":
	project_name = "SP"

	parameter_dict = {
		
		"lr":{
			"value": 2e-4
		},
		"lambda_reg":{
			"values": [1e-4, 1e-5]
		},
		"reg":{
			"values": [1e-5, 1e-4]
		},
		"gc": {
			"values": [8, 16, 32]
		},
		'embedding_dim': {
			"values": [128, 256]
		},
		'target_nb_patches': {
			"values": [[5, 30], [3, 10], [10, 50], [5, 15, 25]]
		},
		'pooling': {
			"values": ["maxpooltomin", "avgpooltomin", "maxpoolto1", "avgpoolto1", "padortruncate"]
		},
		'slide_aggregation': {
			"values": ["early", "mid", "late"]
		},
		'mlp_depth': {
			"values": [1, 2, 4]
		},
		'lr_patience': {
			"value": 7
		},
		'ff': {
			"values": [1, 2]
		},
		"data_name": {
			"value": "tcga_ov_os"
		},
		"feats_dir": {
			"value": "/media/nfs/SURV/TCGA_OV/Feats1024/CONCH/"
		},
		"coords_dir": {
			"value": "/media/nfs/SURV/TCGA_OV/SP1024/patches/"
		},
		"results_dir": {
			"value": f"./results/wandb/{project_name}"
		},
		"model_type": {
			"value": "spact"
		},
		"omics": {
			"value": "rna,dna,cnv,pro"
		},
		"fusion": {
			"values": ["bilinear", "concat"]
		},
		"selected_features": {
			"value": True
		},
		"separate_branches": {
			"values": [True, False]
		},
		"early_stopping": {
			"value": 15
		},
		"max_epochs": {
			"value": 20
		}
	}
	sweep_config = {
		'method': 'random',
		'metric': {
			'name': 'test_cindex',
			'goal': 'maximize'
		},
		'parameters': parameter_dict
	}
	sweep_id = wandb.sweep(sweep_config, project=project_name) 
	wandb.agent(sweep_id, function=run_experiment)
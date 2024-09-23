import os
import pandas as pd 
import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler

from utils.model_utils import SPACT, MIL, SNN

device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def dataloader_builder(args, cur, datasets, save=True):
	train_split, val_split, test_split = datasets
	train_survival = np.array(list(zip(train_split.slide_data["event"].values, train_split.slide_data["survival_months"].values)), dtype=[('event', bool), ('time', np.float64)])
	max_surv_limit = int(np.min([train_split.slide_data["survival_months"].max(), val_split.slide_data["survival_months"].max(), test_split.slide_data["survival_months"].max()]))
	if args.surv_model == "discrete":
		time_intervals = list(train_split.time_breaks[1:-1])
		time_intervals.append(max_surv_limit)
	else:  
		time_intervals = np.array(range(0, max_surv_limit, max_surv_limit//10))[1:]
	if save:
		save_splits(datasets, ['train', 'val', "test"], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
	
	print('Done!')
	print("Training on {} samples".format(len(train_split)))
	print("Validating on {} samples".format(len(val_split)))
	print("Testing on {} samples".format(len(test_split)))
	train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample, batch_size=args.batch_size, separate_branches=args.separate_branches)
	val_loader = get_split_loader(val_split, batch_size=args.batch_size, separate_branches=args.separate_branches)
	test_loader = get_split_loader(test_split, batch_size=args.batch_size, separate_branches=args.separate_branches)
	return train_loader, val_loader, test_loader, train_survival, time_intervals

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)

def get_simple_loader(dataset, batch_size=1):
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 
	
def get_split_loader(split_dataset, training = False, weighted = False, batch_size=1, separate_branches=False):
	"""
		return either the validation loader or training loader 
	"""
	
	collate = collate_MIL_separate if separate_branches else collate_MIL_survival

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(split_dataset)
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), drop_last = True if batch_size > 1 else False, collate_fn = collate, **kwargs)    
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), drop_last = True if batch_size > 1 else False, collate_fn = collate, **kwargs)
	else:
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)

	return loader

def collate_MIL_separate(batch):
	img = batch[0][0]
	label = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
	event_time = torch.FloatTensor([item[2] for item in batch])
	event = torch.FloatTensor([item[3] for item in batch])
	tabular = [torch.cat([item[4][i] for item in batch], dim=0).type(torch.FloatTensor) for i in range(len(batch[0][4]))]
	case_id = np.array([item[5] for item in batch])
	return [img, label, event_time, event, tabular, case_id]

def collate_MIL_survival(batch):
	img = batch[0][0]
	label = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
	event_time = torch.FloatTensor([item[2] for item in batch])
	event = torch.FloatTensor([item[3] for item in batch])
	tabular = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
	case_id = np.array([item[5] for item in batch])
	
	return [img, label, event_time, event, tabular, case_id]

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

################# MODEL BUILDER ########################
def model_builder(args, cur, ckpt_path=None):
	# Initialize the model
	# print(args.omics)
	model_dict = {
		"path_input_dim": args.path_input_dim,
		"omic_input_dim": args.omics_input_dim if args.omics is not None else None,
		"embedding_dim": args.embedding_dim,
		"fusion": args.fusion,
		"drop_out": args.drop_out,
		"n_classes": args.n_classes,
		"heads": args.mha_heads,
		"dim_head": args.dim_head,
		"mlp_skip": args.mlp_skip,
		"mlp_type": args.mlp_type,
		"mlp_depth": args.mlp_depth,
		"activation": args.activation,
		"batch_norm": True if args.batch_size > 1 else False,
		"ff": args.ff,
		"pooled_clusters": True if args.pooling in ["maxpoolto1", "avgpoolto1"] else False,
		"slide_aggregation": args.slide_aggregation,
		"nb_cluster_groups":len(args.target_nb_patches) if args.target_nb_patches else 1
	}
	if args.model_type == "spact":
		model = SPACT(**model_dict)
	elif args.model_type == "mil":
		model = MIL(**model_dict)
	elif args.model_type == "snn":
		model = SNN(**model_dict)
	else:
		raise NotImplementedError
		
	if cur == 0:
		num_params = 0
		num_params_train = 0
		print(model)
		
		for param in model.parameters():
			n = param.numel()
			num_params += n
			if param.requires_grad:
				num_params_train += n
		
		print('Total number of parameters: %.2fM' % (num_params*1e-6))
		print('Total number of trainable parameters: %.2fM' % (num_params_train*1e-6))
	
	if ckpt_path is not None:
		model.load_state_dict(torch.load(ckpt_path, weights_only=True))
	model = model.to(device)

	# Initialize the optimizer
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError

	# Initialize the LR scheduler
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-7)

	# Initialize the loss function
	if args.surv_model == "cont":
		loss_fn = CoxSurvLoss(device)
	else:
		loss_fn = NLLSurvLoss()

	# Initialize the early stopping class
	if args.early_stopping > 0:
		early_stopping = EarlyStopping(mode="min", warmup=2, patience=args.early_stopping, stop_epoch=20, verbose=True)
	else:
		early_stopping = None

	# Initialize tensorboard logger
	if not args.wandb:
		writer_dir = os.path.join(args.results_dir, str(cur))
		os.makedirs(writer_dir, exist_ok=True)
		from tensorboardX import SummaryWriter
		writer = SummaryWriter(writer_dir, flush_secs=15)
	else:
		writer = None
	
	return model, optimizer, loss_fn, scheduler, early_stopping, writer

############################# LOSS FUNCTIONS ###################
class NLLSurvLoss(nn.Module):
	"""
	The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
	Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
	Parameters
	----------
	alpha: float
		
	eps: float
		Numerical constant; lower bound to avoid taking logs of tiny numbers.
	reduction: str
		Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
	"""
	def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
		super().__init__()
		self.alpha = alpha
		self.eps = eps
		self.reduction = reduction

	def __call__(self, h, y, e):
		"""
		Parameters
		----------
		h: (n_batches, n_classes)
			The neural network output discrete survival predictions such that hazards = sigmoid(h).
		y_c: (n_batches, 2) or (n_batches, 3)
			The true time bin label (first column) and censorship indicator (second column).
		"""

		return nll_loss(h=h, y=y.unsqueeze(dim=1), e=e.unsqueeze(dim=1),
						alpha=self.alpha, eps=self.eps,
						reduction=self.reduction)


def nll_loss(h, y, e, alpha, eps=1e-7, reduction='mean'):

	
	y = y.type(torch.int64)
	e = e.type(torch.int64)

	S = torch.cumprod(1 - h, dim=1)
	
	S_padded = torch.cat([torch.ones_like(e), S], 1)
	s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
	h_this = torch.gather(h, dim=1, index=y).clamp(min=eps)
	s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

	uncensored_loss = -e * (torch.log(s_prev) + torch.log(h_this))
	censored_loss = -(1-e) * torch.log(s_this)
	
	neg_l = censored_loss + uncensored_loss
	if alpha is not None:
		loss = (1 - alpha) * neg_l + alpha * uncensored_loss
	
	if reduction == 'mean':
		loss = loss.mean()
	elif reduction == 'sum':
		loss = loss.sum()
	else:
		raise ValueError("Bad input for reduction: {}".format(reduction))
	
	return loss


class CoxSurvLoss(nn.Module):
	def __init__(self, device, eps=1e-8):
		"""
		This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
		Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
		"""
		super(CoxSurvLoss, self).__init__()
		self.device = device
		self.eps = eps

	def __call__(self, risk, t, e):
		n = len(t)
		R_mat = torch.zeros((n, n), dtype=int, device=self.device)

		# Creating the risk set matrix
		for i in range(n):
			for j in range(n):
				R_mat[i, j] = t[j] >= t[i]

		theta = risk.reshape(-1)
		exp_theta = torch.exp(theta)
		log_sum = torch.log(torch.sum(exp_theta*R_mat, dim=1) + self.eps)
		loss_cox = -torch.mean((theta - log_sum) * e)
		return loss_cox

def l1_reg_all(model):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg
############################# EARLY STOPPING ###################
class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, mode="max", warmup=5, patience=15, stop_epoch=10, verbose=False):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 20
			stop_epoch (int): Earliest epoch possible for stopping
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
		"""
		self.warmup = warmup
		self.patience = patience
		self.stop_epoch = stop_epoch
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.mode = mode

	def __call__(self, epoch, score, model, ckpt_name = 'checkpoint.pt'):

		if self.mode == "min":
			score = -score

		if epoch < self.warmup:
			pass
		elif self.best_score is None:
			self.best_score = score
			self.save_checkpoint(model, ckpt_name)
		elif score <= self.best_score or score == np.inf or np.isnan(score):
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience and epoch > self.stop_epoch:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(model, ckpt_name)
			self.counter = 0

	def save_checkpoint(self, model, ckpt_name):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Saving the best model ...')
		torch.save(model.state_dict(), ckpt_name)

#################### SURVIVAL METRICS #########################
def surv_metrics(train_survival, time_intervals, all_risk_scores, all_events, all_event_times, all_surv_probs, training=False, cidx_only=False):
	c_index = concordance_index_censored(all_events.astype(bool), all_event_times, np.squeeze(all_risk_scores), tied_tol=1e-08)[0]
	if training or cidx_only:       
		mean_auc, ibs = 0., 100
	else:
		survival = np.array(list(zip(all_events, all_event_times)), dtype=[('event', bool), ('time', np.float64)])
		try:
			_, mean_auc = cumulative_dynamic_auc(train_survival, survival, np.squeeze(all_risk_scores), time_intervals)
		except ValueError:
			mean_auc = 0.
			
		ibs = 100
		if all_surv_probs is not None:
			try:
				ibs = integrated_brier_score(train_survival, survival, all_surv_probs, time_intervals)
			except ValueError:
				ibs = 100
	return c_index, mean_auc, ibs
from argparse import Namespace
import os
import numpy as np
import torch
from torch import nn
from utils.train_utils import model_builder, dataloader_builder, surv_metrics, get_split_loader, l1_reg_all
import wandb

device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Running on: ", device)

def train(datasets: tuple, cur: int, args: Namespace):
	"""   
		train for a single fold
	"""
	print('\nTraining Fold {}!'.format(cur))
	model, optimizer, loss_fn, scheduler, early_stopping, writer = model_builder(args, cur)
	train_loader, val_loader, test_loader, train_survival, time_intervals = dataloader_builder(args, cur, datasets)
	reg_fn = l1_reg_all if args.lambda_reg > 0 else None
	train_kwargs = {
		"cur": cur,
		"model": model,
		"scheduler": scheduler,
		"optimizer": optimizer,
		"early_stopping": early_stopping,
		"writer": writer,
		"loss_fn": loss_fn,
		"reg_fn": reg_fn,
		"train_survival": train_survival,
		"time_intervals": time_intervals
	}
		
	for epoch in range(args.max_epochs):
		print('Epoch: {}/{}'.format(epoch, args.max_epochs))
		loop_survival(args, loader=train_loader, epoch=epoch, training=True, **train_kwargs)
		stop = loop_survival(args, loader=val_loader, epoch=epoch, training=False, **train_kwargs)
		if stop:
			break

	if os.path.isfile(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))):
		model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)), weights_only=True))
	else:
		"Saving the last model weights."
		torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
	results_val_dict, val_cindex, val_auc, val_ibs, val_loss = loop_survival(args, loader=val_loader, training=False, return_summary=True, **train_kwargs)
	print('Val loss: {:4f} | c-Index: {:.4f} | mean AUC: {:.4f} | mean IBS: {:.4f}'.format(val_loss, val_cindex, val_auc, val_ibs))
	log = {
		"val_loss": val_loss,
		"val_cindex": val_cindex,
		"val_auc": val_auc,
		"val_ibs": val_ibs,
	}
	
	results_test_dict, test_cindex, test_auc, test_ibs, test_loss = loop_survival(args, loader=test_loader, training=False, return_summary=True, **train_kwargs)
	print("Test loss: {:4f} | c-Index: {:4f} | mean AUC: {:.4f} | mean IBS: {:.4f}".format(test_loss, test_cindex, test_auc, test_ibs))
	log.update({
		"test_loss": test_loss,
		"test_cindex": test_cindex,
		"test_auc": test_auc,
		"test_ibs": test_ibs,
	})
	
	if writer:
		writer.add_scalar('bestval_c_index', log["val_cindex"])
		writer.add_scalar('test_c_index', log["test_cindex"])
		writer.add_scalar('bestval_loss', log["val_loss"])
		writer.add_scalar('test_loss', log["test_loss"])
		writer.add_scalar('bestval_auc', log["val_auc"])
		writer.add_scalar('test_auc', log["test_auc"])
		writer.add_scalar('bestval_ibs', log["val_ibs"])
		writer.add_scalar('test_ibs', log["test_ibs"])
		writer.close()
	
	return log, results_val_dict, results_test_dict


def loop_survival(
	args, cur, model, loader, epoch=None, scheduler=None, optimizer=None,
	early_stopping=None, writer=None, loss_fn=None, reg_fn=None, training=False, 
	return_summary=False, return_feats=False,
	train_survival=None, time_intervals=None, cidx_only=False
):
	split = "Train" if training else "Validation"
	model.train() if training else model.eval()
	loss_surv = 0.
	
	if return_summary or return_feats:
		patient_results = {}

	all_events, all_risk_scores, all_event_times = [], [], []
	if args.surv_model == "discrete":
		all_surv_probs = []
	else:
		accumulated_e, accumulated_risks, accumulated_t = [], [], []
	
	for batch_idx, (data_WSI, y_disc, event_time, event, data_tab, case_id) in enumerate(loader):
		if isinstance(data_WSI, list):
			data_WSI = [[i.to(device) for i in j] for j in data_WSI]
		else:
			data_WSI = data_WSI.to(device)

		if isinstance(data_tab, list):
			data_tab = [i.to(device) for i in data_tab]
		else:
			data_tab = data_tab.to(device)
				
		y_disc = y_disc.to(device)
		event_time = event_time.to(device)
		event = event.to(device)
		
		with torch.set_grad_enabled(training):
			if args.mode == "tab":
				hazards = model(data_tab)
			elif args.mode == "path":
				hazards = model(data_WSI)
			else:
				hazards = model(data_WSI, data_tab)

			if args.surv_model == "discrete":
				S = torch.cumprod(1 - hazards, dim=1)
				all_surv_probs.append(S.detach().cpu().numpy())
				risk = -torch.mean(S, dim=1).detach().cpu().numpy()
				loss = loss_fn(h=hazards, y=y_disc, e=event)	
			else:
				estimated_risk = hazards
				risk = estimated_risk.detach().cpu().numpy()

		reg_loss = args.lambda_reg * reg_fn(model) if reg_fn is not None else 0
		if args.surv_model == "discrete":
			if training:
				loss = loss / args.gc + reg_loss
				loss.backward()
				if (batch_idx + 1) % args.gc == 0:
					if args.grad_norm:
						nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
					optimizer.step()
					optimizer.zero_grad()
			loss_surv += loss.item()

			if args.mode != "tab" and batch_idx>0 and batch_idx % 100 == 0:
				print('batch {}, loss: {:.2f}, event: {}, event_time: {:.2f}, risk: {:.2f}'.format(batch_idx, loss.item(), event.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk)))
		else:
			accumulated_e.append(event)
			accumulated_risks.append(estimated_risk)
			accumulated_t.append(event_time)
			if len(accumulated_risks) >= args.accumulation_steps:
				risk_stack = torch.cat(accumulated_risks, dim=0)
				t_stack = torch.cat(accumulated_t, dim=0)
				e_stack = torch.cat(accumulated_e, dim=0)
				loss = loss_fn(risk_stack, t_stack, e_stack)
				
				accumulated_risks, accumulated_t, accumulated_e = [], [], []

				if training:
					loss = loss / args.accumulation_steps + reg_loss
					loss.backward()
					
					if args.grad_norm:
						nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
					optimizer.step()
					optimizer.zero_grad()
				loss_surv += loss.item()
		
		all_events.append(event.detach().cpu().numpy())
		all_event_times.append(event_time.detach().cpu().numpy())
		all_risk_scores.append(risk)
		
		if return_summary:			
			patient_results.update({
				case_id.item(): {
				'risk': risk, 
				'time': event_time.detach().cpu().numpy(), 
				'event': event.detach().cpu().numpy(),
				"risk": risk,
				"y_disc": y_disc.detach().cpu().numpy(),
				"hazards": np.squeeze(hazards.detach().cpu().numpy()) if args.surv_model == "discrete" else None
			}})

	if args.surv_model == "discrete":
		loss_surv /= len(loader)
	else:
		loss_surv /= (len(loader)/args.accumulation_steps)
	all_surv_probs = np.concatenate(all_surv_probs) if args.surv_model == "discrete" else None
	c_index, mean_auc, ibs = surv_metrics(
		train_survival, time_intervals, all_risk_scores=np.concatenate(all_risk_scores), 
		all_events=np.concatenate(all_events), all_event_times=np.concatenate(all_event_times), 
		all_surv_probs=all_surv_probs, training=training, cidx_only=cidx_only
	)
	
	if return_summary:
		return patient_results, c_index, mean_auc, ibs, loss_surv
	
	print('{}, loss: {:.4f}, c_index: {:.4f},  mean_auc: {:.4f}, ibs: {:.4f}\n'.format(split, loss_surv, c_index, mean_auc, ibs))
	
	if scheduler is not None:
		last_lr = scheduler.get_last_lr()
		if isinstance(last_lr, list):
			last_lr = last_lr[0]
	
	log_dict = {
		f'{split}/loss': loss_surv,
		f'{split}/c_index': c_index,
		f'{split}/mean_auc': mean_auc,
		f'{split}/ibs': ibs,
		f'{split}/lr': last_lr
	}
	if writer:
		for k, v in log_dict.items():
			writer.add_scalar(k, v, epoch)
	else:
		wandb.log(log_dict)
		
	if early_stopping and not training:
		assert args.results_dir
		if early_stopping.mode == "max":
			score = c_index
		else:
			score = round(loss_surv, 6)
		early_stopping(epoch, score, model, ckpt_name=os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
		if early_stopping.early_stop:
			print("Early stopping")
			return True
		if scheduler is not None:
			scheduler.step(loss_surv)
			updated_lr = scheduler.get_last_lr()
			if isinstance(updated_lr, list):
				updated_lr = updated_lr[0]
			if updated_lr < last_lr:
				print("Learning rate decreased from {:.4f} to {:.4f}".format(last_lr, updated_lr))

	return False


def eval_model(dataset, args, cur):
	"""   
		eval for a single fold
	"""
	print("Testing on {} samples".format(len(dataset)))
	test_loader = get_split_loader(dataset, batch_size=args.batch_size, separate_branches=args.separate_branches)
	model, _, loss_fn, _, _, _ = model_builder(args, cur, ckpt_path=os.path.join(args.load_from, f"s_{cur}_checkpoint.pt"))

	patient_results, cindex, _, _, loss = loop_survival(args, cur, model, test_loader, loss_fn=loss_fn, return_summary=True, cidx_only=True)
	
	print("Test c-Index: {:4f} | loss: {:4f}".format(cindex, loss))
	return patient_results, {
			'fold': int(cur), 
			'cindex': cindex, 
			"loss": loss
		}


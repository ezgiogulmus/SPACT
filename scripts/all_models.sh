#!/bin/bash
output_file="./scripts/errors/out_allmodels.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

cancer_type=$(echo "${data_name::-3}" | tr '[:lower:]' '[:upper:]')
# Omics only
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name" --data_name "$data_name" --model_type snn --mlp_depth 4 --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15"

# WSI only
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type mil --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15"

# Multi-modal
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type porpoise --fusion bilinear --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --lambda_reg 1e-5 --max_epochs 20 --early_stopping 15 --reg_type pathomic"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type amil --fusion bilinear --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type deepset --fusion bilinear --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type deepattnmisl --fusion bilinear --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type mcat --fusion concat --apply_sig --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/ --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type motcat --fusion concat --apply_sig --selected_features --omics rna,dna,cnv,pro --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping 15 --bs_micro 16384 --ot_impl pot-uot-l2 --ot_reg 0.1 --ot_tau 0.5"

python ./scripts/check_errors.py "$output_file"

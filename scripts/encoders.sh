#!/bin/bash
output_file="./scripts/errors/out_encoders.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/img_encoders/tcga_ov_os --data_name tcga_ov_os --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/CONCH/ --model_type mil"
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/CONCH/ --load_from ./results/img_encoders/tcga_ov_os/MIL_CONCH_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/img_encoders/tcga_ov_os --data_name tcga_ov_os --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --model_type mil"
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/img_encoders/tcga_ov_os/MIL_UNI_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/img_encoders/tcga_ov_os --data_name tcga_ov_os --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/PLIP/ --model_type mil"
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/PLIP/ --load_from ./results/img_encoders/tcga_ov_os/MIL_PLIP_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/img_encoders/tcga_ov_os --data_name tcga_ov_os --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/CTP/ --model_type mil"
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/CTP/ --load_from ./results/img_encoders/tcga_ov_os/MIL_CTP_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/img_encoders/tcga_ov_os --data_name tcga_ov_os --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/RESNET50/ --model_type mil"
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/RESNET50/ --load_from ./results/img_encoders/tcga_ov_os/MIL_RESNET50_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/img_encoders/tcga_ov_os --data_name tcga_ov_os --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/SSL/ --model_type mil"
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/SSL/ --load_from ./results/img_encoders/tcga_ov_os/MIL_SSL_path/run/"

python ./scripts/check_errors.py "$output_file"
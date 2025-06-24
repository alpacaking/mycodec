#!/bin/bash
# 训练FSQ模型，levels=[8,6,5,5]
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$(pwd)  python academicodec/models/encodec/main_launch.py \
  --local_rank 0 \
  --dataset_type librispeech \
  --librispeech_root_dir "./LibriSpeech" \
  --librispeech_train_subset "train-clean-100" \
  --librispeech_valid_subset "dev-clean" \
  --segment_duration_secs 1.5 \
  --sr 16000 \
  --BATCH_SIZE 8 \
  --N_EPOCHS 40 \
  --PATH "./experiments/librispeech_rvqgan_fsq8655" \
  --save_dir "./runs/librispeech_rvqgan_fsq8655" \
  --n_filters 32 \
  --D 128 \
  --ratios 8 5 4 2 \
  --target_bandwidths 1.5 3.0 6.0 \
  --use_fsq \
  --fsq_levels 8 8 8 8 \
  --LAMBDA_COM 0.0 \
  --LAMBDA_FEAT 2.0 \
  --LAMBDA_ADV 1.0 \
  --LAMBDA_REC 1.0 \
  --print_freq 100 \
  --tensorboard

echo "FSQ模型训练完成: levels=[8,6,5,5]"
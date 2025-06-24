# 设置基础参数
TEST_AUDIO_DIR="/home/chenkuangwei/chenkuangwei_nfs_data/rvqgan/codec/LibriSpeech/test-clean"
BASE_OUTPUT_DIR="evaluation_results"
SAMPLE_RATE=16000
RATIOS="8 5 4 2"
TARGET_BANDWIDTHS="1.5 3.0 6.0"
N_FILTERS=32
D=128

# 创建主输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}开始音频编解码模型评估${NC}"
echo -e "${BLUE}============================================${NC}"

# 1. 评估RVQ不同码本层数的模型
echo -e "${GREEN}1. 评估RVQ不同码本层数模型${NC}"

# RVQ n_q_max=4
echo -e "${BLUE}评估 RVQ n_q_max=4 模型${NC}"
for BW in 1.5 3.0 6.0; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/rvq_nq4_bins1024_${BW}kbps"
    echo "评估带宽: ${BW} kbps"
    
    PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
        --model_checkpoint_path experiments/librispeech_rvqgan_rqv4/2025-06-13-03-58/latest.pth \
        --test_audio_dir "$TEST_AUDIO_DIR" \
        --output_recon_dir "$OUTPUT_DIR" \
        --sample_rate $SAMPLE_RATE \
        --target_eval_bandwidth $BW \
        --n_filters $N_FILTERS \
        --D $D \
        --ratios $RATIOS \
        --target_bandwidths $TARGET_BANDWIDTHS \
        --bins 1024 \
        --n_q_max 4
done

# RVQ n_q_max=8 (基准模型)
echo -e "${BLUE}评估 RVQ n_q_max=8 模型 (基准)${NC}"
for BW in 1.5 3.0 6.0; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/rvq_nq8_bins1024_${BW}kbps"
    echo "评估带宽: ${BW} kbps"
    
    PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
        --model_checkpoint_path experiments/librispeech_rvqgan_base/2025-06-13-03-56/latest.pth \
        --test_audio_dir "$TEST_AUDIO_DIR" \
        --output_recon_dir "$OUTPUT_DIR" \
        --sample_rate $SAMPLE_RATE \
        --target_eval_bandwidth $BW \
        --n_filters $N_FILTERS \
        --D $D \
        --ratios $RATIOS \
        --target_bandwidths $TARGET_BANDWIDTHS \
        --bins 1024 \
        --n_q_max 8
done

# RVQ n_q_max=12
echo -e "${BLUE}评估 RVQ n_q_max=12 模型${NC}"
for BW in 1.5 3.0 6.0; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/rvq_nq12_bins1024_${BW}kbps"
    echo "评估带宽: ${BW} kbps"
    
    PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
        --model_checkpoint_path experiments/librispeech_rvqgan_rqv12/2025-06-13-03-58/latest.pth \
        --test_audio_dir "$TEST_AUDIO_DIR" \
        --output_recon_dir "$OUTPUT_DIR" \
        --sample_rate $SAMPLE_RATE \
        --target_eval_bandwidth $BW \
        --n_filters $N_FILTERS \
        --D $D \
        --ratios $RATIOS \
        --target_bandwidths $TARGET_BANDWIDTHS \
        --bins 1024 \
        --n_q_max 12
done

# 2. 评估RVQ不同码本大小的模型
echo -e "${GREEN}2. 评估RVQ不同码本大小模型${NC}"

# RVQ bins=512
echo -e "${BLUE}评估 RVQ bins=512 模型${NC}"
for BW in 1.5 3.0 6.0; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/rvq_nq8_bins512_${BW}kbps"
    echo "评估带宽: ${BW} kbps"
    
    PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
        --model_checkpoint_path experiments/librispeech_rvqgan_bins512/2025-06-13-03-56/latest.pth \
        --test_audio_dir "$TEST_AUDIO_DIR" \
        --output_recon_dir "$OUTPUT_DIR" \
        --sample_rate $SAMPLE_RATE \
        --target_eval_bandwidth $BW \
        --n_filters $N_FILTERS \
        --D $D \
        --ratios $RATIOS \
        --target_bandwidths $TARGET_BANDWIDTHS \
        --bins 512 \
        --n_q_max 8
done

# RVQ bins=2048
echo -e "${BLUE}评估 RVQ bins=2048 模型${NC}"
for BW in 1.5 3.0 6.0; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/rvq_nq8_bins2048_${BW}kbps"
    echo "评估带宽: ${BW} kbps"
    
    PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
        --model_checkpoint_path experiments/librispeech_rvqgan_bins2048/2025-06-13-03-57/latest.pth \
        --test_audio_dir "$TEST_AUDIO_DIR" \
        --output_recon_dir "$OUTPUT_DIR" \
        --sample_rate $SAMPLE_RATE \
        --target_eval_bandwidth $BW \
        --n_filters $N_FILTERS \
        --D $D \
        --ratios $RATIOS \
        --target_bandwidths $TARGET_BANDWIDTHS \
        --bins 2048 \
        --n_q_max 8
done

# 3. 评估FSQ模型
echo -e "${GREEN}3. 评估FSQ模型${NC}"

# FSQ [8,6,5,5]
echo -e "${BLUE}评估 FSQ [8,6,5,5] 模型${NC}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/fsq_8655"
PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
    --model_checkpoint_path experiments/librispeech_rvqgan_fsq8655/2025-06-13-03-57/latest.pth \
    --test_audio_dir "$TEST_AUDIO_DIR" \
    --output_recon_dir "$OUTPUT_DIR" \
    --sample_rate $SAMPLE_RATE \
    --n_filters $N_FILTERS \
    --D $D \
    --ratios $RATIOS \
    --target_bandwidths $TARGET_BANDWIDTHS \
    --use_fsq \
    --fsq_levels 8 6 5 5

# # FSQ [8,8,8,8]
# echo -e "${BLUE}评估 FSQ [8,8,8,8] 模型${NC}"
# OUTPUT_DIR="${BASE_OUTPUT_DIR}/fsq_8888"
# PYTHONPATH=$(pwd) python academicodec/models/encodec/evaluate.py \
#     --model_checkpoint_path experiments/librispeech_rvqgan_fsq8888/latest.pth \
#     --test_audio_dir "$TEST_AUDIO_DIR" \
#     --output_recon_dir "$OUTPUT_DIR" \
#     --sample_rate $SAMPLE_RATE \
#     --n_filters $N_FILTERS \
#     --D $D \
#     --ratios $RATIOS \
#     --target_bandwidths $TARGET_BANDWIDTHS \
#     --use_fsq \
#     --fsq_levels 8 8 8 8

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}所有评估完成!${NC}"
echo -e "${BLUE}============================================${NC}"
#!/usr/bin/env bash
# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "${SCRIPT_DIR}"

# ================= 环境配置 =================

# USE_CONDA=true
# CONDA_ENV_NAME="dreammat"
# PYTHON_BIN="python"

# if [[ "${USE_CONDA}" == "true" ]]; then
#   CONDA_BASE="$(conda info --base)"
#   source "${CONDA_BASE}/etc/profile.d/conda.sh"
#   conda activate "${CONDA_ENV_NAME}"
# fi

export CUDA_HOME=/home/pubNAS3/zhiqi/.conda/envs/dreammat
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export CPATH=$CONDA_PREFIX/include/eigen3:$CPATH
export PATH=$PATH:/home/pubNAS3/Github/tools/blender-3.2.2-linux-x64
export PATH=$PATH:/home/pubNAS3/zhiqi/Github/tools/blender-3.2.2-linux-x64
export CPATH=/usr/local/cuda/include:$CONDA_PREFIX/targets/x86_64-linux/include
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LIBRARY_PATH

# ================= 参数配置 =================

CONFIG_PATH="configs/dreammat.yaml"

# 多 GPU 配置
# GPU_IDS: 使用哪些 GPU，逗号分隔
# NUM_GPUS: 实际启用多少张卡，0 表示使用 GPU_IDS 里的全部
# WORKERS_PER_GPU: 每张卡并行多少个 DreamMat 子进程

# GPU_IDS="0,1"
# NUM_GPUS=2
# WORKERS_PER_GPU=1

# GPU_IDS="0,1,2,3"
# NUM_GPUS=4
# WORKERS_PER_GPU=1

GPU_IDS="3,4,5,6"
NUM_GPUS=4
WORKERS_PER_GPU=1

# 批量输入 TSV
BATCH_TSV="/nfshome/zhiqi/pubNAS3/Github/experiments/common_splits/test.tsv"

# 输出目录 = ${EXP_ROOT_DIR}/${EXP_NAME}
EXP_ROOT_DIR="/nfshome/zhiqi/pubNAS3/Github/experiments"
EXP_NAME="dreammat"

# manifest 输出路径；相对路径时，相对于 ${EXP_ROOT_DIR}/${EXP_NAME}
RESULT_TSV="generated_manifest.tsv"

# 使用哪个文本列：caption_short / caption_long
CAPTION_FIELD="caption_short"  # set to caption_long to match TexGaussian texverse_stage1_fidclip_v17_15epoch

# 最多处理多少条，-1 表示全部
MAX_SAMPLES=50

# DreamMat 单样本核心参数
SHAPE_INIT_PARAMS=0.8
MAX_STEPS=3000
BLENDER_GENERATE=True
TEXTURE_SIZE=1024

# 目录组织
BATCH_WORK_DIR_NAME="__runs__"
BATCH_TEXTURES_DIR_NAME="textures"

# 是否保留每个样本的中间 DreamMat 实验目录
KEEP_INTERMEDIATE=True

# 其他需要透传给 launch.py 的覆写项，按需继续追加
EXTRA_OVERRIDES=(
  "exp_root_dir=${EXP_ROOT_DIR}"
  "name=${EXP_NAME}"
  "system.geometry.shape_init_params=${SHAPE_INIT_PARAMS}"
  "trainer.max_steps=${MAX_STEPS}"
  "data.blender_generate=${BLENDER_GENERATE}"
  "system.exporter.texture_size=${TEXTURE_SIZE}"
)

# ================= 打印配置 =================

OUTPUT_ROOT="${EXP_ROOT_DIR}/${EXP_NAME}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
GPU_ID_COUNT="${#GPU_ID_ARRAY[@]}"
if [[ "${NUM_GPUS}" -gt 0 && "${NUM_GPUS}" -lt "${GPU_ID_COUNT}" ]]; then
  EFFECTIVE_GPU_COUNT="${NUM_GPUS}"
else
  EFFECTIVE_GPU_COUNT="${GPU_ID_COUNT}"
fi
TOTAL_WORKERS=$(( EFFECTIVE_GPU_COUNT * WORKERS_PER_GPU ))

echo "Starting DreamMat Batch Export..."
echo "Config:            ${CONFIG_PATH}"
echo "GPU IDs:           ${GPU_IDS}"
echo "Num GPUs:          ${NUM_GPUS}"
echo "Workers / GPU:     ${WORKERS_PER_GPU}"
echo "Total Workers:     ${TOTAL_WORKERS}"
echo "Batch TSV:         ${BATCH_TSV}"
echo "Caption Field:     ${CAPTION_FIELD}"
echo "Max Samples:       ${MAX_SAMPLES}"
echo "Output Root:       ${OUTPUT_ROOT}"
echo "Result TSV:        ${RESULT_TSV}"
echo "Shape Init Params: ${SHAPE_INIT_PARAMS}"
echo "Max Steps:         ${MAX_STEPS}"
echo "Blender Generate:  ${BLENDER_GENERATE}"
echo "Texture Size:      ${TEXTURE_SIZE}"
echo "Keep Intermediate: ${KEEP_INTERMEDIATE}"
echo "Textures Dir:      ${OUTPUT_ROOT}/${BATCH_TEXTURES_DIR_NAME}"

# ================= 执行命令 =================

if [[ "${KEEP_INTERMEDIATE,,}" == "true" ]]; then
  python launch.py \
    --config "${CONFIG_PATH}" \
    --train \
    --gpu "${GPU_IDS%%,*}" \
    --batch_tsv "${BATCH_TSV}" \
    --caption_field "${CAPTION_FIELD}" \
    --max_samples "${MAX_SAMPLES}" \
    --result_tsv "${RESULT_TSV}" \
    --gpu_ids "${GPU_IDS}" \
    --num_gpus "${NUM_GPUS}" \
    --workers_per_gpu "${WORKERS_PER_GPU}" \
    --batch_work_dir_name "${BATCH_WORK_DIR_NAME}" \
    --batch_textures_dir_name "${BATCH_TEXTURES_DIR_NAME}" \
    --keep_intermediate \
    "${EXTRA_OVERRIDES[@]}"
else
  python launch.py \
    --config "${CONFIG_PATH}" \
    --train \
    --gpu "${GPU_IDS%%,*}" \
    --batch_tsv "${BATCH_TSV}" \
    --caption_field "${CAPTION_FIELD}" \
    --max_samples "${MAX_SAMPLES}" \
    --result_tsv "${RESULT_TSV}" \
    --gpu_ids "${GPU_IDS}" \
    --num_gpus "${NUM_GPUS}" \
    --workers_per_gpu "${WORKERS_PER_GPU}" \
    --batch_work_dir_name "${BATCH_WORK_DIR_NAME}" \
    --batch_textures_dir_name "${BATCH_TEXTURES_DIR_NAME}" \
    "${EXTRA_OVERRIDES[@]}"
fi

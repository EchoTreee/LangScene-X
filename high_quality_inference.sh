#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash high_quality_inference.sh <first_image> <last_image> [work_root]" >&2
    exit 1
fi

first_image="$1"
last_image="$2"
work_root="${3:-demo_hq}"

sam_model_path="./sam_vit_h_4b8939.pth"
sam2_model_path="./sam2_hiera_large.pt"
model_path="./CogVideoX-ft"

rgb_dir="${work_root}/rgb"
seg_dir="${work_root}/seg"
normal_dir="${work_root}/normal"
video_root="${work_root}/video"
data_dir="${work_root}/data"
output_dir="${work_root}/output_paper_like"

mkdir -p "$rgb_dir"
cp "$first_image" "${rgb_dir}/0001.png"
cp "$last_image" "${rgb_dir}/0002.png"

echo "[1/6] Generating segmentation priors..."
python auto-seg/auto-mask-align.py \
    --sam1_checkpoint "$sam_model_path" \
    --sam2_checkpoint "$sam2_model_path" \
    --video_path "$rgb_dir" \
    --output_dir "$seg_dir" \
    --level "default"

echo "[2/6] Generating normal priors..."
python get_normal.py --base_path "$work_root"

echo "[3/6] Synthesizing RGB video..."
python video_inference.py \
    --model_path "$model_path" \
    --output_dir "${video_root}/rgb" \
    --first_image "${rgb_dir}/0001.png" \
    --last_image "${rgb_dir}/0002.png"

echo "[4/6] Synthesizing segmentation video..."
python video_inference.py \
    --model_path "$model_path" \
    --output_dir "${video_root}/seg" \
    --first_image "${seg_dir}/0001.png" \
    --last_image "${seg_dir}/0002.png"

echo "[5/6] Synthesizing normal video..."
python video_inference.py \
    --model_path "$model_path" \
    --output_dir "${video_root}/normal" \
    --first_image "${normal_dir}/0001.png" \
    --last_image "${normal_dir}/0002.png"

mkdir -p "$data_dir"
cp "${seg_dir}/colors.npy" "$data_dir"

echo "[6/6] Running high-quality field construction..."
python entry_point.py \
    pipeline.rgb_video_path="${video_root}/rgb/video_ckpt_800.mp4" \
    pipeline.normal_video_path="${video_root}/normal/video_ckpt_800.mp4" \
    pipeline.seg_video_path="${video_root}/seg/video_ckpt_800.mp4" \
    pipeline.data_path="$data_dir" \
    gaussian.dataset.source_path="$data_dir" \
    gaussian.dataset.model_path="$output_dir" \
    pipeline.selection=True \
    pipeline.chunk_num=8 \
    pipeline.keep_num_per_chunk=5 \
    gaussian.opt.max_geo_iter=1500 \
    gaussian.opt.normal_optim=False \
    gaussian.opt.optim_pose=True \
    pipeline.skip_video_process=False \
    pipeline.skip_pose_estimate=False \
    pipeline.skip_lang_feature_extraction=False

echo
echo "High-quality inference finished."
echo "Key outputs:"
echo "  ${output_dir}/chkpnt12000.pth"
echo "  ${output_dir}/point_cloud/iteration_12000/point_cloud.ply"
echo "  ${output_dir}/valid/"
echo
echo "Optional final render:"
echo "  python entry_point.py \\"
echo "    pipeline.mode=render \\"
echo "    pipeline.load_iteration=12000 \\"
echo "    pipeline.data_path=${data_dir} \\"
echo "    gaussian.dataset.source_path=${data_dir} \\"
echo "    gaussian.dataset.model_path=${output_dir} \\"
echo "    gaussian.opt.optim_pose=True"

# LangScene-X Inference 技术文档

这份文档描述的是当前仓库里已经实际跑通的一条推理链路：从两张输入图片出发，生成 RGB / segmentation / normal 三路视频，再完成位姿估计、OpenSeg 语言特征提取、AE 压缩和 3DGS 场重建。

本文档只写实际验证过的流程，不写论文层面的背景介绍。

## 1. 目标与最终产物

完整 inference 跑通后，会得到两类结果：

- 中间数据：
  - `demo/video/rgb/video_ckpt_800.mp4`
  - `demo/video/seg/video_ckpt_800.mp4`
  - `demo/video/normal/video_ckpt_800.mp4`
  - `demo/data/camera/*.npz`
  - `demo/data/lang_features/*.npy`
  - `demo/data/lang_features_dim3/*_f.npy`
  - `demo/data/ckpt/best_ckpt.pth`
- 最终 3DGS 结果：
  - `demo/output_real_openseg/chkpnt12000.pth`
  - `demo/output_real_openseg/point_cloud/iteration_12000/point_cloud.ply`
  - `demo/output_real_openseg/app_model/iteration_12000/app.pth`
  - `demo/output_real_openseg/pose/iter_12000/pose_optimized.npy`

## 2. 当前验证通过的环境

当前仓库已经验证通过的环境是：

- Python: `3.10`
- Conda 环境: `langscenex_bw`
- PyTorch: `2.11.0+cu128`
- TorchVision: `0.26.0+cu128`
- TorchAudio: `2.11.0+cu128`
- GPU: Blackwell

如果你不是 Blackwell，可以不完全照搬这套环境；如果你是 Blackwell，优先沿用这套。

## 3. 需要保留的权重和模型

以下内容是完整跑通 inference 所必需的：

- `sam_vit_h_4b8939.pth`
- `sam2_hiera_large.pt`
- `CogVideoX-ft/`
- `model_zoo/openseg_exported_clip/`

其中：

- `sam_vit_h_4b8939.pth` 和 `sam2_hiera_large.pt` 用于分割
- `CogVideoX-ft/` 用于三路视频插帧
- `model_zoo/openseg_exported_clip/` 用于语言特征提取

`OpenSeg` 目录必须是 TensorFlow SavedModel 结构，至少包含：

```text
model_zoo/openseg_exported_clip/
├── saved_model.pb
├── graph_def.pbtxt
└── variables/
```

仓库当前配置已经把 OpenSeg 路径指到：

- [configs/field_construction.yaml](/workspace/projects/LangScene-X/configs/field_construction.yaml)

即：

```yaml
feature_extractor:
  type: "open-seg"
  model_path: "/workspace/projects/LangScene-X/model_zoo/openseg_exported_clip"
```

## 4. 启动前准备

先激活工作环境，并把 PyTorch / NVIDIA runtime 的动态库路径补进当前 shell：

```bash
conda activate langscenex_bw
source /workspace/projects/LangScene-X/langscenex_env.sh
cd /workspace/projects/LangScene-X
```

如果这里没做，后面有概率遇到 `libc10.so`、`libtorch_cuda.so` 一类的运行时加载错误。

## 5. 最小环境自检

在正式跑 inference 之前，建议先确认三件事：

### 5.1 PyTorch CUDA 是否正常

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("ok:", y.shape, y.device)
PY
```

### 5.2 本地 CUDA 扩展是否可导入

```bash
python - <<'PY'
import simple_knn._C
import diff_LangSurf_rasterization._C
print("extensions ok")
PY
```

### 5.3 OpenSeg SavedModel 是否存在

```bash
test -f model_zoo/openseg_exported_clip/saved_model.pb && echo ok
```

## 6. 最简单的完整入口

仓库自带的一键脚本是：

- [quick_start.sh](/workspace/projects/LangScene-X/quick_start.sh)

它的输入是两张 RGB 图：

```bash
chmod +x quick_start.sh
./quick_start.sh <first_rgb_image_path> <last_rgb_image_path>
```

例如：

```bash
./quick_start.sh my_inputs/1.jpg my_inputs/2.jpg
```

这个脚本会顺序执行：

1. 把输入图复制到 `demo/rgb/0001.png` 和 `demo/rgb/0002.png`
2. 用 SAM + SAM2 生成 `demo/seg`
3. 用 StableNormal 生成 `demo/normal`
4. 分别对 RGB / seg / normal 做视频插帧
5. 调 `entry_point.py` 跑位姿估计、OpenSeg、AE 和 3DGS

## 7. 推荐的手动执行版

如果你要更清楚地观察每一阶段，建议按下面顺序手动执行。这样出错时更容易定位。

### 7.1 准备输入

```bash
mkdir -p demo/rgb
cp my_inputs/1.jpg demo/rgb/0001.png
cp my_inputs/2.jpg demo/rgb/0002.png
```

### 7.2 生成分割图

```bash
python auto-seg/auto-mask-align.py \
  --sam1_checkpoint ./sam_vit_h_4b8939.pth \
  --sam2_checkpoint ./sam2_hiera_large.pt \
  --video_path demo/rgb \
  --output_dir demo/seg \
  --level default
```

成功后应看到：

- `demo/seg/0001.png`
- `demo/seg/0002.png`
- `demo/seg/colors.npy`

### 7.3 生成法线图

```bash
python get_normal.py --base_path demo
```

成功后应看到：

- `demo/normal/0001.png`
- `demo/normal/0002.png`

### 7.4 生成三路视频

```bash
python video_inference.py \
  --model_path ./CogVideoX-ft \
  --output_dir demo/video/rgb \
  --first_image demo/rgb/0001.png \
  --last_image demo/rgb/0002.png

python video_inference.py \
  --model_path ./CogVideoX-ft \
  --output_dir demo/video/seg \
  --first_image demo/seg/0001.png \
  --last_image demo/seg/0002.png

python video_inference.py \
  --model_path ./CogVideoX-ft \
  --output_dir demo/video/normal \
  --first_image demo/normal/0001.png \
  --last_image demo/normal/0002.png
```

成功后应看到：

- `demo/video/rgb/video_ckpt_800.mp4`
- `demo/video/seg/video_ckpt_800.mp4`
- `demo/video/normal/video_ckpt_800.mp4`

### 7.5 准备 field construction 输入目录

```bash
mkdir -p demo/data
cp demo/seg/colors.npy demo/data/
```

### 7.6 跑完整 field construction

这是当前已经实际验证通过的一条命令。它会：

- 使用 `VGGT` 做位姿估计
- 使用 `OpenSeg` 提取语言特征
- 训练 per-scene AE，把语言特征压到 3 维
- 启动 3DGS 训练并保存 checkpoint

```bash
python entry_point.py \
  pipeline.rgb_video_path="demo/video/rgb/video_ckpt_800.mp4" \
  pipeline.normal_video_path="demo/video/normal/video_ckpt_800.mp4" \
  pipeline.seg_video_path="demo/video/seg/video_ckpt_800.mp4" \
  pipeline.data_path="demo/data" \
  gaussian.dataset.source_path="demo/data" \
  gaussian.dataset.model_path="demo/output_real_openseg" \
  pipeline.selection=False \
  gaussian.opt.max_geo_iter=1500 \
  gaussian.opt.normal_optim=True \
  gaussian.opt.optim_pose=False \
  pipeline.skip_video_process=False \
  pipeline.skip_pose_estimate=False \
  pipeline.skip_lang_feature_extraction=False
```

说明：

- `gaussian.dataset.model_path="demo/output_real_openseg"` 是推荐写法，避免和旧结果混在 `demo/output`
- `pipeline.skip_video_process=False` 表示由 `entry_point.py` 自己从三路视频继续处理
- `pipeline.skip_pose_estimate=False` 表示走 `VGGT`
- `pipeline.skip_lang_feature_extraction=False` 表示走真实 `OpenSeg`

### 7.7 更贴近论文效果的高质量入口

如果你的目标不是“先跑通”，而是尽量贴近论文/项目主页里的公开效果，优先不要继续使用默认 `quick_start.sh` 的那组开关。

仓库默认配置和当前训练代码更一致的做法是：

- 打开位姿优化：`gaussian.opt.optim_pose=True`
- 关闭 normal prior 直接监督：`gaussian.opt.normal_optim=False`
- 打开视角筛选：`pipeline.selection=True`

仓库根目录已经提供了一条更适合高质量复现的脚本：

```bash
bash high_quality_inference.sh <first_image> <last_image> [work_root]
```

例如：

```bash
bash high_quality_inference.sh demo_examples/chair_1.png demo_examples/chair_2.png demo_hq
```

这条脚本会：

- 生成 RGB / segmentation / normal 三路视频
- 跑 `VGGT` 位姿估计
- 跑真实 `OpenSeg`
- 启用位姿优化
- 使用位姿阶段输出的高置信帧索引做视角筛选
- 将结果输出到 `demo_hq/output_paper_like`

注意：

- 这能显著拉近公开仓库版本的上限，但不能保证完全等同于论文主页展示效果
- 项目主页里的展示结果可能依赖尚未完全公开的训练权重或更强的插帧/AE 版本
- 因此高质量复现的目标应理解为“尽量逼近公开版本上限”，而不是机械地要求与主页逐帧一致

## 8. 各阶段的实际输出

### 8.1 位姿估计

位姿估计完成后，通常会产生：

- `demo/data/camera/*.npz`
- `demo/data/render_camera/*.npz`

当前仓库已经去掉了对旧 `dust3r` 路径的必须依赖，默认走 `VGGT`。

### 8.2 OpenSeg 特征提取

真实 `OpenSeg` 提取完成后会生成：

- `demo/data/lang_features/0001.npy`
- ...
- `demo/data/lang_features/0049.npy`

### 8.3 AE 压缩

AE 训练完成后会生成：

- `demo/data/ckpt/best_ckpt.pth`
- `demo/data/lang_features_dim3/0001_f.npy`
- ...
- `demo/data/lang_features_dim3/0049_f.npy`

另外还有每帧对应的：

- `demo/data/lang_features_dim3/0001_s.npy`
- ...
- `demo/data/lang_features_dim3/0049_s.npy`

### 8.4 3DGS 训练

3DGS 会在这些迭代点保存结果：

- `100`
- `500`
- `1000`
- `2000`
- `5000`
- `10000`
- `12000`

最终结果目录是：

- `demo/output_real_openseg`

其中最关键的是：

- `chkpnt12000.pth`
- `point_cloud/iteration_12000/point_cloud.ply`
- `app_model/iteration_12000/app.pth`

## 9. 实际跑通过程中遇到的关键坑

### 9.1 Blackwell 不要继续混用旧 12.4 运行时

如果你是 Blackwell，重点不是“能不能装上 torch”，而是运行时是否真的支持当前架构。当前已经验证通过的是：

- `torch 2.11.0+cu128`
- `langscenex_bw`

### 9.2 CUDA 扩展必须重编

这两个扩展必须能正常导入：

- `simple_knn._C`
- `diff_LangSurf_rasterization._C`

如果 torch / CUDA 版本换了，旧 `.so` 不能直接复用。

### 9.3 OpenSeg 权重不是仓库自带

必须额外准备：

- `model_zoo/openseg_exported_clip/`

没有这一步，`entry_point.py` 会在语言特征提取时报：

```text
OSError: SavedModel file does not exist at: .../openseg_exported_clip/
```

### 9.4 `quick_start.sh` 默认输出到 `demo/output`

为了避免和历史结果混淆，推荐手动执行第 7.6 节中的命令，把输出定到：

- `demo/output_real_openseg`

### 9.5 下载很慢是正常现象

完整跑通过程中会触发多次大模型下载，包括：

- StableNormal
- DINOv2
- VGGT

第一次跑时耗时主要经常不在计算，而在下载。

## 10. 我建议的复现顺序

最稳的做法是：

1. 激活 `langscenex_bw` 并 `source langscenex_env.sh`
2. 跑第 5 节的最小自检
3. 手动完成第 7.2 到 7.4，确认三路视频都成功
4. 再跑第 7.6 的 `entry_point.py`
5. 最后检查 `demo/output_real_openseg/chkpnt12000.pth`

## 11. 最终验收标准

满足下面这些条件，才算完整 inference 真正跑通：

```text
demo/video/rgb/video_ckpt_800.mp4 存在
demo/video/seg/video_ckpt_800.mp4 存在
demo/video/normal/video_ckpt_800.mp4 存在
demo/data/lang_features/ 有 49 个 .npy
demo/data/lang_features_dim3/ 有 49 个 *_f.npy
demo/data/ckpt/best_ckpt.pth 存在
demo/output_real_openseg/chkpnt12000.pth 存在
demo/output_real_openseg/point_cloud/iteration_12000/point_cloud.ply 存在
demo/output_real_openseg/app_model/iteration_12000/app.pth 存在
```

可以直接用下面的命令做一次快速检查：

```bash
test -f demo/video/rgb/video_ckpt_800.mp4 &&
test -f demo/video/seg/video_ckpt_800.mp4 &&
test -f demo/video/normal/video_ckpt_800.mp4 &&
test -f demo/data/ckpt/best_ckpt.pth &&
test -f demo/output_real_openseg/chkpnt12000.pth &&
test -f demo/output_real_openseg/point_cloud/iteration_12000/point_cloud.ply &&
test -f demo/output_real_openseg/app_model/iteration_12000/app.pth &&
echo "inference complete"
```

## 12. 相关文件

- [quick_start.sh](/workspace/projects/LangScene-X/quick_start.sh)
- [entry_point.py](/workspace/projects/LangScene-X/entry_point.py)
- [configs/field_construction.yaml](/workspace/projects/LangScene-X/configs/field_construction.yaml)
- [langscenex_env.sh](/workspace/projects/LangScene-X/langscenex_env.sh)
- [BLACKWELL_MIGRATION.md](/workspace/projects/LangScene-X/BLACKWELL_MIGRATION.md)

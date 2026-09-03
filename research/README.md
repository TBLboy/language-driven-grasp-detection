# Research Materials Archive

本目录用于保存第一阶段调研中用到的资料，避免依赖 `/tmp` 中的临时文件。

## Diffusion baseline smoke

`scripts/diffusion_smoke.py` 用真实 Grasp-Anything++ stem 跑官方 `LGDM`
diffusion baseline 的工程链路：

```bash
HF_ENDPOINT=https://hf-mirror.com \
PYTHONNOUSERSITE=1 \
python \
  research/scripts/diffusion_smoke.py \
  --max-stems 2 \
  --sample-steps 10 \
  --out outputs/diffusion_smoke_2
```

默认使用 GPU；完整官方 diffusion 为 1000 步，smoke 的
`p_sample_loop` 使用 10 步 respaced cosine schedule，只为验证
dataset -> dense maps -> forward -> loss -> backward ->
sampling -> post-process -> IoU evaluation 全部可执行。

当前确认结果：

- 2/2 个不同 scene 的真实样本执行成功
- `sample_shape=(1,1,224,224)`，sample 数值 finite
- 官方 `train_network_diffusion.py` 计算 diffusion loss 但未对 diffusion
  loss 调 backward；其实际 backward 目标是 dense-map loss
- 官方 README 示例里的 `--network lgd` 未在 `get_network` 中注册，
  LGDM 的注册名为 `lgdm`

## 1000-sample formal experiments

800 train / 200 val / 20 epochs / 10-step diffusion sampling on 1000 real
cross-scene stems:

| condition | single eval | mean over 3 sampling seeds |
| --- | ---: | ---: |
| `none` (official LGDM) | 33/200 | 37.0 |
| `plain-y` | 37/200 | 40.0 |
| `lsar` (learnable scale) | 13/200 | - |
| `lsar` fixed scale 0.05 (`lsar_tuned`) | 39/200 | 38.3 |

The learnable `scale` grows to `0.224` and hurts the model; fixing it to
`0.05` restores stability. `lsar_tuned` is slightly above the official
baseline and close to raw `y` injection. These numbers are engineering
evidence, not a performance claim.

Run commands:

```bash
PYTHONNOUSERSITE=1 \
python \
  research/scripts/train_lgdm_clean.py \
  --stems-tsv research/smoke-data/train_subset_1000.tsv \
  --out outputs/lgdm_exp1000/lsar_tuned \
  --epochs 20 --train-ratio 0.8 --batch-size 2 --eval-steps 10 \
  --condition-mode lsar --lsar-scale 0.05 --lsar-fixed-scale \
  --lsar-affordance-weight 0.1 --seed 42
```

Repeated evaluation and qualitative rendering:

```bash
PYTHONNOUSERSITE=1 \
python \
  research/scripts/eval_lgdm_checkpoint.py \
  --checkpoint outputs/lgdm_exp1000/lsar_tuned/last.pt \
  --stems-tsv research/smoke-data/train_subset_1000.tsv \
  --out outputs/lgdm_exp1000/lsar_tuned_repeat_eval \
  --repeats 3 --start-seed 100

PYTHONNOUSERSITE=1 \
python \
  research/scripts/visualize_lgdm_samples.py \
  --checkpoint outputs/lgdm_exp1000/lsar_tuned/last.pt \
  --stems-tsv research/smoke-data/train_subset_1000.tsv \
  --out outputs/lgdm_exp1000/visuals --n-samples 6
```

## 5000-sample validation

5000 stems / 5000 unique scenes / 4000 train / 1000 val / 15 epochs /
batch 2 / 10-step diffusion sampling. Same training config for both rows.

| Method | single eval /1000 | 3-repeat mean /1000 | std |
| --- | ---: | ---: | ---: |
| Official LGDM (`none`) | 211 | 202.7 | 4.93 |
| LSAR-full (ours) | 299 | 309.7 | 3.51 |

Prepare the larger subset with the RGB scene list from the local archive:

```bash
PYTHONNOUSERSITE=1 \
python \
  research/scripts/prepare_training_subset.py \
  --positive-dir /mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive \
  --instruction-dir /mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions \
  --scene-list research/smoke-data/image_scenes_full.txt \
  --out research/smoke-data/train_subset_5k.tsv \
  --num-stems 5000 --max-per-scene 1 --seed 42
```

Train and repeat-evaluate with the same commands used for the 2968-sample
rows, changing only `--stems-tsv` and `--out`. Qualitative output with
affordance overlay is in `outputs/lgdm_5k/visuals_affordance/qualitative.png`.

## Clean LGDM 100-sample sanity

准备 100 个不同 scene 的真实 stem：

```bash
PYTHONNOUSERSITE=1 \
python \
  research/scripts/prepare_training_subset.py \
  --positive-dir /mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive \
  --instruction-dir /mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions \
  --out research/smoke-data/train_subset_100.tsv \
  --num-stems 100 --max-per-scene 1 --seed 42
```

只解压这 100 张对应 RGB，不展开完整 image zip：

```bash
./research/scripts/extract_rgb_subset.sh \
  --stems research/smoke-data/train_subset_100.tsv
```

跑 Clean LGDM training sanity：

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
python \
  research/scripts/train_lgdm_clean.py \
  --stems-tsv research/smoke-data/train_subset_100.tsv \
  --out outputs/train_lgdm_clean_100 \
  --epochs 20 --train-ratio 0.8 --batch-size 2 --eval-steps 10
```

当前 sanity 结果：

- 80 train / 20 val，20 epochs，40 batches/epoch
- `outputs/train_lgdm_clean_100/last.pt` 可经 `--resume` 加载
- 10-step respaced sampling 评估：`5/20 correct`
- 这是工程验证，不是性能结论；官方 contrast 项主导 clean loss

## LGDM tensor-flow debug

官方 forward：

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
python \
  research/scripts/lgdm_tensorflow_debug.py \
  --out outputs/lgdm_tensorflow_debug_official.json
```

`--inject-y` 对照：

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
python \
  research/scripts/lgdm_tensorflow_debug.py \
  --inject-y \
  --out outputs/lgdm_tensorflow_debug_inject_y.json
```

结论：官方 `image_atts` 为全 1 mask；`y` 分支无梯度；把
`y_view (8x19x19)` 加入 GG-CNN `conv3` 后文本分支梯度可传播，因此
LSAR V1 固定在 ALBEF `y` 到 `conv3` 的 conditioning 分支。

## LSAR V1 minimal experiment

LSAR V1 实现在 `models/lgdm_lsar.py`，不改官方 `LGD-main` 文件：

- `condition_mode=none`：官方 LGDM，`y` 不注入
- `condition_mode=plain-y`：直接把 `y_view` 加进 `conv3`
- `condition_mode=lsar`：先用 `SpatialAffordanceRefinement` 生成残差，
  再以可学习 `scale` 注入 `conv3`

训练命令：

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
python \
  research/scripts/train_lgdm_clean.py \
  --stems-tsv research/smoke-data/train_subset_100.tsv \
  --out outputs/lgdm_exp100/lsar_scale01 \
  --epochs 20 --train-ratio 0.8 --batch-size 2 \
  --eval-steps 10 --condition-mode lsar --seed 42
```

最小三条件结果（100 样本、80/20、20 epochs、10-step sampling）：

| condition | eval correct |
| --- | --- |
| none | 4/20 |
| plain-y | 4/20 |
| lsar | 4/20 |

限制：100 样本 20 epochs 不构成性能结论；它的作用是证明 Baseline /
plain-y ablation / LSAR 使用同一数据、训练目标和评估协议可复现比较。
训练后 LSAR `scale` 为 0.158，说明模块确实参与计算，不是 no-op。

## 来源与归档

| 资料 | 原位置 | 归档位置 | 说明 |
| --- | --- | --- | --- |
| CVPR 2024 LGD 论文 PDF | `/tmp/LGD_CVPR2024.pdf` | `paper/LGD_CVPR2024.pdf` | 官方论文 |
| CVPR 2024 LGD 论文文本 | `/tmp/LGD_CVPR2024.txt` | `paper/LGD_CVPR2024.txt` | 便于检索/引用 |
| plusplus instruction 样本 | `/tmp/instruction_sample.pkl` | `data-samples/instruction_sample.pkl` | 实测为纯字符串 |
| plusplus positive grasp 样本 | `/tmp/positive_sample.pt` | `data-samples/positive_sample.pt` | shape `[3,6]` float32 |
| plusplus part mask 样本 | `/tmp/mask_sample.npy` | `data-samples/mask_sample.npy` | `(416,416)` uint8 |
| base scene description 样本 | `/tmp/ga_scene_description_sample.pkl` | `data-samples/ga_scene_description_sample.pkl` | `(text, [object_names])` |
| base negative grasp 样本 | `/tmp/ga_label_negative_sample.pt` | `data-samples/ga_label_negative_sample.pt` | shape `[39,6]` |
| base object mask 样本 | `/tmp/ga_mask_sample.npy` | `data-samples/ga_mask_sample.npy` | `(416,416)` uint8 |
| plusplus zip 尾部中央目录证据 | `/tmp/*_tail.bin` | `hf-zip-evidence/` | 用于确认 zip 内部文件名 |
| base zip 尾部中央目录证据 | `/tmp/base_*_tail.bin` | `hf-zip-evidence/` | 用于确认 zip 内部文件名 |
| 原始 Grasp-Anything 代码 | `/tmp/grasp-lgd-research/grasp-anything-fsoft` | `reference-code/Grasp-Anything/` | `Fsoft-AIC/Grasp-Anything` 快照，不含 `.git` |
| Grasp-Anything 官网/docs 仓库 | `/tmp/grasp-lgd-research/grasp-anything-airvlab` | `reference-code/grasp-anything-airvlab/` | `airvlab/grasp-anything` 快照，不含 `.git` |
| LGD 官方 baseline 代码 | `/tmp/grasp-lgd-research/LGD` | 根目录 `LGD-main/` | 已先在工作空间保存的完整快照，不再重复归档 |

## 官方来源

- Grasp-Anything++：`https://huggingface.co/datasets/airvlab/Grasp-Anything-pp`
- Grasp-Anything：`https://huggingface.co/datasets/airvlab/Grasp-Anything`
- LGD 代码：`https://github.com/Fsoft-AIC/LGD`
- Grasp-Anything 代码：`https://github.com/Fsoft-AIC/Grasp-Anything`
- 项目官网：`https://airvlab.github.io/grasp-anything/`

## 使用注意

- `data-samples/` 只用于格式调查，不代表完整数据集。
- `reference-code/` 保留为第三方代码快照，后续实现中如需复用，应在自己的模块里明确引用来源。
- 本目录是调研资料归档，不是最终提交代码结构；最终 GitHub repo 整理时可再决定保留或排除。

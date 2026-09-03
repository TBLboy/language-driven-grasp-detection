# Final Method Summary: LGDM + LSAR (10k)

> 状态：experiment evidence frozen
> 日期：2026-09-03 至 2026-09-04
> 用途：论文主表、ablation 表、sampling-step 表与 qualitative figure 的后续材料来源

## 1. Final Method

- Baseline: official diffusion-based `LGDM`
- Method: `LGDM + LSAR`
- LSAR 行为: language-conditioned spatial residual injected before the GG-CNN
  decoder; fixed residual scale `0.01`
- Auxiliary spatial affordance loss weight: `lambda_aff = 0.05`
- Two training seeds: `seed=42, split-seed=42` and `seed=43, split-seed=42`

```text
Image + instruction
  -> ALBEF fusion -> y_view
  -> LSAR residual with fixed scale 0.01
  -> GG-CNN decoder -> diffusion grasp generation
  -> (x, y, w, h, theta)
```

## 2. Dataset and Training

- Stems: `research/smoke-data/train_subset_10k.tsv`
- 10000 stems / 10000 unique scenes
- RGB source: Grasp-Anything `<scene>.jpg`, 10000/10000 available
- Split: 8000 train / 2000 validation, seed 42
- Scene-disjoint by construction (each stem uses a unique scene)
- Epochs: 15
- Batch size: 2
- Learning rate: 1e-3
- Weight decay: 1e-4
- Evaluation diffusion steps: 10

## 3. Repeated Evaluation

Three 10-step diffusion sampling runs per checkpoint, sampling seeds 100/101/102.

| Method | Config | Single eval /2000 | Repeat mean /2000 | Std |
| --- | --- | ---: | ---: | ---: |
| LGDM baseline | `condition-mode none` | 449 | 470.0 | 10.82 |
| LSAR-full | `lambda_aff=0.1` | 625 | 605.3 | 18.88 |
| LSAR-no-aff | `lambda_aff=0.0` | 643 | 653.7 | 6.51 |
| LSAR final seed42 | `lambda_aff=0.05`, `seed=42` | 686 | 678.0 | 16.52 |
| LSAR final seed43 | `lambda_aff=0.05`, `seed=43` | 666 | 661.7 | 14.01 |

### lambda_aff sweep

| lambda_aff | Repeat mean /2000 | Note |
| ---: | ---: | --- |
| 0.0 | 653.7 | no-aff ablation |
| 0.01 | not repeated | single eval 662/2000 |
| 0.05 | 678.0 | final method, seed42 |
| 0.1 | 605.3 | LSAR-full ablation |

## 4. Sampling-Step Sensitivity

Fixed 200-sample validation subset, `subsample_seed=7`, sampling seed 200.

| Method | 10 steps | 50 steps |
| --- | ---: | ---: |
| LGDM baseline | 39 | 44 |
| LSAR-full | 55 | 61 |
| LSAR-no-aff | 57 | 65 |
| LSAR final (`lambda_aff=0.05`) | 65 | 67 |

Increasing sampling steps improves all settings and preserves the LSAR ordering.

## 5. Qualitative Visualization

- Final method paired figure:
  `outputs/lgdm_10k/paired_visuals_final_005/paired_qualitative.png`
- Paper asset copy:
  `research/assets/qualitative_10k_paired_lsar_0.05.png`
- The figure renders 6 validation samples: pen ink/lead, highlighter bright
  color, marker cap, duck bill, apple stem, and keychain keys.

## 6. Conclusion for Current Experimental Stage

At the 10k scene-disjoint subset, LSAR with a small affordance-refinement loss
(`lambda_aff=0.05`) improves the LGDM baseline repeated-eval mean from 470.0 to
678.0/661.7 across two training seeds. The ablation rows (`lambda_aff=0.0`
and `0.1`) show that the chosen small weight is better than both no auxiliary
loss and the previously used larger `0.1` weight.

Next work is limited to paper/GitHub organization unless the user asks for
additional experiments.

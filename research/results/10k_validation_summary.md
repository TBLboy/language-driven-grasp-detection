# 10k Validation Summary

Date: 2026-09-03 to 2026-09-04

## Dataset

- Stems: `research/smoke-data/train_subset_10k.tsv`
- 10000 stems / 10000 unique scenes
- Split: 8000 train / 2000 validation, seed 42, scene-disjoint by
  construction because each stem selects a unique scene.
- RGB source: Grasp-Anything `<scene>.jpg`; 10000/10000 available.

## Training

All models use `train_lgdm_clean.py` with:

- 15 epochs
- batch size 2
- lr 1e-3
- weight decay 1e-4
- grad clip 1.0
- diffusion eval steps 10
- seed 42

Conditioning variants:

- `none`: official LGDM decoder without language-conditioned residual.
- `lsar_full`: LSAR with `scale=0.01` fixed and
  `lsar_affordance_weight=0.1` (ablation).
- `lsar_no_aff`: LSAR with `scale=0.01` fixed and
  `lsar_affordance_weight=0.0` (ablation).
- `lsar_final`: LSAR with `scale=0.01` fixed and
  `lsar_affordance_weight=0.05` (final proposed configuration).

Final-method second training seed uses the same seed-42 split and
`--seed 43` for model initialization only:
`outputs/lgdm_10k/lsar_final_lambda_0.05_seed43_split42`.

## Repeated Evaluation

Three 10-step diffusion sampling runs per model, seeds 100/101/102.

| Method | Seeds | Mean | Std |
| --- | --- | ---: | ---: |
| LGDM baseline | 473, 479, 458 | 470.0 | 10.82 |
| LSAR-full | 601, 626, 589 | 605.3 | 18.88 |
| LSAR-no-aff (ours) | 647, 660, 654 | 653.7 | 6.51 |
| LSAR-0.05 seed 42 (final ours) | 662, 677, 695 | 678.0 | 16.52 |
| LSAR-0.05 seed 43 (final ours) | 661, 648, 676 | 661.7 | 14.01 |

Single training-loop evaluations were 449/2000 (baseline), 625/2000
(LSAR-full), 643/2000 (LSAR-no-aff), 686/2000 (final seed 42), and
666/2000 (final seed 43).

## Sampling-Step Sensitivity

Fixed 200-sample validation subset, `subsample_seed=7`, sampling seed 200.

| Method | 10 steps | 50 steps |
| --- | ---: | ---: |
| LGDM baseline | 39 | 44 |
| LSAR-full | 55 | 61 |
| LSAR-no-aff | 57 | 65 |
| LSAR-0.05 final (`seed42`) | 65 | 67 |

## Visualizations

- `outputs/lgdm_10k/paired_visuals/paired_qualitative.png`
  (LGDM vs LSAR-full)
- `outputs/lgdm_10k/paired_visuals_no_aff/paired_qualitative.png`
  (LGDM vs final LSAR-no-aff)
- `outputs/lgdm_10k/paired_visuals_final_005/paired_qualitative.png`
  (LGDM vs final LSAR-0.05)
- copy: `research/assets/qualitative_10k_paired_lsar_0.05.png`
- Each panel contains selected part-level prompts, including pen cap,
  apple stem, fork handle, banana flesh, and highlighter cap.

## Conclusion

At the 10k scene-disjoint subset, the final LSAR configuration uses
`lsar_affordance_weight=0.05`. Two training seeds give repeated-eval means
of 678.0 and 661.7, compared with LGDM baseline mean 470.0 and
`lambda_aff=0.0` mean 653.7. The `lambda_aff=0.1` variant gives 605.3, so
the final method uses a small affordance-refinement loss while avoiding the
larger weight used in the earlier LSAR-full ablation.

# Language-driven Grasp Detection

Short research-and-engineering project on **language-driven grasp detection**
using the Grasp-Anything++ dataset. The model takes an RGB image plus a
language grasping instruction and predicts a 2D grasp rectangle
`(x, y, w, h, theta)`.

This repository contains the official LGDM diffusion baseline, a
clean training objective that actually back-propagates the diffusion loss,
and our **Language-conditioned Spatial Affordance Refinement (LSAR)**
module inserted in the language conditioning branch.

## Method

Official LGDM fuses image and text with ALBEF, but its upstream code does not
connect the language feature `y` to the GG-CNN decoder, and it computes the
diffusion loss without calling `backward` on it. We fix the training objective
and add a lightweight spatial refinement:

```text
Image + instruction
        -> ALBEF fusion -> y_view (8x19x19)
        -> LSAR residual (spatial affordance refinement)
        -> GG-CNN decoder -> diffusion grasp generation -> (x, y, w, h, theta)
```

`models/lgdm_lsar.py` provides three reproducible conditioning modes:

- `none`: official LGDM forward, no `y` injection
- `plain-y`: raw ALBEF text feature added at GG-CNN `conv3`
- `lsar`: LSAR-refined residual added at `conv3` (our method)

Only the conditioning branch changes. Grasp representation, diffusion
schedule, dense-map decoding, and evaluation remain identical to the official
LGD pipeline.

## Install

```bash
conda create -n grasp-lgd python=3.10 -y
conda activate grasp-lgd
conda env config vars set PYTHONNOUSERSITE=1 -n grasp-lgd
conda activate grasp-lgd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r env/requirements-project.txt
```

Set `PYTHONNOUSERSITE=1` (or activate the environment after setting the env
var) so `~/.local` packages cannot shadow the conda environment.

## Dataset

The experiments use real Grasp-Anything++ stems aligned with RGB images from
Grasp-Anything:

- instructions: `grasp_instructions/<scene>_<obj>_<part>.pkl`
- positive grasps: `grasp_label_positive/<scene>_<obj>_<part>.pt`
- RGB: `<scene>.jpg` from Grasp-Anything

Prepare a deterministic 1000-sample cross-scene subset:

```bash
PYTHONNOUSERSITE=1 python \
  research/scripts/prepare_training_subset.py \
  --positive-dir /mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_label_positive/grasp_label_positive \
  --instruction-dir /mnt/data/grasp-anything-lgd/data/processed/grasp-anything-pp/grasp_instructions/grasp_instructions \
  --out research/smoke-data/train_subset_1000.tsv \
  --num-stems 1000 --max-per-scene 1 --seed 42
```

Extract only the needed RGB files:

```bash
./research/scripts/extract_rgb_subset.sh \
  --stems research/smoke-data/train_subset_1000.tsv
```

## Train

Clean LGDM baseline:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 python \
  train.py \
  --stems-tsv research/smoke-data/train_subset_1000.tsv \
  --out outputs/lgdm_exp1000/none \
  --epochs 20 --train-ratio 0.8 --batch-size 2 --eval-steps 10 \
  --condition-mode none --seed 42
```

LSAR (final proposed method, fixed residual scale):

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 python \
  train.py \
  --stems-tsv research/smoke-data/train_subset_10k.tsv \
  --out outputs/lgdm_10k/lambda_aff_0.05 \
  --epochs 15 --train-ratio 0.8 --batch-size 2 --eval-steps 10 \
  --condition-mode lsar --lsar-scale 0.01 --lsar-fixed-scale \
  --lsar-affordance-weight 0.05 --seed 42 --split-seed 42
```

For the ablations, set `--lsar-affordance-weight 0.0` (no-aff) or `0.1`
(LSAR-full with explicit affordance supervision). Run the final-method
second training seed with `--seed 43 --split-seed 42`.

## Evaluate and Visualize

Repeat 10-step sampling evaluation on a saved checkpoint:

```bash
PYTHONNOUSERSITE=1 python \
  evaluate.py \
  --checkpoint outputs/lgdm_exp1000/lsar_tuned/last.pt \
  --stems-tsv research/smoke-data/train_subset_1000.tsv \
  --out outputs/lgdm_exp1000/lsar_tuned_repeat_eval \
  --repeats 3 --start-seed 100
```

Render GT (green) and predicted (red) grasp rectangles:

```bash
PYTHONNOUSERSITE=1 python \
  inference.py \
  --checkpoint outputs/lgdm_exp1000/lsar_tuned/last.pt \
  --stems-tsv research/smoke-data/train_subset_1000.tsv \
  --out outputs/lgdm_exp1000/visuals --n-samples 6
```

Render the paired LGDM vs LSAR qualitative figure used in the paper:

```bash
PYTHONNOUSERSITE=1 python \
  research/scripts/visualize_lgdm_paired.py \
  --baseline-checkpoint outputs/lgdm_10k/none/last.pt \
  --lsar-checkpoint outputs/lgdm_10k/lsar_no_aff/last.pt \
  --stems-tsv research/smoke-data/train_subset_10k.tsv \
  --out outputs/lgdm_10k/paired_visuals_no_aff --n-samples 6
```

The current final-method figure can be reproduced with the `lambda_aff=0.05`
checkpoint:

```bash
PYTHONNOUSERSITE=1 python \
  research/scripts/visualize_lgdm_paired.py \
  --baseline-checkpoint outputs/lgdm_10k/none/last.pt \
  --lsar-checkpoint outputs/lgdm_10k/lambda_aff_0.05/last.pt \
  --stems-tsv research/smoke-data/train_subset_10k.tsv \
  --out outputs/lgdm_10k/paired_visuals_final_005 --n-samples 6
```

Run a deterministic sampling-step sensitivity subset:

```bash
PYTHONNOUSERSITE=1 python \
  evaluate.py \
  --checkpoint outputs/lgdm_10k/lsar_no_aff/last.pt \
  --stems-tsv research/smoke-data/train_subset_10k.tsv \
  --out outputs/lgdm_10k/sensitivity/lsar_no_aff_50.json \
  --repeats 1 --eval-steps 50 --max-samples 200 \
  --subsample-seed 7 --start-seed 200
```

Summarize all saved `eval_metrics.json`:

```bash
PYTHONNOUSERSITE=1 python \
  research/scripts/summarize_experiments.py outputs/lgdm_exp1000
```

## Results

The frozen 10k experimental summary used for the paper materials is in
[`research/results/final_method_summary.md`](research/results/final_method_summary.md).

Qualitative paired figure for the final `lambda_aff=0.05` method:
`research/assets/qualitative_10k_paired_lsar_0.05.png`.
The rendered six samples include pen, highlighter, marker cap, duck bill,
apple stem, and keychain prompts.

### 1000-sample diagnosis

800 train / 200 val / 20 epochs / 10-step diffusion sampling.
Not a state-of-the-art claim; these numbers validate that the full
dataset -> train -> checkpoint -> eval -> visualize chain is reproducible.

| Method | single eval correct/200 | mean over 3 sampling seeds |
| --- | ---: | ---: |
| Official LGDM (`none`) | 33 | 37.0 |
| Raw `y` injection (`plain-y`) | 37 | 40.0 |
| LSAR, learnable scale | 13 | - |
| LSAR, fixed scale 0.05 (ours) | 39 | 38.3 |
| LSAR, scale 0.01 + affordance loss (ours) | 43 | 41.7 |
| LSAR, scale 0.01, no affordance loss | 15 | 18.7 |

The learnable LSAR scale grew to `0.224` and degraded the model; fixing the
residual scale to `0.05` restored stability and kept LSAR slightly above the
official baseline. The follow-up scale sweep selected `0.01` as the best fixed
residual scale, and removing the LSAR affordance loss clearly degraded the
conditioning module. See `.project-log/docs/lsar-experimental-validation-plan-20260902.md`
for the full tables and decision. This conclusion did not transfer to the
10k experiment below; the affordance-loss comparison must therefore be read
at the matching training scale.

### 2968-sample large subset

2968 stems / 1010 unique scenes / 2374 train / 594 val / 15 epochs /
10-step diffusion sampling. Same training config for both rows.

| Method | single eval /594 | 3-repeat mean /594 |
| --- | ---: | ---: |
| Official LGDM (`none`) | 151 | 152.0 |
| LSAR-full (ours) | 185 | 179.0 |

The larger subset confirms the direction: LSAR improves the repeated-eval
mean by about 27/594 with lower sampling variance. It is still a 1010-scene
experiment, not a final performance claim.

### 5000-sample validation

5000 stems / 5000 unique scenes / 4000 train / 1000 val / 15 epochs /
10-step diffusion sampling. The 5000 RGB scenes are extracted from the local
Grasp-Anything archive on demand; no full archive expansion is required.

| Method | single eval /1000 | 3-repeat mean /1000 | std |
| --- | ---: | ---: | ---: |
| Official LGDM (`none`) | 211 | 202.7 | 4.93 |
| LSAR-full (ours) | 299 | 309.7 | 3.51 |

With 5000 distinct scenes, LSAR improves the repeated-eval mean by about
107/1000 and also has lower sampling variance. This supports freezing the
current LSAR configuration for the final training/paper stage rather than
redesigning the module.

### 10000-sample final validation

10000 stems / 10000 unique scenes / 8000 train / 2000 val / 15 epochs /
10-step diffusion sampling. Because each stem is a unique scene, the
80/20 split is scene-disjoint by construction.

| Method | single eval /2000 | 3-repeat mean /2000 | std |
| --- | ---: | ---: | ---: |
| Official LGDM (`none`) | 449 | 470.0 | 10.82 |
| LSAR-full (`lambda_aff=0.1`) | 625 | 605.3 | 18.88 |
| LSAR-no-aff (`lambda_aff=0.0`) | 643 | 653.7 | 6.51 |
| LSAR (`lambda_aff=0.05`, seed 42, final ours) | 686 | 678.0 | 16.52 |
| LSAR (`lambda_aff=0.05`, seed 43, final ours) | 666 | 661.7 | 14.01 |

At the 10k scale, the LSAR residual conditioning remains clearly better than
the LGDM baseline. The final method uses `lambda_aff=0.05`, which improves
repeated-eval mean from 470.0 to 678.0/661.7 across two training seeds.
`lambda_aff=0.0` and `lambda_aff=0.1` are retained as ablation rows; the
larger affordance distillation weight degrades performance.

Sampling-step sensitivity on a fixed 200-sample validation subset
(`--subsample-seed 7`, `--start-seed 200`):

| Method | 10 steps | 50 steps |
| --- | ---: | ---: |
| Official LGDM (`none`) | 39 | 44 |
| LSAR-full | 55 | 61 |
| LSAR-no-aff | 57 | 65 |
| LSAR (`lambda_aff=0.05`, final ours) | 65 | 67 |

Increasing diffusion sampling steps improves all settings and preserves the
LSAR ordering. The values are subset diagnostics, not a replacement for the
full 2000-sample repeated evaluation above. The final row uses
`outputs/lgdm_10k/sensitivity/final_0.05_{10,50}.json`.

## Repository Layout

```text
train.py                             top-level training entry point
evaluate.py                          top-level repeated evaluation entry point
inference.py                         top-level sample inference/visualization
models/lgdm_lsar.py                LSAR conditioning variants
research/scripts/train_lgdm_clean.py
research/scripts/eval_lgdm_checkpoint.py
research/scripts/visualize_lgdm_samples.py
research/scripts/visualize_lgdm_paired.py
research/scripts/run_10k_lambda_sweep.sh
research/scripts/summarize_experiments.py
research/scripts/prepare_training_subset.py
research/scripts/extract_rgb_subset.sh
research/assets/qualitative_lsar_tuned.png
LGD-main/                          official LGD baseline snapshot
docs/technical-investigation-report.md
```

## Paper

The final submission is a 2-page CVPR-style English paper.

- Public repository: <https://github.com/TBLboy/language-driven-grasp-detection>
- Paper source: [`research/paper/main.tex`](research/paper/main.tex)
- Compiled two-page PDF: [`research/paper/main.pdf`](research/paper/main.pdf)

Before emailing the final PDF to HR, copy `research/paper/main.pdf` to
`FirstName_LastName.pdf` and replace the placeholder author name in `main.tex`
with the actual name.

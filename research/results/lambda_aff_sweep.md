# lambda_aff Sweep Summary (10k subset)

> Status: complete
> Date: 2026-09-03 to 2026-09-04

Purpose: separate "affordance supervision is generally harmful" from
"lambda_aff=0.1 is a bad weight".

## Fixed training config

- stems: `research/smoke-data/train_subset_10k.tsv`
- split: 8000 train / 2000 val, seed 42
- 15 epochs
- batch size 2
- lr 1e-3
- weight decay 1e-4
- eval steps 10
- condition mode: `lsar`
- LSAR fixed scale: 0.01

## Result table

| lambda_aff | run status | eval result /2000 | repeat mean /2000 | notes |
| ---: | --- | ---: | ---: | --- |
| 0.0 | done | 643 | 653.7 | no-aff ablation |
| 0.01 | done | 662 | not repeated | not selected as winner |
| 0.05 | done | 686 | 678.0 | winner: 662/677/695, std 16.52 |
| 0.1 | done | 625 | 605.3 | LSAR-full ablation |

## Decision rule

- If 0.01 or 0.05 is not clearly better than 0.0, fix 0.0 as final LSAR.
- If a nonzero weight wins, run 3x repeated evaluation for the winner and
  update the final method/README/paper table.

## Latest check

- 0.01 finished: single eval `662/2000`.
- 0.05 seed42: single eval `686/2000`; 3x repeat eval `662/677/695`,
  mean `678.0`, std `16.52`.
- Second training seed `seed=43, split-seed=42`: single eval `666/2000`;
  3x repeat eval `661/648/676`, mean `661.7`, std `14.01`.
- `lambda_aff=0.05` is the final LSAR training configuration. Both training
  seeds clearly exceed `lambda_aff=0.0` (repeat mean `653.7`).

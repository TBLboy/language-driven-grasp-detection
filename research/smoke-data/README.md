# Smoke Data

This directory contains one real Grasp-Anything++ sample extracted from the
Hugging Face releases without downloading the full dataset.

Sample stem:

```text
805944ac6070b2c8f52a2ef228c9b660e116af1221284245dfa4930c8be865a6_0_1
```

Verified files:

```text
image/805944ac...jpg               416x416 RGB scene image
grasp_instructions/<stem>.pkl      "Pick up apple by its flesh."
grasp_label_positive/<stem>.pt     (5, 6) float32 positive grasps
grasp_label_negative/<stem>.pt     (7, 6) float32 negative grasps
```

`part_mask/<stem>.npy` is intentionally absent. The selected stem was not
found in the small local tail index used for the smoke fixture, and the
baseline smoke chain does not consume part masks.

The image was fetched from `airvlab/Grasp-Anything` `image_part_ab` via an
HTTP Range request. The plusplus annotation files were fetched from
`airvlab/Grasp-Anything-pp` in the same way. No full 87 GB download is required
for this fixture.

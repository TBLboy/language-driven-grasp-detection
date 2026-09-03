#!/usr/bin/env python3
"""Top-level inference/visualization entry point for individual samples.

Forwards directly to ``research.scripts.visualize_lgdm_samples``. It loads a
saved checkpoint, runs the same diffusion sampling used in evaluation, and
writes RGB + GT + prediction + optional LSAR affordance overlays.
"""

from research.scripts.visualize_lgdm_samples import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Top-level training entry point for LGDM / LGDM+LSAR.

Forwards directly to ``research.scripts.train_lgdm_clean`` so the clean
diffusion objective and reproducible experiment flags stay in one place.
"""

from research.scripts.train_lgdm_clean import main


if __name__ == "__main__":
    raise SystemExit(main())

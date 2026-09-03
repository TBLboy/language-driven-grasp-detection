#!/usr/bin/env python3
"""Top-level evaluation entry point for saved LGDM / LGDM+LSAR checkpoints.

Forwards directly to ``research.scripts.eval_lgdm_checkpoint``.
"""

from research.scripts.eval_lgdm_checkpoint import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Compatibility wrapper: keep old CLI path working.

Canonical entrypoint lives in `cbm.phase1.eval`.
"""

from cbm.phase1.eval import main


if __name__ == "__main__":
    main()

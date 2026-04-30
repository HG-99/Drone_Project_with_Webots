"""Webots controller entry point for the rectangular line-tracker robot.

Place this file at:
  project/controllers/line_tracker/line_tracker.py

The actual implementation lives in line_nav/runtime.py so that the
controller entry point stays small and easy to replace.
"""

from __future__ import annotations
from line_nav.runtime import main

if __name__ == "__main__":
    main()
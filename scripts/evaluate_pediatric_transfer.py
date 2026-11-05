#!/usr/bin/env python3
"""
Entrypoint wrapper for evaluate_pediatric_transfer.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_pediatric_transfer import run_pediatric_transfer_evaluation

if __name__ == "__main__":
    run_pediatric_transfer_evaluation()


import os

SYN_DATA_DIR = "./data/synthetic_data"
SYN_DISTRACTOR_DIR = "./data/synthetic_data_distractors"
REAL_DATA_DIR = "./data/real_data"

if not os.path.exists(SYN_DATA_DIR):
    print("Warning: synthetic data dir not found")
if not os.path.exists(REAL_DATA_DIR):
    print("Warning: real data dir not found")
if not os.path.exists(SYN_DISTRACTOR_DIR):
    print("Warning: synthetic distractor data dir not found")

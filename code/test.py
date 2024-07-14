import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from test_utils import *

def parse_args():
    # -----------------------------PARSER-----------------------------
    parser = argparse.ArgumentParser(description="Blur Retrieval Testing")
    # -----------------------------DATA-----------------------------
    parser.add_argument(
        "--database_ratio",
        type=float,
        default=10 / 12,
        help="database ratio in test data",
    )

    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of workers for dataloader"
    )

    # -----------------------------MODEL-----------------------------
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./results/train_results/best.pkl",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--pred_cls", 
        type=bool, 
        default=True, 
        help="whether to predict classes"
    )
    parser.add_argument(
        "--pred_blur",
        type=bool,
        default=True,
        help="whether to predict blur levels",
    )
    parser.add_argument(
        "--pred_descriptor",
        type=bool,
        default=True,
        help="whether to predict descriptors",
    )
    parser.add_argument(
        "--pred_loc",
        type=bool,
        default=True,
        help="whether to predict location",
    )

    parser.add_argument(
        "--descriptor_size", type=int, default=128, help="descriptor size"
    )
    parser.add_argument("--num_classes", type=int, default=792, help="number of classes")
    parser.add_argument(
        "--num_blur_levels", type=int, default=6, help="number of blur levels"
    )
    parser.add_argument("--image_height", type=int, default=240, help="image height")
    parser.add_argument("--image_width", type=int, default=320, help="image width")

    parser.add_argument(
        "--encoder_pretrained",
        type=bool,
        default=True,
        help="whether to use pretrained model",
    )
    parser.add_argument(
        "--encoder_norm_type",
        type=str,
        default=None,
        help="encoder normalization type, None, BatchNorm, InstanceNorm, LayerNorm",
    )

    # ------------------------------TESTING-----------------------------
    parser.add_argument(
        "--test_log_dir",
        type=str,
        default="./results/test_results",
        help="path to save test logs",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="test batch size"
    )
    
    # -------------------------------OTHERS-------------------------------
    args = parser.parse_args()

    return args
 
if __name__ == "__main__":
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    args = parse_args()
    
    test_synthetic(args, with_distractors=False, per_blur_level_db=False, device=device)
        
    test_real(args, device)
        
        
        

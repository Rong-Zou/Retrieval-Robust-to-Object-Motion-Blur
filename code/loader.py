import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from PIL import Image
import json
import cv2
import math
from tqdm import tqdm
from models import *
from utils import *

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from set_data_dirs import SYN_DATA_DIR, SYN_DISTRACTOR_DIR, REAL_DATA_DIR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class dataset_database_query_synthetic(Dataset):
    def __init__(
        self,
        instance_paths,
        normalize=True,
        transform=None,
        database_ratio=11 / 12,
        take_blur_levels=[0, 1, 2, 3, 4, 5],
        save_load_imgs_dir=None,
    ):
        super().__init__()

        self.instance_paths = instance_paths
        # for each path in instance_paths, split by 'synthetic_data' and take the second part and join it with SYN_DATA_DIR
        self.instance_paths = [
            os.path.join(SYN_DATA_DIR, path.split("synthetic_data/")[1])
            for path in self.instance_paths
        ]

        self.database_ratio = database_ratio

        # sort blur_levels_blur_val from large to small
        if isinstance(take_blur_levels, int):
            take_blur_levels = [take_blur_levels]
        self.blur_levels_blur_val_large_to_small = sorted(take_blur_levels, reverse=True)
        self.num_blur_levels = len(self.blur_levels_blur_val_large_to_small)

        self.bins = np.round(np.arange(0, 1.1, 0.1), 3).tolist()

        self.instance_names = [
            os.path.basename(instance_path) for instance_path in instance_paths
        ]
        self.instance_names = sorted(list(set(self.instance_names)))
        num_instances = len(self.instance_names)
        # map the unique instance ids to an integer from 0 to num_instances
        # map the unique instance ids to a one-hot label
        self.instance_id_to_label = {
            instance_id: np.eye(num_instances)[i]
            for i, instance_id in enumerate(self.instance_names)
        }

        self.normalize = normalize
        self.transform = transform

        self.save_load_imgs_dir = save_load_imgs_dir
        assure_dir(self.save_load_imgs_dir)

        saved_ins_im_db_mixed_path = os.path.join(
            save_load_imgs_dir, "instance_images_db_mixed.npy"
        )
        saved_ins_im_perBL_db_path = os.path.join(
            save_load_imgs_dir, "instance_images_perBL_db.npy"
        )
        saved_ins_im_q_mixed_path = os.path.join(
            save_load_imgs_dir, "instance_images_q_mixed.npy"
        )
        saved_ins_im_db_only_sharp_path = os.path.join(
            save_load_imgs_dir, "instance_images_db_only_sharp.npy"
        )
        saved_ins_im_q_only_sharp_path = os.path.join(
            save_load_imgs_dir, "instance_images_q_only_sharp.npy"
        )

        # if there are no saved files, then load the images and save them
        if (
            not os.path.exists(saved_ins_im_db_mixed_path)
            or not os.path.exists(saved_ins_im_q_mixed_path)
            or not os.path.exists(saved_ins_im_db_only_sharp_path)
            or not os.path.exists(saved_ins_im_q_only_sharp_path)
            or not os.path.exists(saved_ins_im_perBL_db_path)
        ):
            self.instance_traj_blur_val_stats = np.load(
                os.path.join(
                    SYN_DATA_DIR, "stats/traj_stats", "instance_traj_blur_val_stats.npy"
                ),
                allow_pickle=True,
            ).item()
            self.instance_traj_blur_val_counts = np.load(
                os.path.join(
                    SYN_DATA_DIR, "stats/traj_stats", "instance_traj_blur_val_counts.npy"
                ),
                allow_pickle=True,
            ).item()
            # for k v in instance_traj_blur_val_stats.items(), v is a list, keep only elements whose index is in blur_levels_blur_val
            # instance_traj_blur_val_stats = {k: [v[i] for i in blur_levels_blur_val] for k, v in instance_traj_blur_val_stats.items()}

            blur_values = np.load(
                os.path.join(
                    SYN_DATA_DIR, "stats/blur_stats", "blur_values_by_blurlevel.npz"
                ),
                allow_pickle=True,
            )
            blur_values = [blur_values[i] for i in blur_values]

            with open(
                os.path.join(
                    SYN_DATA_DIR, "stats/blur_stats", "img_paths_by_blurlevel.json"
                ),
                "r",
            ) as f:
                img_paths = json.load(f)

            self.all_blur_vals = blur_values
            self.all_img_paths = img_paths

            self.instance_images_db = {
                instance_name: [] for instance_name in self.instance_names
            }
            self.instance_images_perBL_db = {
                instance_name: [] for instance_name in self.instance_names
            }
            self.instance_images_q = {
                instance_name: [] for instance_name in self.instance_names
            }
            self.instance_images_db_only_sharp = {
                instance_name: [] for instance_name in self.instance_names
            }
            self.instance_images_q_only_sharp = {
                instance_name: [] for instance_name in self.instance_names
            }

            self.all_instance_images_db = []
            self.all_instance_images_perBL_db = []
            self.all_instance_images_q = []
            self.all_instance_images_db_only_sharp = []
            self.all_instance_images_q_only_sharp = []

            # for instance_path in self.instance_paths:
            for instance_path in tqdm(self.instance_paths):
                # when use blur_val to define blur level
                (
                    this_instance_db,
                    this_instance_perBL_db,
                    this_instance_q,
                    this_instance_db_only_sharp,
                    this_instance_q_only_sharp,
                ) = self.get_instance_imgs_db_q(
                    instance_path, type="rendered", 
                )
                instance_name = os.path.basename(instance_path)

                self.instance_images_db[instance_name] = this_instance_db
                self.instance_images_perBL_db[instance_name] = this_instance_perBL_db
                self.instance_images_q[instance_name] = this_instance_q
                self.instance_images_db_only_sharp[instance_name] = (
                    this_instance_db_only_sharp
                )
                self.instance_images_q_only_sharp[instance_name] = (
                    this_instance_q_only_sharp
                )

                self.all_instance_images_db += this_instance_db
                self.all_instance_images_perBL_db += this_instance_perBL_db
                self.all_instance_images_q += this_instance_q
                self.all_instance_images_db_only_sharp += this_instance_db_only_sharp
                self.all_instance_images_q_only_sharp += this_instance_q_only_sharp

            # save all the above lists
            np.save(saved_ins_im_db_mixed_path, self.instance_images_db)
            np.save(saved_ins_im_perBL_db_path, self.instance_images_perBL_db)
            np.save(saved_ins_im_q_mixed_path, self.instance_images_q)
            np.save(saved_ins_im_db_only_sharp_path, self.instance_images_db_only_sharp)
            np.save(saved_ins_im_q_only_sharp_path, self.instance_images_q_only_sharp)

            del self.instance_traj_blur_val_stats
            del self.instance_traj_blur_val_counts
            del self.all_blur_vals
            del self.all_img_paths

        else:  # load the above lists
            self.instance_images_db = np.load(
                saved_ins_im_db_mixed_path, allow_pickle=True
            ).item()
            self.instance_images_perBL_db = np.load(
                saved_ins_im_perBL_db_path, allow_pickle=True
            ).item()
            self.instance_images_q = np.load(
                saved_ins_im_q_mixed_path, allow_pickle=True
            ).item()
            self.instance_images_db_only_sharp = np.load(
                saved_ins_im_db_only_sharp_path, allow_pickle=True
            ).item()
            self.instance_images_q_only_sharp = np.load(
                saved_ins_im_q_only_sharp_path, allow_pickle=True
            ).item()
            self.all_instance_images_db = []
            self.all_instance_images_perBL_db = []
            self.all_instance_images_q = []
            self.all_instance_images_db_only_sharp = []
            self.all_instance_images_q_only_sharp = []
            for instance_name in self.instance_names:
                self.all_instance_images_db += self.instance_images_db[instance_name]
                self.all_instance_images_perBL_db += self.instance_images_perBL_db[
                    instance_name
                ]
                self.all_instance_images_q += self.instance_images_q[instance_name]
                self.all_instance_images_db_only_sharp += (
                    self.instance_images_db_only_sharp[instance_name]
                )
                self.all_instance_images_q_only_sharp += (
                    self.instance_images_q_only_sharp[instance_name]
                )

        self.dataset_type = None
        self.data = None

    def get_instance_imgs_db_q(self, instance_path, type="rendered"):
        instance_images_db = []
        instance_blur_vals_db = []
        instance_blur_levels_db = []

        instance_images_db_only_sharp = []
        instance_blur_vals_db_only_sharp = []
        instance_blur_levels_db_only_sharp = []

        instance_images_perBL_db = []
        instance_blur_vals_perBL_db = []
        instance_blur_levels_perBL_db = []

        instance_images_q = []
        instance_blur_vals_q = []
        instance_blur_levels_q = []

        instance_images_q_only_sharp = []
        instance_blur_vals_q_only_sharp = []
        instance_blur_levels_q_only_sharp = []

        instance_name = os.path.basename(instance_path)
        label = self.instance_id_to_label[instance_name]
        # for database
    
        # get the blur_val stats and counts of this instance
        instance_traj_blur_val_stats_this = self.instance_traj_blur_val_stats[
            instance_name
        ]
        instance_traj_blur_val_counts_this = self.instance_traj_blur_val_counts[
            instance_name
        ]
        # sort the two lists by counts, we also need the indices of the sorted lists
        instance_traj_blur_val_counts_this_sorted, blur_levels_sorted = torch.sort(
            torch.tensor(instance_traj_blur_val_counts_this), descending=False
        )  
        instance_traj_blur_val_stats_this_sorted = [
            instance_traj_blur_val_stats_this[i] for i in blur_levels_sorted
        ]
        # get the idxes of the blur levels in blur_levels_sorted that are in blur_levels_blur_val
        idxes = [
            i
            for i in range(len(blur_levels_sorted))
            if blur_levels_sorted[i] in self.blur_levels_blur_val_large_to_small
        ]
        # keep only the blur levels in blur_levels_blur_val
        instance_traj_blur_val_counts_this_sorted = [
            instance_traj_blur_val_counts_this_sorted[i].item() for i in idxes
        ]
        instance_traj_blur_val_stats_this_sorted = [
            instance_traj_blur_val_stats_this_sorted[i] for i in idxes
        ]
        blur_levels_sorted = [blur_levels_sorted[i].item() for i in idxes]

        take_trajs_for_diff_blur_levels = {
            blur_level: [] for blur_level in self.blur_levels_blur_val_large_to_small
        }
        take_num_trajs_for_diff_blur_levels = {
            blur_level: 0 for blur_level in self.blur_levels_blur_val_large_to_small
        }

        """query"""
        take_trajs_for_diff_blur_levels_q = {
            blur_level: [] for blur_level in self.blur_levels_blur_val_large_to_small
        }
        take_num_trajs_for_diff_blur_levels_q = {
            blur_level: 0 for blur_level in self.blur_levels_blur_val_large_to_small
        }

        # The following code is for database
        num_traj = max(instance_traj_blur_val_counts_this)
        num_traj_each_blur_level = [
            math.floor(num_traj / self.num_blur_levels)
        ] * self.num_blur_levels
        remains = num_traj % self.num_blur_levels
        # ranomly choose remains blur levels and add 1 to num_traj_each_blur_level
        if remains > 0:
            take_blur_levels = random.sample(range(self.num_blur_levels), remains)
            for i in take_blur_levels:
                num_traj_each_blur_level[i] += 1

        for i in range(self.num_blur_levels):
            cur_blur_level = blur_levels_sorted[i]
            trajs_this_blur_level = instance_traj_blur_val_stats_this_sorted[i]
            traj_count_this_blur_level = len(trajs_this_blur_level)

            if traj_count_this_blur_level < num_traj_each_blur_level[i]:
                take_trajs_this_blur_level = trajs_this_blur_level

                if i < self.num_blur_levels - 1:
                    # how less trajs do we take
                    spare_num_traj = (
                        num_traj_each_blur_level[i] - traj_count_this_blur_level
                    )
                    # distribute this num trajs evenly to j>i num_traj_each_blur_level[j]
                    for j in range(i + 1, self.num_blur_levels):
                        num_traj_each_blur_level[j] += spare_num_traj // (
                            self.num_blur_levels - i - 1
                        )
                    spare_num_traj = spare_num_traj % (self.num_blur_levels - i - 1)
                    # randomly choose spare_num_traj blur levels from i+1 to self.num_blur_levels-1 and add 1
                    if spare_num_traj > 0:
                        take_blur_levels = random.sample(
                            range(i + 1, self.num_blur_levels), spare_num_traj
                        )
                        for j in take_blur_levels:
                            num_traj_each_blur_level[j] += 1

            elif traj_count_this_blur_level > num_traj_each_blur_level[i]:
                take_trajs_this_blur_level = random.sample(
                    trajs_this_blur_level, num_traj_each_blur_level[i]
                )

            else:
                take_trajs_this_blur_level = trajs_this_blur_level

            # if non zero
            if len(take_trajs_this_blur_level) > 0:
                # remove the chosen trajs from instance_traj_blur_val_stats_this_sorted[j] for all j > i
                for j in range(i + 1, self.num_blur_levels):
                    instance_traj_blur_val_stats_this_sorted[j] = sorted(
                        list(
                            set(instance_traj_blur_val_stats_this_sorted[j])
                            - set(take_trajs_this_blur_level)
                        )
                    )

                """query"""
                # take some trajs from this blur level for query
                num_q = math.ceil(
                    len(take_trajs_this_blur_level) * (1 - self.database_ratio)
                )
                take_trajs_this_blur_level_q = random.sample(
                    take_trajs_this_blur_level, num_q
                )
                take_trajs_this_blur_level = sorted(
                    list(
                        set(take_trajs_this_blur_level)
                        - set(take_trajs_this_blur_level_q)
                    )
                )
                take_trajs_for_diff_blur_levels_q[cur_blur_level] = (
                    take_trajs_this_blur_level_q
                )
                take_num_trajs_for_diff_blur_levels_q[cur_blur_level] = num_q
                """query"""

                take_trajs_for_diff_blur_levels[cur_blur_level] = (
                    take_trajs_this_blur_level
                )
                take_num_trajs_for_diff_blur_levels[cur_blur_level] = len(
                    take_trajs_this_blur_level
                )

        """database"""
        # get the images, one image from each traj
        for i in range(self.num_blur_levels):
            cur_blur_level = blur_levels_sorted[i]
            take_trajs_this_blur_level = take_trajs_for_diff_blur_levels[
                cur_blur_level
            ]
            take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels[
                cur_blur_level
            ]
            # get the images of this blur level
            if take_num_trajs_this_blur_level > 0:
                # get the images of this blur level
                # get the image paths of all the trajs in take_trajs_this_blur_level
                take_img_paths_this_blur_level = []
                take_img_blur_vals_this_blur_level = []
                for traj in take_trajs_this_blur_level:
                    traj_path = os.path.join(instance_path, traj).split("/")[
                        -3:
                    ]  # ['02691156', '1c93b0eb9c313f5d9a6e43b878d5b335', '000']
                    traj_path = os.path.join(*traj_path)
                    # get the image paths of all the images in this traj
                    im_paths = [
                        os.path.join(traj_path, str(i) + "_blurred.png")
                        for i in range(0, 11)
                    ]

                    paths_ = []
                    blur_vals_ = []
                    for im_path in im_paths:
                        try:
                            im_idx = self.all_img_paths[cur_blur_level].index(
                                im_path
                            )
                        except:
                            continue
                        im_blur_val = self.all_blur_vals[cur_blur_level][im_idx]
                        paths_.append(im_path)
                        blur_vals_.append(im_blur_val)
                    # randomly choose one image
                    idx_ = random.choice(range(len(paths_)))

                    take_img_blur_vals_this_blur_level.append(blur_vals_[idx_])
                    im_idx_in_traj = paths_[idx_].split("/")[-1].split("_")[0]
                    take_img_paths_this_blur_level.append(
                        os.path.join(
                            instance_path,
                            traj,
                            str(im_idx_in_traj) + "_" + type + ".png",
                        )
                    )

                    """database with only sharp"""
                    im_idx = self.all_img_paths[0].index(im_paths[0])
                    instance_images_db_only_sharp.append(
                        os.path.join(
                            instance_path, traj, str(0) + "_" + type + ".png"
                        )
                    )
                    instance_blur_vals_db_only_sharp.append(
                        self.all_blur_vals[0][im_idx]
                    )
                    instance_blur_levels_db_only_sharp.append(0)  # must be 0

                # add the images of this blur level to instance_images
                instance_images_db.extend(take_img_paths_this_blur_level)
                instance_blur_vals_db.extend(take_img_blur_vals_this_blur_level)
                instance_blur_levels_db.extend(
                    [cur_blur_level] * len(take_img_paths_this_blur_level)
                )

        """database version 2 - per blur level database"""
        # get the images, one image for each blur level in each traj, namely num_blur_levels images for each traj
        for i in range(self.num_blur_levels):
            cur_blur_level = blur_levels_sorted[i]
            take_trajs_this_blur_level = take_trajs_for_diff_blur_levels[
                cur_blur_level
            ]
            take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels[
                cur_blur_level
            ]
            # get the images
            if take_num_trajs_this_blur_level > 0:
                # get the images of this blur level
                # get the image paths of all the trajs in take_trajs_this_blur_level
                take_img_paths_these_trajs = []
                take_img_blur_vals_these_trajs = []
                take_img_blur_levels_these_trajs = []
                for traj in take_trajs_this_blur_level:
                    traj_path = os.path.join(instance_path, traj).split("/")[-3:]
                    traj_path = os.path.join(*traj_path)
                    # get the image paths of all the images in this traj
                    im_paths = [
                        os.path.join(traj_path, str(i) + "_blurred.png")
                        for i in range(0, 11)
                    ]

                    im_blur_vals = []
                    im_blur_levels = []
                    for im_path in im_paths:
                        for bl in range(len(self.bins) - 1):
                            try:
                                im_idx = self.all_img_paths[bl].index(im_path)
                            except:
                                continue
                            im_blur_val = self.all_blur_vals[bl][im_idx]
                            im_blur_vals.append(im_blur_val)
                            im_blur_levels.append(bl)
                            break

                    """take only one image for each blur level in blur_levels_blur_val"""
                    for blur_level in self.blur_levels_blur_val_large_to_small:
                        idxes_this_blur_level = [
                            i
                            for i in range(len(im_blur_levels))
                            if im_blur_levels[i] == blur_level
                        ]
                        if len(idxes_this_blur_level) > 0:
                            # randomly choose one image from idxes_this_blur_level
                            idx = random.choice(idxes_this_blur_level)
                            take_img_paths_these_trajs.append(
                                os.path.join(
                                    instance_path,
                                    traj,
                                    str(idx) + "_" + type + ".png",
                                )
                            )
                            take_img_blur_vals_these_trajs.append(im_blur_vals[idx])
                            take_img_blur_levels_these_trajs.append(blur_level)

                # add the images of this blur level to instance_images
                instance_images_perBL_db.extend(take_img_paths_these_trajs)
                instance_blur_vals_perBL_db.extend(take_img_blur_vals_these_trajs)
                instance_blur_levels_perBL_db.extend(take_img_blur_levels_these_trajs)

        """query"""
        # get the images, one image for each blur level in each traj, namely num_blur_levels images for each traj
        for i in range(self.num_blur_levels):
            cur_blur_level = blur_levels_sorted[i]
            take_trajs_this_blur_level = take_trajs_for_diff_blur_levels_q[
                cur_blur_level
            ]
            take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels_q[
                cur_blur_level
            ]
            # get the images
            if take_num_trajs_this_blur_level > 0:
                # get the images of this blur level
                # get the image paths of all the trajs in take_trajs_this_blur_level
                take_img_paths_these_trajs = []
                take_img_blur_vals_these_trajs = []
                take_img_blur_levels_these_trajs = []
                for traj in take_trajs_this_blur_level:
                    traj_path = os.path.join(instance_path, traj).split("/")[-3:]
                    traj_path = os.path.join(*traj_path)
                    # get the image paths of all the images in this traj
                    im_paths = [
                        os.path.join(traj_path, str(i) + "_blurred.png")
                        for i in range(0, 11)
                    ]

                    im_blur_vals = []
                    im_blur_levels = []
                    for im_path in im_paths:
                        for bl in range(len(self.bins) - 1):
                            try:
                                im_idx = self.all_img_paths[bl].index(im_path)
                            except:
                                continue
                            im_blur_val = self.all_blur_vals[bl][im_idx]
                            im_blur_vals.append(im_blur_val)
                            im_blur_levels.append(bl)
                            break

                    """query"""
                    """take only one image for each blur level in blur_levels_blur_val"""

                    for blur_level in self.blur_levels_blur_val_large_to_small:

                        idxes_this_blur_level = [
                            i
                            for i in range(len(im_blur_levels))
                            if im_blur_levels[i] == blur_level
                        ]
                        if len(idxes_this_blur_level) > 0:
                            # randomly choose one image from idxes_this_blur_level
                            idx = random.choice(idxes_this_blur_level)
                            take_img_paths_these_trajs.append(
                                os.path.join(
                                    instance_path,
                                    traj,
                                    str(idx) + "_" + type + ".png",
                                )
                            )
                            take_img_blur_vals_these_trajs.append(im_blur_vals[idx])
                            take_img_blur_levels_these_trajs.append(blur_level)
                    """query only sharp"""
                    instance_images_q_only_sharp.append(
                        os.path.join(
                            instance_path, traj, str(0) + "_" + type + ".png"
                        )
                    )
                    instance_blur_vals_q_only_sharp.append(im_blur_vals[0])
                    instance_blur_levels_q_only_sharp.append(
                        im_blur_levels[0]
                    )  # must be 0

                # add the images of this blur level to instance_images
                instance_images_q.extend(take_img_paths_these_trajs)
                instance_blur_vals_q.extend(take_img_blur_vals_these_trajs)
                instance_blur_levels_q.extend(take_img_blur_levels_these_trajs)

        # zip into a list of tuples (image_path, blur_val, blur_level, label)
        database = list(
            zip(
                instance_images_db,
                instance_blur_vals_db,
                instance_blur_levels_db,
                [label] * len(instance_images_db),
            )
        )
        perBL_database = list(
            zip(
                instance_images_perBL_db,
                instance_blur_vals_perBL_db,
                instance_blur_levels_perBL_db,
                [label] * len(instance_images_perBL_db),
            )
        )
        query = list(
            zip(
                instance_images_q,
                instance_blur_vals_q,
                instance_blur_levels_q,
                [label] * len(instance_images_q),
            )
        )

        database_only_sharp = list(
            zip(
                instance_images_db_only_sharp,
                instance_blur_vals_db_only_sharp,
                instance_blur_levels_db_only_sharp,
                [label] * len(instance_images_db_only_sharp),
            )
        )
        query_only_sharp = list(
            zip(
                instance_images_q_only_sharp,
                instance_blur_vals_q_only_sharp,
                instance_blur_levels_q_only_sharp,
                [label] * len(instance_images_q_only_sharp),
            )
        )

        return database, perBL_database, query, database_only_sharp, query_only_sharp

    def set_dataset_type(self, dataset_type):

        assert len(dataset_type) == 2
        # convert to list
        dataset_type = list(dataset_type)
        # dataset_type[0] must be "database" or "db", or "query" or "q"
        assert dataset_type[0] in [
            "database",
            "db",
            "query",
            "q",
            "perBL_database",
            "perBL_db",
        ]
        # if it is "db" or "q", change it to "database" or "query"
        if dataset_type[0] == "db":
            dataset_type[0] = "database"
        elif dataset_type[0] == "q":
            dataset_type[0] = "query"
        elif dataset_type[0] == "perBL_db":
            dataset_type[0] = "perBL_database"

        # dataset_type[1] must be "sharp" or "s", or "mixed" or "m"
        assert dataset_type[1] in ["sharp", "s", "mixed", "m"]
        # if it is "s" or "m", change it to "sharp" or "mixed"
        if dataset_type[1] == "s":
            dataset_type[1] = "sharp"
        elif dataset_type[1] == "m":
            dataset_type[1] = "mixed"
        # set self.dataset_type
        dataset_type = tuple(dataset_type)
        self.dataset_type = dataset_type
        # set self.data
        if dataset_type == ("database", "sharp"):
            self.data = self.all_instance_images_db_only_sharp
        elif dataset_type == ("database", "mixed"):
            self.data = self.all_instance_images_db
        elif dataset_type == ("perBL_database", "sharp"):
            self.data = self.all_instance_images_db_only_sharp
        elif dataset_type == ("perBL_database", "mixed"):
            self.data = self.all_instance_images_perBL_db
        elif dataset_type == ("query", "sharp"):
            self.data = self.all_instance_images_q_only_sharp
        elif dataset_type == ("query", "mixed"):
            self.data = self.all_instance_images_q

        print("Setting data type to {}.".format(dataset_type))
        print("The data contain {} images.".format(len(self.data)))

    def __getitem__(self, idx):

        img_path, blur_val, blur_level, label = self.data[idx]
        img_path = os.path.join(SYN_DATA_DIR, img_path.split("synthetic_data/")[1])

        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).float().permute(2, 0, 1)  # [3 x H x W]

        # normalize the image
        if self.normalize:
            img = img / 255.0

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path, blur_level

    def __len__(self):
        if self.dataset_type is None:
            raise ValueError(
                "You need to set self.dataset_type before getting the length. Use self.set_dataset_type()."
            )

        return len(self.data)


class dataset_distractors_synthetic(Dataset):
    def __init__(
        self,
        data_dir=None,
        normalize=True,
        transform=None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.normalize = normalize
        self.transform = transform

        if self.data_dir is None:
            self.data_dir = SYN_DISTRACTOR_DIR

        self.stats_dir = os.path.join(SYN_DISTRACTOR_DIR, "stats")

        self.All_blur_vals, self.All_blur_levels, self.All_img_paths = self.get_distractors()

        self.all_blur_vals = None
        self.all_blur_levels = None
        self.all_img_paths = None
        self.dataset_type = None
        
    def get_distractors(self):
        blur_vals = np.load(
            os.path.join(self.stats_dir, "blur_values.npy")
        )
        blur_levels = np.load(os.path.join(self.stats_dir, "blur_levels.npy"))
        with open(os.path.join(self.stats_dir, "img_paths.json"), "r") as f:
            img_paths = json.load(f)

        for i in range(len(img_paths)):
            img_paths[i] = img_paths[i].replace(
                "_blurred.png", "_rendered.png"
            )
        
        return blur_vals, blur_levels, img_paths

    def set_dataset_type(self, dataset_type):
        # set self.dataset_type
        self.dataset_type = dataset_type
        # set self.data
        if dataset_type == "a" or dataset_type == "all":
            idxes = None
        elif dataset_type == "s" or dataset_type == "sharp":
            idxes = [
                i
                for i in range(len(self.All_blur_vals))
                if self.All_img_paths[i].split("/")[-1].split("_")[0] == "0"
            ]
        elif isinstance(dataset_type, tuple) or isinstance(dataset_type, list):
            dataset_type = list(dataset_type)
            idxes = [
                i
                for i in range(len(self.All_blur_vals))
                if self.All_blur_levels[i] in dataset_type
            ]
        elif isinstance(dataset_type, int) or (
            dataset_type.isdigit() and len(dataset_type) == 1
        ):
            dataset_type = int(dataset_type)
            idxes = [
                i
                for i in range(len(self.All_blur_vals))
                if self.All_blur_levels[i] == dataset_type
            ]

        else:
            raise NotImplementedError

        if idxes is None:
            self.all_blur_vals = self.All_blur_vals
            self.all_blur_levels = self.All_blur_levels
            self.all_img_paths = self.All_img_paths
        else:
            self.all_blur_vals = self.All_blur_vals[idxes]
            self.all_blur_levels = self.All_blur_levels[idxes]
            self.all_img_paths = [self.All_img_paths[i] for i in idxes]

    def __getitem__(self, idx):

        blur_level = self.all_blur_levels[idx]
        img_path = os.path.join(SYN_DISTRACTOR_DIR, self.all_img_paths[idx])

        img = np.array(Image.open(img_path))

        img = torch.from_numpy(img).float().permute(2, 0, 1)  # [3 x H x W]

        # normalize the image
        if self.normalize:
            img = img / 255.0

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        return img, img_path, blur_level

    def __len__(self):
        if self.dataset_type is None:
            raise ValueError(
                "You need to set self.dataset_type before getting the length. Use self.set_dataset_type()."
            )

        return len(self.all_blur_vals)


class dataset_database_query_real(Dataset):
    def __init__(
        self,
        data_dir=None,
        normalize=True,
        transform=None,
    ):
        super().__init__()

        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = REAL_DATA_DIR
        self.normalize = normalize
        self.transform = transform
        self.dataset_type = None
        self.data = None

        self.database, self.queries = self.get_db_q_real()

    def get_db_q_real(self):

        saved_db_path = os.path.join(self.data_dir, "stats", "database.npy")
        saved_q_path = os.path.join(self.data_dir, "stats", "queries.npy")
        
        if (
            not os.path.exists(saved_db_path)
            or not os.path.exists(saved_q_path)
        ):
            # read images
            all_images = json.load(open(os.path.join(self.data_dir, "stats", "all_images.json")))
            all_images = [os.path.join(REAL_DATA_DIR, path.split("real_data/")[1]) 
                          for path in all_images]
            # get the unique instance ids
            unique_instance_ids = [img.split("/")[-1].split("_")[0] for img in all_images]
            unique_instance_ids = sorted(list(set(unique_instance_ids)))
            num_instances = len(unique_instance_ids)
            # map the unique instance ids to a one-hot label
            instance_id_to_label = {
                instance_id: np.eye(num_instances)[i]
                for i, instance_id in enumerate(unique_instance_ids)
            }
            label_to_instance_id = {
                tuple(v): k for k, v in instance_id_to_label.items()
            }

            # construct a dict like this: {instance_id: {traj_id: [(img_path, blur_level), ...], ...}, ...}
            instance_imgs = {instance_id: {} for instance_id in unique_instance_ids}
            for img_path in all_images:
                instance_id = img_path.split("/")[-1].split("_")[0]
                traj_id = img_path.split("/")[-1].split("_")[1]
                blur_level = int(img_path.split("/")[-2].split("_")[-1]) - 1
                if traj_id not in instance_imgs[instance_id].keys():
                    instance_imgs[instance_id][traj_id] = []
                instance_imgs[instance_id][traj_id].append((img_path, blur_level))

            # construct database and queries, for each instance, take the trajectory with the least number of images as query and the rest as database
            database = []
            queries = []
            # count for each instance id, how many imgs are there in database and query
            for instance_id in unique_instance_ids:
                # get the label
                label = instance_id_to_label[instance_id]
                # get the trajectories
                trajectories = instance_imgs[instance_id]
                # get the trajectory with the least number of images as query
                query_traj_id = min(trajectories, key=lambda x: len(trajectories[x]))
                # get the query images
                query_imgs = trajectories[query_traj_id]
                # get the database images
                database_imgs = []
                for traj_id in trajectories.keys():
                    if traj_id != query_traj_id:
                        database_imgs.extend(trajectories[traj_id])
                # append the images and labels to database
                database.extend(
                    [(image, blur_level, label) for image, blur_level in database_imgs]
                )
                queries.extend([(image, blur_level, label) for image, blur_level in query_imgs])
            
            # write database and queries into data_dir/stats using npy, first convert label to instance_id
            database_ = [(img, blur_level, label_to_instance_id[tuple(label)]) for img, blur_level, label in database]
            queries_ = [(img, blur_level, label_to_instance_id[tuple(label)]) for img, blur_level, label in queries]
            np.save(saved_db_path, database_)
            np.save(saved_q_path, queries_)
        else: # load from disk
            database_ = np.load(saved_db_path, allow_pickle=True)
            queries_ = np.load(saved_q_path, allow_pickle=True)
            unique_instance_ids = list(set([label for _, _, label in database_]))
            unique_instance_ids = sorted(unique_instance_ids)
            num_instances = len(unique_instance_ids)
            # map the unique instance ids to a one-hot label
            instance_id_to_label = {
                instance_id: np.eye(num_instances)[i]
                for i, instance_id in enumerate(unique_instance_ids)
            }
            database = [(os.path.join(REAL_DATA_DIR, img.split("real_data/")[1]), 
                         int(blur_level), 
                         instance_id_to_label[label]) 
                        for img, blur_level, label in database_
                        ]
            queries = [(os.path.join(REAL_DATA_DIR, img.split("real_data/")[1]), 
                        int(blur_level), 
                        instance_id_to_label[label]) 
                       for img, blur_level, label in queries_
                       ]
        
        return database, queries

    def set_dataset_type(self, dataset_type):

        if isinstance(dataset_type, tuple) or isinstance(dataset_type, list):
            dataset_type = dataset_type[0]

        assert dataset_type in ["database", "db", "query", "q"]
        # if it is "db" or "q", change it to "database" or "query"
        if dataset_type == "db":
            dataset_type = "database"
            # Do not use tuple, o.w. the above line gives error: TypeError: 'tuple' object does not support item assignment
        elif dataset_type == "q":
            dataset_type = "query"

        self.dataset_type = dataset_type
        # set self.data
        if dataset_type == "database":
            self.data = self.database
        elif dataset_type == "query":
            self.data = self.queries

        print("Setting data type to {}.".format(dataset_type))
        print("The data contain {} images.".format(len(self.data)))

    def __getitem__(self, idx):
        blur_level = 0

        if len(self.data[idx]) == 2:
            img_path, label = self.data[idx]
        elif len(self.data[idx]) == 4:
            img_path, blur_value, blur_level, label = self.data[idx]
        elif len(self.data[idx]) == 3:
            img_path, blur_level, label = self.data[idx]
        img = (
            torch.from_numpy(np.array(Image.open(img_path))).float().permute(2, 0, 1)
        )  # [3 x H x W]

        # normalize the image
        if self.normalize:
            img = img / 255.0

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path, blur_level

    def __len__(self):
        if self.dataset_type is None:
            raise ValueError(
                "You need to set self.dataset_type before getting the length. Use self.set_dataset_type()."
            )

        return len(self.data)


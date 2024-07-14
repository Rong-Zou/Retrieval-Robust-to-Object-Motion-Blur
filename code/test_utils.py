import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *
from models import *
from loader import *
from metrics import *
from set_data_dirs import SYN_DATA_DIR

def get_distractor_descriptors(
    model, distractor_loader, device, mode, save_load_path=None
):
    P = save_load_path
    distractor_loader.dataset.set_dataset_type(mode)
    print("Distractor mode: {}".format(mode))
    print("Distractor set contains {} images".format(len(distractor_loader.dataset)))

    # if mode is a list
    if isinstance(mode, list):
        # for example mode is [0,1,2], make it to be '0_1_2'
        mode = "_".join([str(m) for m in mode])

    if os.path.exists(os.path.join(P, f"distractor_descriptors_{mode}.pt")):
        print("Distractor descriptors already exist. Loading descriptors from the disk")
        descriptors = torch.load(
            os.path.join(P, f"distractor_descriptors_{mode}.pt"), map_location=device
        )
        blur_levels = torch.load(
            os.path.join(P, f"distractor_blur_levels_{mode}.pt"), map_location=device
        )

        with open(os.path.join(P, f"distractor_img_paths_{mode}.json"), "r") as f:
            distractor_img_paths = json.load(f)
        img_paths = distractor_img_paths["distractor_img_paths"]

    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        descriptors = []
        blur_levels = []
        img_paths = []
        with torch.no_grad():
            for i, (images, img_path, blur_level) in tqdm(
                enumerate(distractor_loader), total=len(distractor_loader)
            ):
                images = images.to(device)
                descriptors_ = model(images, only_descriptor=True)
                descriptors_ = F.normalize(descriptors_, p=2, dim=1)
                descriptors.append(descriptors_)
                blur_levels.append(blur_level)

                img_paths.extend(img_path)
            descriptors = torch.cat(descriptors, dim=0)
            blur_levels = torch.cat(blur_levels, dim=0)

        torch.save(descriptors, os.path.join(P, f"distractor_descriptors_{mode}.pt"))
        torch.save(blur_levels, os.path.join(P, f"distractor_blur_levels_{mode}.pt"))
        # save the image paths to the disk for future use, save to a json file
        distractor_img_paths_ = {
            "len_distractor": len(img_paths),
            "distractor_img_paths": img_paths,
        }
        with open(os.path.join(P, f"distractor_img_paths_{mode}.json"), "w") as f:
            json.dump(distractor_img_paths_, f, indent=4)

    return descriptors, blur_levels, img_paths


def get_db_q_descriptors(
    model, db_q_loader, device, q_db_mode="mm", save_load_path=None, per_BL_db=False
):

    q_s_m = q_db_mode[0]
    db_s_m = q_db_mode[1]
    assert q_s_m in ["s", "m"]
    assert db_s_m in ["s", "m"]
    P = save_load_path
    if q_s_m == "s":
        q_id = "sharp"
    elif q_s_m == "m":
        q_id = "mixed"
    if db_s_m == "s":
        db_id = "sharp"
    elif db_s_m == "m":
        db_id = "mixed"

    # if paths exists, load from the disk
    if os.path.exists(os.path.join(P, f"queries_descriptors_{q_id}.pt")):
        print("Query descriptors already exist. Loading descriptors from the disk")
        # load from the disk
        queries_descriptors = torch.load(
            os.path.join(P, f"queries_descriptors_{q_id}.pt"), map_location=device
        )
        queries_labels = torch.load(
            os.path.join(P, f"queries_labels_{q_id}.pt"), map_location=device
        )
        queries_blur_levels = torch.load(
            os.path.join(P, f"queries_blur_levels_{q_id}.pt"), map_location=device
        )

        with open(os.path.join(P, f"q_img_paths_{q_id}.json"), "r") as f:
            q_img_paths = json.load(f)
        q_img_paths = q_img_paths["q_img_paths"]
    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        db_q_loader.dataset.set_dataset_type(("q", q_s_m))
        queries_descriptors, queries_labels, queries_blur_levels, q_img_paths = (
            get_descriptor(model, db_q_loader, device)
        )

        torch.save(
            queries_descriptors, os.path.join(P, f"queries_descriptors_{q_id}.pt")
        )

        torch.save(queries_labels, os.path.join(P, f"queries_labels_{q_id}.pt"))
        if queries_blur_levels is not None:
            torch.save(
                queries_blur_levels, os.path.join(P, f"queries_blur_levels_{q_id}.pt")
            )
        # save the image paths to the disk for future use, save to a json file
        q_img_paths_ = {"len_q": len(q_img_paths), "q_img_paths": q_img_paths}
        with open(os.path.join(P, f"q_img_paths_{q_id}.json"), "w") as f:
            json.dump(q_img_paths_, f, indent=4)

    if os.path.exists(os.path.join(P, f"database_descriptors_{db_id}.pt")):
        print("Database descriptors already exist. Loading descriptors from the disk")
        database_descriptors = torch.load(
            os.path.join(P, f"database_descriptors_{db_id}.pt"), map_location=device
        )
        database_labels = torch.load(
            os.path.join(P, f"database_labels_{db_id}.pt"), map_location=device
        )
        database_blur_levels = torch.load(
            os.path.join(P, f"database_blur_levels_{db_id}.pt"), map_location=device
        )

        with open(os.path.join(P, f"db_img_paths_{db_id}.json"), "r") as f:
            db_img_paths = json.load(f)
        db_img_paths = db_img_paths["db_img_paths"]
    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        db_q_loader.dataset.set_dataset_type(("db", db_s_m))
        database_descriptors, database_labels, database_blur_levels, db_img_paths = (
            get_descriptor(model, db_q_loader, device)
        )

        torch.save(
            database_descriptors, os.path.join(P, f"database_descriptors_{db_id}.pt")
        )
        torch.save(database_labels, os.path.join(P, f"database_labels_{db_id}.pt"))
        if database_blur_levels is not None:
            torch.save(
                database_blur_levels,
                os.path.join(P, f"database_blur_levels_{db_id}.pt"),
            )
        db_img_paths_ = {"len_db": len(db_img_paths), "db_img_paths": db_img_paths}
        with open(os.path.join(P, f"db_img_paths_{db_id}.json"), "w") as f:
            json.dump(db_img_paths_, f, indent=4)

    if not per_BL_db:
        return (
            queries_descriptors,
            queries_labels,
            queries_blur_levels,
            database_descriptors,
            database_labels,
            database_blur_levels,
            q_img_paths,
            db_img_paths,
        )

    if os.path.exists(os.path.join(P, f"perBL_database_descriptors_{db_id}.pt")):
        print(
            "Per blur level database descriptors already exist. Loading descriptors from the disk"
        )
        perBL_database_descriptors = torch.load(
            os.path.join(P, f"perBL_database_descriptors_{db_id}.pt"), map_location=device
        )
        perBL_database_labels = torch.load(
            os.path.join(P, f"perBL_database_labels_{db_id}.pt"), map_location=device
        )
        perBL_database_blur_levels = torch.load(
            os.path.join(P, f"perBL_database_blur_levels_{db_id}.pt"), map_location=device
        )

        with open(os.path.join(P, f"perBL_db_img_paths_{db_id}.json"), "r") as f:
            perBL_db_img_paths = json.load(f)
        perBL_db_img_paths = perBL_db_img_paths["perBL_db_img_paths"]
    else:
        assure_dir(P)
        model.to(device)
        model.eval()

        db_q_loader.dataset.set_dataset_type(("perBL_db", db_s_m))
        (
            perBL_database_descriptors,
            perBL_database_labels,
            perBL_database_blur_levels,
            perBL_db_img_paths,
        ) = get_descriptor(model, db_q_loader, device)

        torch.save(
            perBL_database_descriptors,
            os.path.join(P, f"perBL_database_descriptors_{db_id}.pt"),
        )

        torch.save(
            perBL_database_labels, os.path.join(P, f"perBL_database_labels_{db_id}.pt")
        )

        if perBL_database_blur_levels is not None:
            torch.save(
                perBL_database_blur_levels,
                os.path.join(P, f"perBL_database_blur_levels_{db_id}.pt"),
            )
        perBL_db_img_paths_ = {
            "len_perBL_db": len(perBL_db_img_paths),
            "perBL_db_img_paths": perBL_db_img_paths,
        }
        with open(os.path.join(P, f"perBL_db_img_paths_{db_id}.json"), "w") as f:
            json.dump(perBL_db_img_paths_, f, indent=4)

    return (
        queries_descriptors,
        queries_labels,
        queries_blur_levels,
        database_descriptors,
        database_labels,
        database_blur_levels,
        q_img_paths,
        db_img_paths,
        perBL_database_descriptors,
        perBL_database_labels,
        perBL_database_blur_levels,
        perBL_db_img_paths,
    )


def get_descriptor(model, loader, device):
    model.to(device)
    model.eval()
    img_paths = []

    with torch.no_grad():
        for i, (images, labels, img_path, blur_level) in tqdm(
            enumerate(loader), total=len(loader)
        ):
            images = images.to(device)  # [B, 3, H, W]
            descriptors = model(images, only_descriptor=True)  # [B, descriptor_size]
            descriptors = F.normalize(descriptors, p=2, dim=1)  # [B, descriptor_size]

            labels = labels.to(device)  # [B, num_instances_in_testset]
            blur_level = None if blur_level[0] == -1 else blur_level.to(device)  # [B]
            img_paths.extend(img_path)
            if i == 0:
                D = descriptors
                L = labels
                B = blur_level
            else:
                D = torch.cat((D, descriptors), 0)  # [N, descriptor_size]
                L = torch.cat((L, labels), 0)  # [N, num_instances_in_testset]
                if blur_level is not None:
                    B = torch.cat((B, blur_level), 0)  # [N]
    return D, L, B, img_paths


def test_synthetic(args,
                   with_distractors=False,
                   per_blur_level_db=False,
                   device="cuda:0"
                   ):
    model = BlurRetrievalNet(
        args.num_classes if args.pred_cls else None,
        args.num_blur_levels if args.pred_blur else None,
        args.descriptor_size,
        [args.image_height, args.image_width],
        args.pred_loc,
        args.encoder_pretrained,
        args.encoder_norm_type,
    )
    model.to(device)
    
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    args.test_log_dir = args.ckpt_path.replace("train_results", "test_results").split(
        ".pkl"
    )[0]
    assure_dir(args.test_log_dir)
    
    data_split_dict = os.path.join(
        SYN_DATA_DIR, "stats/loader/data_split_info.json"
    )
    print_str = "Loading data split info from {}".format(data_split_dict)
    data_split_dict = json.load(open(data_split_dict, "r"))
    test_instance_folders = data_split_dict["test_instance_folders"]
    print(print_str)

    test_blur_levels = [0, 1, 2, 3, 4, 5]

    result_dir = os.path.join(
        args.test_log_dir, "synthetic_data"
    )

    assure_dir(result_dir)
    test_database_query = dataset_database_query_synthetic(
        test_instance_folders,
        normalize=True,
        database_ratio=args.database_ratio,
        take_blur_levels=test_blur_levels,
        save_load_imgs_dir=os.path.join(
            SYN_DATA_DIR,
            "stats/loader/test",
        ),
    )

    # create the dataloader
    test_database_query_loader = DataLoader(
        test_database_query,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    if with_distractors:
        distractors = dataset_distractors_synthetic(
            normalize=True,
        )
        distractors_loader = DataLoader(
            distractors,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    test_database_query.set_dataset_type(("db", "s"))
    test_dbs_len = len(test_database_query)
    test_database_query.set_dataset_type(("db", "m"))
    test_dbm_len = len(test_database_query)

    test_database_query.set_dataset_type(("q", "s"))
    test_qs_len = len(test_database_query)
    test_database_query.set_dataset_type(("q", "m"))
    test_qm_len = len(test_database_query)
    print_str = "test database and query set loaded, num imgs: \n db_sharp: {}, db_mixed: {}, q_sharp: {}, q_mixed: {}".format(
        test_dbs_len, test_dbm_len, test_qs_len, test_qm_len
    )

    print(print_str)

    for q_db_mode in ["mm"]: # mm: mixed blur level queries, mixed blur level database

        print("Getting descriptors for {}...".format(q_db_mode))
        if per_blur_level_db:
            (
                queries_descriptors,
                queries_labels,
                queries_blur_levels,
                database_descriptors,
                database_labels,
                database_blur_levels,
                q_img_paths,
                db_img_paths,
                perBL_database_descriptors,
                perBL_database_labels,
                perBL_database_blur_levels,
                perBL_db_img_paths,
            ) = get_db_q_descriptors(
                model,
                test_database_query_loader,
                device,
                q_db_mode,
                result_dir,
                per_BL_db=per_blur_level_db,
            )
        else:
            (
                queries_descriptors,
                queries_labels,
                queries_blur_levels,
                database_descriptors,
                database_labels,
                database_blur_levels,
                q_img_paths,
                db_img_paths,
            ) = get_db_q_descriptors(
                model,
                test_database_query_loader,
                device,
                q_db_mode,
                result_dir,
                per_BL_db=per_blur_level_db,
            )

        assure_dir(
            os.path.join(
                result_dir, q_db_mode, "db_blur_level_all"
            )
        )

        print("Get query, database, and perBL_database descriptors done!")

        compute_mAP(
            database_descriptors,
            queries_descriptors,
            database_labels,
            queries_labels,
            database_blur_levels,
            queries_blur_levels,
            top_ks=[None],
            results_path=os.path.join(
                result_dir, q_db_mode, "db_blur_level_all"
            ),
            database_image_paths=db_img_paths,
            consider_only_db_blur_level=None,
        )

        if with_distractors:
            print("Getting distractor descriptors...")
            distractor_mode = q_db_mode[1]
            if distractor_mode == "m":
                distractor_mode = test_blur_levels
            (
                distractor_descriptors,
                distractor_blur_levels,
                distractor_img_paths,
            ) = get_distractor_descriptors(
                model,
                distractors_loader,
                device,
                distractor_mode,
                result_dir,
            )
            assure_dir(
                os.path.join(
                    result_dir,
                    q_db_mode,
                    "db_blur_level_all",
                    "with_distractors",
                )
            )
            print("Get distractor descriptors done!")
            print(
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            # concat database and distractors
            database_descriptors = torch.cat(
                (database_descriptors, distractor_descriptors), dim=0
            )
            database_blur_levels = torch.cat(
                (database_blur_levels, distractor_blur_levels.to(device)),
                dim=0,
            )

            # distractor_labels is all zeros, shape is [num_distractors, len(instance_folders))]
            distractor_labels = torch.zeros(
                (
                    distractor_descriptors.shape[0],
                    len(test_instance_folders),
                )
            ).to(device)
            database_labels = torch.cat(
                (database_labels, distractor_labels), dim=0
            )

            db_img_paths = db_img_paths + distractor_img_paths

            compute_mAP(
                database_descriptors,
                queries_descriptors,
                database_labels,
                queries_labels,
                database_blur_levels,
                queries_blur_levels,
                top_ks=[100],
                results_path=os.path.join(
                    result_dir,
                    q_db_mode,
                    "db_blur_level_all",
                    "with_distractors",
                ),
                database_image_paths=db_img_paths,
                consider_only_db_blur_level=None,
            )

        if per_blur_level_db:

            # compute mAP for each blur level
            for db_blur_level in test_blur_levels:

                assure_dir(
                    os.path.join(
                        result_dir,
                        q_db_mode,
                        "db_blur_level_{}".format(db_blur_level+1),
                    )
                )
                compute_mAP(
                    perBL_database_descriptors,
                    queries_descriptors,
                    perBL_database_labels,
                    queries_labels,
                    perBL_database_blur_levels,
                    queries_blur_levels,
                    top_ks=[None],
                    results_path=os.path.join(
                        result_dir,
                        q_db_mode,
                        "db_blur_level_{}".format(db_blur_level+1),
                    ),
                    database_image_paths=perBL_db_img_paths,
                    consider_only_db_blur_level=db_blur_level,
                )

                if with_distractors:
                    assure_dir(
                        os.path.join(
                            result_dir,
                            q_db_mode,
                            "db_blur_level_{}".format(db_blur_level+1),
                            "with_distractors",
                        )
                    )
                    # get the indexes of distractor data with blur level db_blur_level
                    distractor_indexes = torch.where(
                        distractor_blur_levels == db_blur_level
                    )[0]
                    if len(distractor_indexes) > 0:
                        # get distractor_descriptors, distractor_blur_levels, distractor_img_paths
                        distractor_descriptors_tmp = distractor_descriptors[
                            distractor_indexes
                        ]
                        distractor_blur_levels_tmp = distractor_blur_levels[
                            distractor_indexes
                        ]

                        distractor_img_paths_tmp = [
                            distractor_img_paths[i]
                            for i in distractor_indexes
                        ]
                        # concat database and distractors, per blur level
                        perBL_database_descriptors_tmp = torch.cat(
                            (
                                perBL_database_descriptors,
                                distractor_descriptors_tmp,
                            ),
                            dim=0,
                        )
                        perBL_database_blur_levels_tmp = torch.cat(
                            (
                                perBL_database_blur_levels,
                                distractor_blur_levels_tmp.to(device),
                            ),
                            dim=0,
                        )

                        # distractor_labels is all zeros, shape is [num_distractors, len(instance_folders))]
                        distractor_labels_tmp = torch.zeros(
                            (
                                distractor_descriptors_tmp.shape[0],
                                len(test_instance_folders),
                            )
                        ).to(device)
                        perBL_database_labels_tmp = torch.cat(
                            (perBL_database_labels, distractor_labels_tmp),
                            dim=0,
                        )

                        perBL_db_img_paths_tmp = (
                            perBL_db_img_paths + distractor_img_paths_tmp
                        )

                        compute_mAP(
                            perBL_database_descriptors_tmp,
                            queries_descriptors,
                            perBL_database_labels_tmp,
                            queries_labels,
                            perBL_database_blur_levels_tmp,
                            queries_blur_levels,
                            top_ks=[100],
                            results_path=os.path.join(
                                result_dir,
                                q_db_mode,
                                "db_blur_level_{}".format(db_blur_level+1),
                                "with_distractors",
                            ),
                            database_image_paths=perBL_db_img_paths_tmp,
                            consider_only_db_blur_level=db_blur_level,
                        )


def test_real(args, device):
    model = BlurRetrievalNet(
        args.num_classes if args.pred_cls else None,
        args.num_blur_levels if args.pred_blur else None,
        args.descriptor_size,
        [args.image_height, args.image_width],
        args.pred_loc,
        args.encoder_pretrained,
        args.encoder_norm_type,
    )
    model.to(device)
    
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    args.test_log_dir = args.ckpt_path.replace("train_results", "test_results").split(
        ".pkl"
    )[0]
    assure_dir(args.test_log_dir)
    
    result_dir = os.path.join(args.test_log_dir, "real_data")
    assure_dir(result_dir)
    
    test_database_query = dataset_database_query_real(
            normalize=True,
        )

    test_database_query_loader = DataLoader(
        test_database_query,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_database_query.set_dataset_type("db")
    test_db_len = len(test_database_query)
    test_database_query.set_dataset_type("q")
    test_q_len = len(test_database_query)
    print_str = (
        "test database and query set loaded, num imgs: \n db: {}, q: {}".format(
            test_db_len, test_q_len
        )
    )
    print(print_str)

    print("Getting descriptors...")
    (
        queries_descriptors,
        queries_labels,
        queries_blur_levels,
        database_descriptors,
        database_labels,
        database_blur_levels,
        q_img_paths,
        db_img_paths,
    ) = get_db_q_descriptors(
        model=model,
        db_q_loader=test_database_query_loader,
        device=device,
        save_load_path=result_dir,
    )
    print("Get query, database descriptors done!")
    
    print("Computing mAP...")
    compute_mAP(
        database_descriptors,
        queries_descriptors,
        database_labels,
        queries_labels,
        database_blur_levels,
        queries_blur_levels,
        top_ks=[None],
        results_path=result_dir,
        database_image_paths=db_img_paths,
    )
    print("mAP computation done!")
    
       
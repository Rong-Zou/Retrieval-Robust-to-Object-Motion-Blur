import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *


def compute_mAP(
    database_descriptors,
    queries_descriptors,
    database_labels,
    queries_labels,
    database_blur_levels=None,
    queries_blur_levels=None,
    top_ks=None,
    results_path=None,
    database_image_paths=None,
    consider_only_db_blur_level=None,
):
    """
    database_descriptors: tensor of shape (num_database_imgs, descriptor_dim)
    queries_descriptors: tensor of shape (num_queries_imgs, descriptor_dim)
    database_labels: tensor of shape (num_database_imgs, num_instances)
    queries_labels: tensor of shape (num_queries_imgs, num_instances)
    database_blur_levels: tensor of shape (num_database_imgs), ground truth blur levels of database images
    queries_blur_levels: tensor of shape (num_queries_imgs), ground truth blur levels of query images
    top_ks: int or list
    """
    device = database_descriptors.device
    num_queries = queries_descriptors.shape[0]
    num_database = database_descriptors.shape[0]

    if database_blur_levels is not None and consider_only_db_blur_level is not None:
        if isinstance(consider_only_db_blur_level, int):
            # filter database, keep only blur level db_blur_level
            database_descriptors = database_descriptors[
                database_blur_levels == consider_only_db_blur_level, :
            ]
            database_labels = database_labels[
                database_blur_levels == consider_only_db_blur_level, :
            ]

            num_database = database_descriptors.shape[0]
            # also change the database_image_paths
            database_image_paths = [
                database_image_paths[id]
                for id in torch.nonzero(
                    database_blur_levels == consider_only_db_blur_level, as_tuple=False
                ).squeeze()
            ]

            database_blur_levels = database_blur_levels[
                database_blur_levels == consider_only_db_blur_level
            ]

        elif isinstance(consider_only_db_blur_level, list):
            # filter database, keep only blur levels in db_blur_levels
            database_descriptors = database_descriptors[
                torch.any(
                    database_blur_levels
                    == torch.tensor(consider_only_db_blur_level)
                    .to(device)
                    .unsqueeze(1),
                    axis=0,
                ),
                :,
            ]
            database_labels = database_labels[
                torch.any(
                    database_blur_levels
                    == torch.tensor(consider_only_db_blur_level)
                    .to(device)
                    .unsqueeze(1),
                    axis=0,
                ),
                :,
            ]

            num_database = database_descriptors.shape[0]
            # also change the database_image_paths
            database_image_paths = [
                database_image_paths[id]
                for id in torch.nonzero(
                    torch.any(
                        database_blur_levels
                        == torch.tensor(consider_only_db_blur_level)
                        .to(device)
                        .unsqueeze(1),
                        axis=0,
                    ),
                    as_tuple=False,
                ).squeeze()
            ]
            database_blur_levels = database_blur_levels[
                torch.any(
                    database_blur_levels
                    == torch.tensor(consider_only_db_blur_level)
                    .to(device)
                    .unsqueeze(1),
                    axis=0,
                )
            ]

    nom_db = torch.norm(database_descriptors, p=2, dim=1)
    nom_q = torch.norm(queries_descriptors, p=2, dim=1)
    tol = 1e-5
    if torch.any(
        torch.abs(nom_db - torch.ones(database_descriptors.shape[0]).to(device)) > tol
    ):
        print_str = (
            "!!! WARNING !!! database_descriptors not normalized with tolerance "
            + str(tol)
            + ", max diff: "
            + str(
                torch.max(
                    torch.abs(
                        nom_db - torch.ones(database_descriptors.shape[0]).to(device)
                    )
                )
            )
        )
        print(print_str)
        # database_descriptors = F.normalize(database_descriptors, p=2, dim=1)
    if torch.any(
        torch.abs(nom_q - torch.ones(queries_descriptors.shape[0]).to(device)) > tol
    ):
        print_str = (
            "!!! WARNING !!! queries_descriptors not normalized with tolerance "
            + str(tol)
            + ", max diff: "
            + str(
                torch.max(
                    torch.abs(
                        nom_q - torch.ones(queries_descriptors.shape[0]).to(device)
                    )
                )
            )
        )
        print(print_str)
        # queries_descriptors = F.normalize(queries_descriptors, p=2, dim=1)

    if top_ks is None:
        top_ks = [None]
    elif isinstance(top_ks, int):
        top_ks = [top_ks]

    mAP = None
    mAP_blur_levels = None
    # top_k is a list, for each element in top_k, compute the mAP
    for top_k in top_ks:
        print("top_k: {}".format(top_k))

        if top_k is None:
            print(
                "!!! WARNING !!! top_k is not specified, set it to the number of database images ({}).".format(
                    num_database
                )
            )
            top_k = num_database

        if top_k > num_database:
            print(
                "!!! WARNING !!! top_k ({}) is larger than the number of database images ({}), set top_k to the number of database images".format(
                    top_k, num_database
                )
            )
            top_k = num_database

    
        AP = torch.zeros(num_queries).to(device)

        if database_blur_levels is not None and queries_blur_levels is not None:
            # Do the same thing as above, but for each query image
            min_blur_level = torch.min(
                torch.min(database_blur_levels), torch.min(queries_blur_levels)
            )
            max_blur_level = torch.max(
                torch.max(database_blur_levels), torch.max(queries_blur_levels)
            )
            blur_levels = range(min_blur_level, max_blur_level + 1)
            query_indexes_of_each_blur_level = [[] for _ in range(len(blur_levels))]
            mAP_blur_levels = []

            for q in range(num_queries):
                gt_matching_matrix = torch.matmul(
                    queries_labels[q, :], database_labels.T
                ).float()  # size(num_database)
                similarity_matrix = torch.matmul(
                    queries_descriptors[q, :], database_descriptors.T
                )  # size(num_database)
                indexes_sorted_similarity_matrix = torch.argsort(
                    similarity_matrix, axis=0
                )  # size(num_database)
                # flip the matrix to sort in descending order
                indexes_sorted_similarity_matrix = torch.flip(
                    indexes_sorted_similarity_matrix, [0]
                )
                top_k_indexes_sorted_similarity_matrix = (
                    indexes_sorted_similarity_matrix[:top_k]
                )
                top_k_retrieval_right_wrong_matrix = gt_matching_matrix[
                    top_k_indexes_sorted_similarity_matrix
                ]

                retrieved_matches = torch.sum(top_k_retrieval_right_wrong_matrix)

                if retrieved_matches != 0:
                    score = torch.linspace(
                        1, retrieved_matches, int(retrieved_matches.item())
                    ).to(device)
                    index = (
                        torch.nonzero(
                            top_k_retrieval_right_wrong_matrix == 1, as_tuple=False
                        ).squeeze()
                        + 1.0
                    ).float()
                    precision = score / index
                    AP[q] = torch.mean(precision)

            for i in range(len(blur_levels)):
                query_indexes_of_each_blur_level[i] = torch.nonzero(
                    queries_blur_levels == blur_levels[i], as_tuple=False
                ).squeeze()

                # mAPs for each blur level
                mAP_blur_levels.append(
                    torch.mean(AP[query_indexes_of_each_blur_level[i]])
                )

            mAP = torch.mean(AP)
            # mAP for each blur level
            mAP_blur_levels = torch.tensor(mAP_blur_levels).to(device)

            if results_path is not None:
                save_results(
                    results_path=results_path,
                    mAP=mAP,
                    mAP_blur_level=mAP_blur_levels,
                    top_k=top_k,
                )

        else:
            for q in range(num_queries):
                gt_matching_matrix = torch.matmul(
                    queries_labels[q, :], database_labels.T
                ).float()  # size(num_database)
                similarity_matrix = torch.matmul(
                    queries_descriptors[q, :], database_descriptors.T
                )  # size(num_database)
                indexes_sorted_similarity_matrix = torch.argsort(
                    similarity_matrix, axis=0
                )  # size(num_database)
                # flip the matrix to sort in descending order
                indexes_sorted_similarity_matrix = torch.flip(
                    indexes_sorted_similarity_matrix, [0]
                )
                top_k_indexes_sorted_similarity_matrix = (
                    indexes_sorted_similarity_matrix[:top_k]
                )
                top_k_retrieval_right_wrong_matrix = gt_matching_matrix[
                    top_k_indexes_sorted_similarity_matrix
                ]

                retrieved_matches = torch.sum(top_k_retrieval_right_wrong_matrix)

                if retrieved_matches != 0:
                    score = torch.linspace(
                        1, retrieved_matches, int(retrieved_matches.item())
                    ).to(device)
                    index = (
                        torch.nonzero(
                            top_k_retrieval_right_wrong_matrix == 1, as_tuple=False
                        ).squeeze()
                        + 1.0
                    ).float()
                    precision = score / index
                    AP[q] = torch.mean(precision)

            mAP = torch.mean(AP)
            if results_path is not None:
                save_results(
                    results_path=results_path,
                    mAP=mAP,
                    mAP_blur_level=None,
                    top_k=top_k,
                )

    return mAP, mAP_blur_levels


def save_results(
    results_path,
    mAP=0,
    mAP_blur_level=None,
    top_k=0,
):

    if isinstance(top_k, torch.Tensor):
        top_k = top_k.cpu().numpy()
    if mAP_blur_level is not None and isinstance(mAP_blur_level, torch.Tensor):
        mAP_blur_level = mAP_blur_level.cpu().numpy()
    if isinstance(mAP, torch.Tensor):
        mAP = mAP.cpu().numpy()

    with open(os.path.join(results_path, "results{}.txt".format(top_k)), "w") as f:
        f.write("top_k: " + "\n")
        f.write(str(top_k) + "\n")
        f.write("mAP: " + "\n")
        f.write(str(mAP) + "\n")
        if mAP_blur_level is not None:
            f.write("mAP blur level: " + "\n")
            f.write(str(mAP_blur_level) + "\n")
import os
import json
import typing as tp
from itertools import accumulate

import h5py
from tqdm.auto import tqdm
import numpy as np
from strap.configs.libero_file_functions import get_libero_lang_instruction
from strap.utils.file_utils import get_demo_grp
from strap.utils.processing_utils import flatten_2d_array
from strap.utils.retrieval_utils import (
    RetrievalArgs,
    load_embeddings_into_memory,
    TrajectoryEmbedding,
    segment_trajectory_by_derivative,
    merge_short_segments,
    get_distance_matrix,
    compute_accumulated_cost_matrix_subsequence_dtw_21,
    compute_optimal_warping_path_subsequence_dtw_21,
    TrajectoryMatchResult,
)

from strap.utils.retrieval_utils import load_single_embedding_into_memory


def process_matches(
    args: RetrievalArgs, nested_match_list: tp.List[tp.List[TrajectoryMatchResult]]
):
    k = int(args.top_k / len(nested_match_list))
    if k == 0:
        print(
            "Requested to retrieve less segments than there are than query sub-trajectories. Defaulting"
            "to one match per query."
        )
        k = 1

    for matches in nested_match_list:
        matches.sort()
        matches[:] = matches[:k]
    return nested_match_list


def run_retrieval(
    args: RetrievalArgs,
) -> tp.Tuple[tp.List[TrajectoryMatchResult], tp.List[tp.List[TrajectoryMatchResult]]]:
    """
    Run retrieval to find matching trajectory segments from an offline dataset.

    Args:
        args (RetrievalArgs): Arguments specifying the retrieval configuration, including:
    Returns:
        tuple:
            - List[TrajectoryMatchResult]: Original full trajectories from task dataset
            - List[List[TrajectoryMatchResult]]: Nested list of retrieved trajectory segments,
              where outer list corresponds to subtask task trajectories and inner lists contain
              the top-k matching segments for that trajectory.
    """
    task_embeddings: tp.List[TrajectoryEmbedding] = load_embeddings_into_memory(args)
    original_demo_trajectories = [
        TrajectoryMatchResult(
            start=0,
            end=len(task_embedding),
            cost=0,
            file_path=task_embedding.file_path,
            file_traj_key=task_embedding.file_traj_key,
        )
        for task_embedding in task_embeddings
    ]
    task_embeddings, target_subtrajectories = slice_embeddings(args, task_embeddings)
    nested_match_list = get_all_matches(args, task_embeddings)
    nested_match_list = process_matches(args, nested_match_list)
    return target_subtrajectories, nested_match_list

def slice_embeddings(
    args: RetrievalArgs, task_embeddings: tp.List[TrajectoryEmbedding]
):
    new_task_embeddings = []
    reference_subtrajectories = []
    for task_embedding in task_embeddings:
        # segment using state derivative heuristic
        segments = segment_trajectory_by_derivative(
            task_embedding.eef_poses, threshold=5e-3
        )
        merged_segments = merge_short_segments(
            segments, min_length=args.min_subtraj_len
        )

        # extract slice indexes
        seg_idcs = [0] + list(accumulate(len(seg) for seg in merged_segments))

        for i in range(len(seg_idcs) - 1):
            new_task_embeddings.append(
                TrajectoryEmbedding(
                    embedding=task_embedding.embedding[seg_idcs[i] : seg_idcs[i + 1]],
                    eef_poses=None,  # we do not need them anymore as we sliced already
                    file_path=task_embedding.file_path,
                    file_traj_key=task_embedding.file_traj_key,
                    file_model_key=task_embedding.file_model_key,
                    file_img_keys=task_embedding.file_img_keys,
                )
            )
            reference_subtrajectories.append(
                {
                    "file_path": task_embedding.file_path,
                    "file_traj_key": task_embedding.file_traj_key,
                    "start": seg_idcs[i],
                    "end": seg_idcs[i + 1],
                }
            )
    del task_embeddings
    return new_task_embeddings, reference_subtrajectories

def get_all_matches(
    args: RetrievalArgs, task_embeddings: tp.List[TrajectoryEmbedding]
) -> tp.List[tp.List[TrajectoryMatchResult]]:
    # Need to loop over offline data
    total_matches = 0
    if args.verbose:
        # calculate total number of matches to create
        for i in range(len(args.offline_dataset)):
            embedding_file_path = args.offline_dataset.embedding_paths[i]
            with h5py.File(embedding_file_path, "r", swmr=True) as embedding_file:
                grp = get_demo_grp(
                    embedding_file, args.offline_dataset.file_structure.demo_group
                )
                total_matches += len(grp.keys())
        total_matches *= len(task_embeddings)

    with tqdm(
        total=total_matches, disable=not args.verbose, desc="Finding Matches"
    ) as pbar:
        result_nested_list = [[] for _ in range(len(task_embeddings))]
        for i in range(len(args.offline_dataset)):
            file_path, embedding_path = (
                args.offline_dataset.dataset_paths[i],
                args.offline_dataset.embedding_paths[i],
            )
            with h5py.File(embedding_path, "r", swmr=True) as embedding_file:
                emb_grp = get_demo_grp(
                    embedding_file, args.offline_dataset.file_structure.demo_group
                )
                for traj_key in emb_grp.keys():
                    # load the trajectory into memory
                    off_traj_embd: TrajectoryEmbedding = (
                        load_single_embedding_into_memory(
                            args, emb_grp, traj_key, file_path=file_path
                        )
                    )
                    for j, sub_traj_embedding in enumerate(task_embeddings):
                        single_match = get_single_match(
                            sub_traj_embedding, off_traj_embd
                        )
                        pbar.update(1)
                        if single_match is None:
                            continue
                        result_nested_list[j].append(single_match)
    return result_nested_list


def get_single_match(
    sub_traj_embedding: TrajectoryEmbedding, off_traj_embd: TrajectoryEmbedding
) -> tp.Union[None, TrajectoryMatchResult]:
    if len(sub_traj_embedding) > len(off_traj_embd):
        # There cannot be a valid match
        return None

    distance_matrix = get_distance_matrix(
        sub_traj_embedding.embedding, off_traj_embd.embedding
    )
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(
        distance_matrix
    )
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)
    start = path[0, 1]
    if start < 0:
        assert start == -1
        start = 0
    end = path[-1, 1]
    cost = accumulated_cost_matrix[-1, end]
    # Note that the actual end index is inclusive in this case so +1 to use python : based indexing
    end = end + 1
    return TrajectoryMatchResult(
        start=start,
        end=end,
        cost=cost,
        file_path=off_traj_embd.file_path,
        file_traj_key=off_traj_embd.file_traj_key,
    )


def save_results(
    args: RetrievalArgs,
    full_task_trajectory_results: tp.List[TrajectoryMatchResult],
    nested_match_list: tp.List[tp.List[TrajectoryMatchResult]],
) -> None:

    if os.path.isfile(args.output_path):
        print(f"Output file already exists, overwriting...")
    # make the output location if it doesn't exist
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    with h5py.File(args.output_path, "w") as f:
        args.task_dataset.initalize_save_file_metadata(f, args.task_dataset)

        if args.task_dataset.file_structure.demo_group is not None:
            if args.task_dataset.file_structure.demo_group not in f:
                demo_grp = f.create_group(args.task_dataset.file_structure.demo_group)
            else:
                demo_grp = f[args.task_dataset.file_structure.demo_group]
        else:
            demo_grp = f

        nested_match_list = flatten_2d_array(nested_match_list)

        cur_idx = 0
        retrieved_keys = []
        gt_keys = []
        for match in tqdm(nested_match_list, desc="Saving Matches"):
            demo_key = f"demo_{cur_idx}"
            # save_grp = demo_grp.create_group(demo_key)
            with h5py.File(match.file_path, "r", swmr=True) as data_file:
                args.offline_dataset.save_trajectory_match(
                    data_grp=data_file,
                    out_grp=demo_grp,
                    result=match,
                    args=args,
                    dataset_config=args.task_dataset,
                    new_demo_key=demo_key,
                )
            cur_idx += 1
            retrieved_keys.append(demo_key)
        for full_trajectory_result in full_task_trajectory_results:
            demo_key = f"demo_{cur_idx}"
            gt_keys.append(demo_key)
            cur_idx += 1
            with h5py.File(
                full_trajectory_result.file_path, "r", swmr=True
            ) as data_file:
                args.offline_dataset.save_trajectory_match(
                    data_grp=data_file,
                    out_grp=demo_grp,
                    result=full_trajectory_result,
                    args=args,
                    dataset_config=args.task_dataset,
                    new_demo_key=demo_key,
                )

        # create the masks
        mask_grp = f.create_group("mask")
        mask_grp.create_dataset("demos", data=np.array(retrieved_keys, dtype="S"))
        mask_grp.create_dataset(
            "all", data=np.array(retrieved_keys + gt_keys, dtype="S")
        )
        mask_grp.create_dataset("retrieved", data=np.array(retrieved_keys, dtype="S"))

        print(f"Output file saved at {args.output_path}")


def save_results_update(
    args: RetrievalArgs,
    target_subtrajectories: tp.List[tp.Dict[str, tp.Union[str, int]]],
    nested_match_list: tp.List[tp.List[TrajectoryMatchResult]],
) -> None:

    if os.path.isfile(args.output_path):
        print(f"Output file already exists, overwriting...")
    # make the output location if it doesn't exist
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    with h5py.File(args.output_path, "w") as f:
        demo_grp = f.create_group('results')

        for i, subtrajectory in enumerate(target_subtrajectories):
            ref_demo_key = f"episode_{i}"
            episode_grp = demo_grp.create_group(ref_demo_key)
            
            td_grp = episode_grp.create_group("target_data")
            td_grp.attrs["file_path"] = subtrajectory["file_path"]
            td_grp.attrs["demo_key"] = subtrajectory["file_traj_key"]
            with h5py.File(
                subtrajectory["file_path"], "r", swmr=True
            ) as data_file:
                data_grp = data_file['data'][subtrajectory["file_traj_key"]]
                start = subtrajectory["start"]
                end = subtrajectory["end"]
                td_grp.create_dataset(
                    "actions", data=np.array(data_grp["actions"][start:end])
                )
                td_grp.create_dataset(
                    "robot_states", data=np.array(data_grp["robot_states"][start:end])
                )
                td_grp.create_group("obs")
                for obs_key in data_grp["obs"].keys():
                    td_grp["obs"].create_dataset(
                        obs_key, data=np.array(data_grp["obs"][obs_key][start:end])
                    )
                language_instruction = get_libero_lang_instruction(data_file, subtrajectory["file_traj_key"])
                td_grp.attrs['ep_meta'] = json.dumps({
                    "lang": language_instruction
                })
            for j, match in enumerate(nested_match_list[i]):
                demo_key = f"match_{j}"
                match_grp = episode_grp.create_group(demo_key)
                match_grp.attrs["file_path"] = match.file_path
                match_grp.attrs["demo_key"] = match.file_traj_key
                with h5py.File(
                    match.file_path, "r", swmr=True
                ) as data_file:
                    data_grp = data_file['data'][match.file_traj_key]
                    max_length = data_grp['actions'].shape[0]
                    extra_start = max(
                        0, 0 - match.start + args.frame_stack - 1
                    )  # we want to pad by 4 if the frame stack is 5
                    extra_end = max(0, match.end - max_length + args.action_chunk - 1)

                    start_idx = max(0, match.start - args.frame_stack)
                    end_idx = min(match.end + args.action_chunk, max_length)

                    for lk in ["actions", "robot_states"]:
                        tmp_copy = np.array(data_grp[lk][start_idx:end_idx]).copy()
                        # pad the start if needed
                        if extra_start:
                            tmp_copy = np.concatenate(
                                [np.stack([tmp_copy[0] for i in range(extra_start)], axis=0), tmp_copy],
                                axis=0,
                            )
                        # pad the end if needed
                        if extra_end:
                            tmp_copy = np.concatenate(
                                [tmp_copy, np.stack([tmp_copy[-1] for i in range(extra_end)], axis=0)],
                                axis=0,
                            )

                        match_grp[lk] = tmp_copy

                    match_grp.create_group("obs")
                    for obs_key in data_grp["obs"].keys():                        
                        tmp_copy = np.array(data_grp["obs"][obs_key][start_idx:end_idx]).copy()
                        # pad the start if needed
                        if extra_start:
                            tmp_copy = np.concatenate(
                                [np.stack([tmp_copy[0] for i in range(extra_start)], axis=0), tmp_copy],
                                axis=0,
                            )
                        # pad the end if needed
                        if extra_end:
                            tmp_copy = np.concatenate(
                                [tmp_copy, np.stack([tmp_copy[-1] for i in range(extra_end)], axis=0)],
                                axis=0,
                            )
                        match_grp["obs"][obs_key] = tmp_copy
                    language_instruction = get_libero_lang_instruction(data_file, match.file_traj_key)
                    match_grp.attrs['ep_meta'] = json.dumps({
                        "lang": language_instruction
                    })
            
    print(f"Output file saved at {args.output_path}")


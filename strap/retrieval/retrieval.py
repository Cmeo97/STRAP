import typing as tp
from strap.utils.retrieval_utils import RetrievalArgs
from strap.utils.constants import REPO_ROOT
from copy import deepcopy
from dataclasses import replace
from strap.retrieval.retrieval_helper import run_retrieval, save_results
import numpy as np
import random
import argparse


def get_args(model: str = "dinov2"):

    from strap.configs.libero_hdf5_config import LIBERO_90_CONFIG, LIBERO_10_CONFIG

    # Map model argument to model_key and folder name
    model = model.lower()
    if model == "dinov2":
        model_key = "DINOv2"
        model_folder = "Dinov2"
    elif model == "dinov3":
        model_key = "DINOv3"
        model_folder = "Dinov3"
    elif model == "clip":
        model_key = "CLIP"
        model_folder = "Clip"
    else:
        raise ValueError(f"Unknown model: {model}. Choose from: dinov2, dinov3")

    # Create deep copies of configs and set embedding_subfolder
    task_dataset = deepcopy(LIBERO_10_CONFIG)
    offline_dataset = deepcopy(LIBERO_90_CONFIG)
    
    # Set embedding_subfolder to match the model folder structure
    task_dataset = replace(task_dataset, embedding_subfolder=model_key)
    offline_dataset = replace(offline_dataset, embedding_subfolder=model_key)

    # Update output path to include model folder
    output_path = f"{REPO_ROOT}/data/retrieval_results/{model_folder}/stove-pot_retrieved_dataset.hdf5"

    return RetrievalArgs(
        task_dataset=task_dataset,
        offline_dataset=offline_dataset,
        output_path=output_path,
        model_key=model_key,
        image_keys="obs/agentview_rgb",
        num_demos=5,
        frame_stack=5,
        action_chunk=5,
        top_k=100,
        task_dataset_filter=".*turn_on_the_stove_and_put_the_moka_pot_on_it.*",
        offline_dataset_filter=None,
        min_subtraj_len=20,
    )


def main(args: RetrievalArgs):
    full_task_trajectory_results, retrieval_results = run_retrieval(args)
    save_results(args, full_task_trajectory_results, retrieval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval with specified model")
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2",
        choices=["dinov2", "dinov3"],
        help="Model to use for retrieval (default: dinov2)"
    )
    
    parsed_args = parser.parse_args()
    args = get_args(model=parsed_args.model)

    print(f"Using model: {args.model_key}")
    print(f"Output path: {args.output_path}")

    np.random.seed(args.retrieval_seed)
    random.seed(args.retrieval_seed)
    main(args)

from strap.embedding.encoders import CLIP, DINOv2, DINOv3
from strap.embedding.embedding_helper import embed_dataset
from strap.configs.libero_hdf5_config import LIBERO_90_CONFIG, LIBERO_10_CONFIG

from tqdm.auto import tqdm
from dataclasses import replace
import argparse

"""
Notes:
- If your embedding process crashes, there is a chance that the last embedding file written will be corrupted!
- If your process crashes, the logic to check if a file has already been processed might mistakenly skip some files!
"""

VERBOSE = True


def get_encoders(encoder_name="dinov3"):
    """
    Get encoders based on the specified encoder name.

    Args:
        encoder_name (str): Name of encoder to use. Options: 'dinov2', 'dinov3', 'clip'

    Returns:
        List[Encoder]: List of encoders to use for embedding
    """
    encoder_name = encoder_name.lower()

    if encoder_name == "dinov2":
        models = [
            DINOv2(model_class="facebook/dinov2-base"),
        ]
    elif encoder_name == "dinov3":
        models = [
            DINOv3(model_class="facebook/dinov3-vitb16-pretrain-lvd1689m"),
        ]
    elif encoder_name == "clip":
        models = [
            CLIP(model_class="openai/clip-vit-base-patch16", mm_vision_select_layer=-2),
        ]
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose from: dinov2, dinov3, clip")

    return models


def get_datasets():
    """
    Overwrite this method in order to change which datasets are encoded.
    You can use this with your own custom datasets as well.

    Returns:
        List[DatasetConfig]: List of datasets to embed
    """
    # NOTE: define the datasets you want to embed here
    datasets = [LIBERO_90_CONFIG, LIBERO_10_CONFIG]
    return datasets


def embed_datasets(encoder_name="dinov3", batch_size=256, flip_images=True):
    """
    Embeds all datasets in get_datasets() using the specified encoder.

    Args:
        encoder_name (str): Name of encoder to use (dinov2, dinov3, clip)
        batch_size (int): Batch size for encoding
        flip_images (bool): Whether to flip images (for LIBERO datasets)
    """

    datasets = get_datasets()
    encoders = get_encoders(encoder_name)

    # NOTE: define the settings you want to use here
    # LIBERO's images are upside down, so flip them
    print("\033[94m" + f"Using encoder: {encoder_name.upper()}" + "\033[0m")
    print("\033[94m" + f"Flip imgs is {flip_images}" + "\033[0m")
    print("\033[94m" + f"Batch size: {batch_size}" + "\033[0m")

    image_size = (224, 224)

    # Set the embedding subfolder based on the encoder name
    # This allows saving DINOv2, DINOv3, CLIP embeddings in separate folders
    if len(encoders) > 0:
        encoder_key = encoders[0].embedding_file_key
        # Create new dataset configs with the embedding_subfolder set
        datasets_with_subfolder = [
            replace(dataset, embedding_subfolder=encoder_key)
            for dataset in datasets
        ]
        print("\033[94m" + f"Saving embeddings to subfolder: {encoder_key}" + "\033[0m")
    else:
        datasets_with_subfolder = datasets

    for dataset in tqdm(datasets_with_subfolder, desc=f"Embedding datasets with {encoder_name.upper()}", disable=VERBOSE):
        embed_dataset(
            dataset,
            encoders,
            saver_threads=4,
            flip_images=flip_images,
            batch_size=batch_size,
            image_size=image_size,
            verbose=VERBOSE,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed datasets using specified encoder")
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov3",
        choices=["dinov2", "dinov3", "clip"],
        help="Encoder to use for embedding (default: dinov3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for encoding (default: 256)"
    )
    parser.add_argument(
        "--flip-images",
        action="store_true",
        default=True,
        help="Flip images (default: True for LIBERO)"
    )
    parser.add_argument(
        "--no-flip-images",
        action="store_false",
        dest="flip_images",
        help="Don't flip images"
    )

    args = parser.parse_args()
    embed_datasets(
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        flip_images=args.flip_images
    )

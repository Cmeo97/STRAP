#!/usr/bin/env python3
"""
Compare different DINOv3 model variants by generating patch similarity visualizations.

This script processes one demo with multiple DINOv3 models and saves
the results in organized folders.

Usage:
    python compare_dinov3_models.py --demo-file <path> --demo-idx 0 --timesteps 0 80 160
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from visualize_libero import load_demo_images
from transformers import AutoModel, AutoImageProcessor
from visualize_patch_features import extract_patch_features, compute_patch_similarity


# All DINOv3 models from Hugging Face
DINOV3_MODELS = {
    # ViT models trained on web images (LVD-1689M)
    'vits16': {
        'name': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'params': '21.6M',
        'type': 'ViT-S/16',
    },
    'vits16plus': {
        'name': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
        'params': '28.7M',
        'type': 'ViT-S+/16',
    },
    'vitb16': {
        'name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'params': '85.7M',
        'type': 'ViT-B/16',
    },
    'vitl16': {
        'name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'params': '0.3B',
        'type': 'ViT-L/16',
    },
    'vith16plus': {
        'name': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'params': '0.8B',
        'type': 'ViT-H+/16',
    },
    'vit7b16': {
        'name': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
        'params': '7B',
        'type': 'ViT-7B/16',
    },

    # ConvNeXt models trained on web images (LVD-1689M)
    'convnext_tiny': {
        'name': 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
        'params': '27.8M',
        'type': 'ConvNeXt-Tiny',
    },
    'convnext_small': {
        'name': 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
        'params': '49.5M',
        'type': 'ConvNeXt-Small',
    },
    'convnext_base': {
        'name': 'facebook/dinov3-convnext-base-pretrain-lvd1689m',
        'params': '87.6M',
        'type': 'ConvNeXt-Base',
    },
    'convnext_large': {
        'name': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
        'params': '0.2B',
        'type': 'ConvNeXt-Large',
    },

    # ViT models trained on satellite imagery (SAT-493M)
    'vitl16_sat': {
        'name': 'facebook/dinov3-vitl16-pretrain-sat493m',
        'params': '0.3B',
        'type': 'ViT-L/16 (Satellite)',
    },
    'vit7b16_sat': {
        'name': 'facebook/dinov3-vit7b16-pretrain-sat493m',
        'params': '7B',
        'type': 'ViT-7B/16 (Satellite)',
    },
}


def visualize_model_comparison(images, metadata, model_key, model_info,
                               timesteps, query_patches, device, output_dir):
    """
    Generate patch similarity visualizations for one model.

    Args:
        images: Image array
        metadata: Metadata dict
        model_key: Short model identifier
        model_info: Model information dict
        timesteps: List of timesteps to process
        query_patches: List of query patch coordinates
        device: Device to use
        output_dir: Output directory path
    """
    model_name = model_info['name']
    model_type = model_info['type']
    model_params = model_info['params']

    print(f"\n{'='*80}")
    print(f"Processing: {model_type} ({model_params})")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")

    # Load model
    try:
        print(f"Loading model...")
        model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model.eval()
        model.to(device)
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Create output directory for this model
    model_output_dir = output_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each timestep
    for t in timesteps:
        print(f"  Processing frame {t}...")

        try:
            # Extract patch features
            patch_features, _ = extract_patch_features(images[t], model, processor, device)
            H, W, D = patch_features.shape
            print(f"    Patch grid: {H}×{W}, Embedding dim: {D}")

            # Use default query patches if not provided
            if query_patches is None:
                default_patches = [
                    (H // 2, W // 2),  # Center
                    (2, 2),            # Top-left
                    (2, W - 3),        # Top-right
                    (H - 3, 2),        # Bottom-left
                    (H - 3, W - 3),    # Bottom-right
                ]
            else:
                default_patches = query_patches

            # Create figure
            n_queries = len(default_patches)
            fig, axes = plt.subplots(2, n_queries, figsize=(4 * n_queries, 8))
            if n_queries == 1:
                axes = axes.reshape(2, 1)

            for i, (query_h, query_w) in enumerate(default_patches):
                # Top row: Image with patch highlighted
                ax_img = axes[0, i]
                ax_img.imshow(images[t])

                # Add grid
                img_h, img_w = images[t].shape[:2]
                patch_h = img_h / H
                patch_w = img_w / W

                for j in range(H + 1):
                    ax_img.axhline(y=j * patch_h, color='white', alpha=0.2, linewidth=0.5)
                for j in range(W + 1):
                    ax_img.axvline(x=j * patch_w, color='white', alpha=0.2, linewidth=0.5)

                # Highlight query patch
                query_rect = plt.Rectangle((query_w * patch_w, query_h * patch_h),
                                           patch_w, patch_h,
                                           fill=False, edgecolor='red', linewidth=3)
                ax_img.add_patch(query_rect)
                ax_img.set_title(f'Query Patch ({query_h}, {query_w})', fontsize=10)
                ax_img.axis('off')

                # Bottom row: Similarity map
                ax_sim = axes[1, i]
                similarity_map = compute_patch_similarity(patch_features, query_h, query_w)
                im = ax_sim.imshow(similarity_map, cmap='jet', vmin=-1, vmax=1)
                ax_sim.plot(query_w, query_h, 'r+', markersize=15, markeredgewidth=3)
                ax_sim.set_title(f'Similarity Map', fontsize=10)
                ax_sim.axis('off')
                plt.colorbar(im, ax=ax_sim, label='Cosine Similarity', fraction=0.046)

            # Add title with model info
            title = f'{model_type} ({model_params}) - t={t}\n{metadata.get("instruction", "N/A")}'
            plt.suptitle(title, fontsize=12, fontweight='bold')
            plt.tight_layout()

            # Save figure
            output_file = model_output_dir / f'patch_similarity_t{t}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved: {output_file}")
            plt.close()

        except Exception as e:
            print(f"    ✗ Error processing timestep {t}: {e}")
            continue

    # Clean up model
    del model
    del processor
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n✓ Completed {model_type}\n")
    return True


def create_comparison_summary(output_dir, models_processed, timesteps):
    """
    Create a summary figure comparing all models.

    Args:
        output_dir: Output directory
        models_processed: List of (model_key, model_info) tuples
        timesteps: List of timesteps
    """
    print(f"\n{'='*80}")
    print("Creating comparison summary...")
    print(f"{'='*80}\n")

    # Create a grid showing one timestep from each model
    t = timesteps[len(timesteps) // 2]  # Middle timestep

    n_models = len(models_processed)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_key, model_info) in enumerate(models_processed):
        model_output_dir = output_dir / model_key
        img_file = model_output_dir / f'patch_similarity_t{t}.png'

        if img_file.exists():
            import matplotlib.image as mpimg
            img = mpimg.imread(img_file)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{model_info["type"]} ({model_info["params"]})',
                              fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f'{model_info["type"]}\nNot available',
                         ha='center', va='center', fontsize=12)
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'DINOv3 Model Comparison - Timestep {t}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    summary_file = output_dir / 'model_comparison_summary.png'
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison summary: {summary_file}\n")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare DINOv3 model variants')
    parser.add_argument('--demo-file', type=str, required=True,
                       help='Path to demo HDF5 file')
    parser.add_argument('--demo-idx', type=int, default=0,
                       help='Demo index (default: 0)')
    parser.add_argument('--timesteps', type=int, nargs='+', default=[0, 80, 160],
                       help='Timesteps to visualize (default: 0 80 160)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to compare (default: all). Options: ' +
                            ', '.join(DINOV3_MODELS.keys()))
    parser.add_argument('--output-dir', type=str, default='dinov3_comparison',
                       help='Output directory (default: dinov3_comparison)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--camera', type=str, default='agentview_rgb',
                       help='Camera view (default: agentview_rgb)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("DINOv3 Model Comparison")
    print("=" * 80)
    print(f"Demo file: {args.demo_file}")
    print(f"Demo index: {args.demo_idx}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")

    # Select models to process
    if args.models:
        models_to_process = {k: DINOV3_MODELS[k] for k in args.models if k in DINOV3_MODELS}
    else:
        models_to_process = DINOV3_MODELS

    print(f"\nModels to compare: {len(models_to_process)}")
    for key, info in models_to_process.items():
        print(f"  - {info['type']} ({info['params']})")
    print()

    # Load demo
    print("Loading demo...")
    images, metadata = load_demo_images(args.demo_file, args.demo_idx, args.camera)
    print(f"  ✓ Loaded {len(images)} frames")
    print(f"  ✓ Instruction: {metadata.get('instruction', 'N/A')}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {output_dir}\n")

    # Process each model
    models_processed = []
    for model_key, model_info in models_to_process.items():
        success = visualize_model_comparison(
            images, metadata,
            model_key, model_info,
            args.timesteps,
            query_patches=None,  # Use default
            device=args.device,
            output_dir=output_dir
        )

        if success:
            models_processed.append((model_key, model_info))

    # Create comparison summary
    if len(models_processed) > 1:
        create_comparison_summary(output_dir, models_processed, args.timesteps)

    # Print summary
    print("=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print(f"\nProcessed {len(models_processed)}/{len(models_to_process)} models")
    print(f"\nResults saved in: {output_dir}/")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    for model_key, model_info in models_processed:
        print(f"    ├── {model_key}/")
        for t in args.timesteps:
            print(f"    │   └── patch_similarity_t{t}.png")
    if len(models_processed) > 1:
        print(f"    └── model_comparison_summary.png")

    print("\nTo view results:")
    print(f"  ls {output_dir}/*/")
    print(f"  open {output_dir}/model_comparison_summary.png")


if __name__ == '__main__':
    main()

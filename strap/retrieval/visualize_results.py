#!/usr/bin/env python3
"""
Visualization script for retrieval results.
Explores the structure of retrieved trajectory data and creates visualizations.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
from typing import List, Dict, Any
import os


def explore_file_structure(file_path: str):
    """Explore and print the structure of the HDF5 file."""
    print("=" * 80)
    print(f"Exploring file structure: {file_path}")
    print("=" * 80)
    
    with h5py.File(file_path, "r") as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\n📁 Group: {name}")
                if obj.attrs: print(f"   Attributes: {dict(obj.attrs)}")
            elif isinstance(obj, h5py.Dataset):
                print(f"   📊 Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                if obj.attrs: print(f"      Attributes: {dict(obj.attrs)}")
        
        print("\nFile structure:")
        #f.visititems(print_structure)
        
        # Get demo keys
        if "data" in f:
            demo_group = f["data"]
            demo_keys = list(demo_group.keys())
            print(f"\n📋 Found {len(demo_keys)} episodes: {demo_keys[:10]}{'...' if len(demo_keys) > 10 else ''}")


def get_demo_info(file_path: str, demo_key: str) -> Dict[str, Any]:
    """Extract information about a specific demo."""
    info = {}
    
    with h5py.File(file_path, "r") as f:
        demo_group = f["data"]
        demo_key = f"{demo_key}/match_0"  # Assuming we want info about the first match
        demo = demo_group[demo_key]
        
        info["demo_key"] = demo_key
        
        # Get attributes
        if demo.attrs:
            info["attributes"] = dict(demo.attrs)
            if "ep_meta" in demo.attrs:
                try:
                    info["language_instruction"] = json.loads(demo.attrs["ep_meta"]).get("lang", "N/A")
                except:
                    info["language_instruction"] = "N/A"
        
        # Get data shapes
        info["data_shapes"] = {}
        for key in demo.keys():
            if isinstance(demo[key], h5py.Dataset):
                info["data_shapes"][key] = demo[key].shape
            elif isinstance(demo[key], h5py.Group):
                info["data_shapes"][key] = {}
                for subkey in demo[key].keys():
                    if isinstance(demo[key][subkey], h5py.Dataset):
                        info["data_shapes"][key][subkey] = demo[key][subkey].shape
    
    return info


def visualize_trajectory(
    file_path: str,
    demo_key: str,
    max_images: int = 20,
    save_path: str = None,
    show_images: bool = True,
    flip_images: bool = False
):
    """Visualize a single trajectory with images, actions, and states."""
    
    with h5py.File(file_path, "r") as f:
        demo_group = f["data"]
        demo_key = f"{demo_key}/match_0"  # Visualize the first match
        demo = demo_group[demo_key]
        
        # Get language instruction
        lang_instruction = "N/A"
        if "ep_meta" in demo.attrs:
            try:
                lang_instruction = json.loads(demo.attrs["ep_meta"]).get("lang", "N/A")
            except:
                pass
        
        # Get images
        images = None
        if "obs" in demo and "agentview_rgb" in demo["obs"]:
            images = np.array(demo["obs"]["agentview_rgb"])
            # Handle different image formats
            if len(images.shape) == 4:  # (T, H, W, C)
                if images.shape[-1] == 3:
                    images = images  # Already RGB
                else:
                    images = images.transpose(0, 2, 3, 1)  # Convert from CHW to HWC
        
        # Get actions
        actions = None
        if "actions" in demo:
            actions = np.array(demo["actions"])
        
        # Get states
        states = None
        if "states" in demo:
            states = np.array(demo["states"])
        
        # Get end-effector positions
        ee_pos = None
        if "obs" in demo and "ee_pos" in demo["obs"]:
            ee_pos = np.array(demo["obs"]["ee_pos"])
        
        # Create visualization
        num_images = len(images) if images is not None else 0
        num_images = min(num_images, max_images)
        
        if num_images == 0:
            print(f"⚠️  No images found in {demo_key}")
            return
        
        # Calculate grid size
        cols = 5
        rows = (num_images + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 4 * rows + 6))
        gs = gridspec.GridSpec(rows + 2, cols, figure=fig, hspace=0.3, wspace=0.2)
        
        # Title
        fig.suptitle(
            f"Trajectory: {demo_key}\nLanguage: {lang_instruction}",
            fontsize=14,
            fontweight="bold"
        )
        
        # Plot images
        k = 0
        for i in range(0, images.shape[0], int(images.shape[0] / num_images)):
            row = k // cols
            col = k % cols
            ax = fig.add_subplot(gs[row, col])
            
            if images is not None:
                img = images[i]
                # Ensure image is in correct format
                if img.max() > 1.0:
                    img = img / 255.0
                if flip_images:
                    img = np.flip(img, axis=0)
                ax.imshow(img)
                ax.set_title(f"Frame {i}", fontsize=8)
            ax.axis("off")
            k += 1
            if k >= num_images:
                break
        
        # Plot actions
        if actions is not None:
            ax_actions = fig.add_subplot(gs[rows, :])
            ax_actions.plot(actions)
            ax_actions.set_title("Actions", fontsize=12, fontweight="bold")
            ax_actions.set_xlabel("Time step")
            ax_actions.set_ylabel("Action value")
            ax_actions.grid(True, alpha=0.3)
            if actions.shape[1] <= 7:
                ax_actions.legend([f"Dim {i}" for i in range(actions.shape[1])])
        
        # Plot end-effector positions
        if ee_pos is not None:
            ax_ee = fig.add_subplot(gs[rows + 1, :])
            ax_ee.plot(ee_pos)
            ax_ee.set_title("End-Effector Position", fontsize=12, fontweight="bold")
            ax_ee.set_xlabel("Time step")
            ax_ee.set_ylabel("Position")
            ax_ee.grid(True, alpha=0.3)
            if ee_pos.shape[1] <= 7:
                ax_ee.legend([f"Dim {i}" for i in range(ee_pos.shape[1])])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 Saved visualization to {save_path}")
        
        if show_images:
            plt.show()
        else:
            plt.close()


def create_summary_statistics(file_path: str):
    """Create summary statistics about the retrieval results."""
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
        
    with h5py.File(file_path, "r") as f:
        data = f["data"]
        grp_keys = list(data.keys())

        trajectory_lengths = []
        language_instructions = []
        for grp_key in grp_keys:
            subtrajec = data[grp_key]
            for subkey in subtrajec.keys():
                if subkey == 'target_data':
                    continue
                match = subtrajec[subkey]
                if "robot_states" in match:
                    trajectory_lengths.append(len(match["robot_states"]))
                
                if "ep_meta" in match.attrs:
                    try:
                        lang = json.loads(match.attrs["ep_meta"]).get("lang", "N/A")
                        language_instructions.append(lang)
                    except:
                        pass

        print(f"   Ground truth demos: {len(grp_keys)}")
        print(f"   Retrieved demos: {len(trajectory_lengths)}")

        if trajectory_lengths:
            print(f"\n📏 Retrieved trajectory length statistics:")
            print(f"   Mean: {np.mean(trajectory_lengths):.2f}")
            print(f"   Median: {np.median(trajectory_lengths):.2f}")
            print(f"   Min: {np.min(trajectory_lengths)}")
            print(f"   Max: {np.max(trajectory_lengths)}")
            print(f"   Std: {np.std(trajectory_lengths):.2f}")
        
        if language_instructions:
            unique_langs = set(language_instructions)
            print(f"\n📝 Retrieved language instructions:")
            print(f"   Unique instructions: {len(unique_langs)}")
            for lang in list(unique_langs)[:5]:
                count = language_instructions.count(lang)
                print(f"   - '{lang}': {count} demos")
            if len(unique_langs) > 5:
                print(f"   ... and {len(unique_langs) - 5} more")
        
        # Create visualization
        if trajectory_lengths:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram of trajectory lengths
            axes[0].hist(trajectory_lengths, bins=20, edgecolor="black", alpha=0.7)
            axes[0].axvline(np.mean(trajectory_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(trajectory_lengths):.1f}")
            axes[0].axvline(np.median(trajectory_lengths), color="green", linestyle="--", label=f"Median: {np.median(trajectory_lengths):.1f}")
            axes[0].set_xlabel("Trajectory Length")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Distribution of Trajectory Lengths")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Bar chart: Retrieved vs Ground Truth
            if trajectory_lengths or grp_keys:
                categories = ["Retrieved", "Ground Truth"]
                counts = [len(trajectory_lengths), len(grp_keys)]
                axes[1].bar(categories, counts, color=["skyblue", "lightcoral"], edgecolor="black")
                axes[1].set_ylabel("Number of Demos")
                axes[1].set_title("Retrieved vs Ground Truth Demos")
                axes[1].grid(True, alpha=0.3, axis="y")
                for i, count in enumerate(counts):
                    axes[1].text(i, count + 0.5, str(count), ha="center", va="bottom", fontweight="bold")
            
            plt.tight_layout()
            save_path = str(Path(file_path).parent / "retrieval_statistics.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n💾 Saved statistics plot to {save_path}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize retrieval results from HDF5 file"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/retrieval_results/Dinov3/stove-pot_retrieved_dataset.hdf5",
        help="Path to the retrieval results HDF5 file"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore and print file structure"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Specific demo key to visualize (e.g., 'demo_0')"
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=3,
        help="Maximum number of demos to visualize (default: 5)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images per demo to visualize (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: same as input file)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=True,
        help="Don't display plots (only save them)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only generate statistics, skip individual visualizations"
    )
    parser.add_argument(
        "--flip-images",
        action="store_true",
        default=True,
        help="Flip images vertically for correct orientation (True for Libero dataset)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File not found: {args.file_path}")
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.file_path).parent)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Explore file structure
    if args.explore or args.demo is None:
        explore_file_structure(args.file_path)
    
    # Get demo keys
    with h5py.File(args.file_path, "r") as f:
        demo_group = f["data"]
        demo_keys = list(demo_group.keys())
    
    # Generate statistics
    create_summary_statistics(args.file_path)
    
    if args.stats_only:
        return
    
    # Visualize specific demo or multiple demos
    if args.demo:
        if args.demo in demo_keys:
            print(f"\n🎨 Visualizing demo: {args.demo}")
            save_path = os.path.join(args.output_dir, f"{args.demo}_visualization.png")
            visualize_trajectory(
                args.file_path,
                args.demo,
                max_images=args.max_images,
                save_path=save_path,
                show_images=not args.no_show
            )
        else:
            print(f"❌ Error: Demo '{args.demo}' not found. Available demos: {demo_keys[:10]}")
    else:
        # Visualize multiple demos
        num_demos = min(args.max_demos, len(demo_keys))
        print(f"\n🎨 Visualizing {num_demos} demos...")
        
        for i, demo_key in enumerate(demo_keys[:num_demos]):
            print(f"\n[{i+1}/{num_demos}] Processing {demo_key}...")
            save_path = os.path.join(args.output_dir, f"{demo_key}_visualization.png")
            
            # Get demo info
            info = get_demo_info(args.file_path, demo_key)
            print(f"   Language: {info.get('language_instruction', 'N/A')}")
            print(f"   Data shapes: {info.get('data_shapes', {})}")
            
            visualize_trajectory(
                args.file_path,
                demo_key,
                max_images=args.max_images,
                save_path=save_path,
                show_images=not args.no_show,
                flip_images=args.flip_images
            )
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()


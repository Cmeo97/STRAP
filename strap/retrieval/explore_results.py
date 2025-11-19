#!/usr/bin/env python3
"""
Quick example script showing how to use the visualization tool.
This script demonstrates iterating over retrieval results and exploring the data format.
"""

import h5py
import numpy as np
import json
from pathlib import Path


def explore_retrieval_results(file_path: str):
    """
    Explore retrieval results file and print information about data formats.
    """
    print("=" * 80)
    print("Exploring Retrieval Results")
    print("=" * 80)
    
    with h5py.File(file_path, "r") as f:
        # Check top-level structure
        print("\n📁 Top-level groups:")
        for key in f.keys():
            print(f"   - {key}: {type(f[key])}")
        
        # Explore data group
        if "data" in f:
            demo_group = f["data"]
            demo_keys = list(demo_group.keys())
            print(f"\n📋 Found {len(demo_keys)} demos")
            
            # Explore first demo in detail
            if demo_keys:
                first_demo_key = demo_keys[0]
                print(f"\n🔍 Detailed structure of '{first_demo_key}':")
                demo = demo_group[first_demo_key]
                
                # Print attributes
                if demo.attrs:
                    print("\n   Attributes:")
                    for attr_name, attr_value in demo.attrs.items():
                        if isinstance(attr_value, bytes):
                            try:
                                attr_value = json.loads(attr_value.decode('utf-8'))
                                print(f"      {attr_name}: {attr_value}")
                            except:
                                print(f"      {attr_name}: {attr_value}")
                        else:
                            print(f"      {attr_name}: {attr_value}")
                
                # Print data structure
                print("\n   Data structure:")
                for key in demo.keys():
                    item = demo[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"      📊 {key}: shape={item.shape}, dtype={item.dtype}")
                        if item.shape[0] < 10:  # Print small arrays
                            print(f"         Sample: {np.array(item)[:3]}")
                    elif isinstance(item, h5py.Group):
                        print(f"      📁 {key}/")
                        for subkey in item.keys():
                            subitem = item[subkey]
                            if isinstance(subitem, h5py.Dataset):
                                print(f"         📊 {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
        
        # Explore mask group
        if "mask" in f:
            mask_group = f["mask"]
            print("\n🎭 Mask groups:")
            for key in mask_group.keys():
                mask_data = np.array(mask_group[key])
                if mask_data.dtype == 'S':
                    mask_data = [s.decode('utf-8') for s in mask_data]
                print(f"   {key}: {len(mask_data)} items")
                print(f"      First 5: {mask_data[:5]}")


def print_demo_summary(file_path: str, demo_key: str):
    """Print a summary of a specific demo."""
    with h5py.File(file_path, "r") as f:
        demo = f["data"][demo_key]
        
        print(f"\n📊 Summary for {demo_key}:")
        
        # Language instruction
        if "ep_meta" in demo.attrs:
            try:
                lang = json.loads(demo.attrs["ep_meta"]).get("lang", "N/A")
                print(f"   Language: {lang}")
            except:
                pass
        
        # Data dimensions
        if "actions" in demo:
            actions = np.array(demo["actions"])
            print(f"   Actions: shape={actions.shape}, range=[{actions.min():.3f}, {actions.max():.3f}]")
        
        if "states" in demo:
            states = np.array(demo["states"])
            print(f"   States: shape={states.shape}, range=[{states.min():.3f}, {states.max():.3f}]")
        
        if "obs" in demo:
            print(f"   Observations:")
            for obs_key in demo["obs"].keys():
                obs_data = np.array(demo["obs"][obs_key])
                if len(obs_data.shape) == 4:  # Images
                    print(f"      {obs_key}: shape={obs_data.shape} (images)")
                else:
                    print(f"      {obs_key}: shape={obs_data.shape}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python explore_results.py <path_to_retrieval_results.hdf5>")
        print("\nExample:")
        print("  python explore_results.py data/retrieval_results/Dinov2/stove-pot_retrieved_dataset.hdf5")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    explore_retrieval_results(file_path)
    
    # Print summary for first few demos
    with h5py.File(file_path, "r") as f:
        demo_keys = list(f["data"].keys())[:5]
        for demo_key in demo_keys:
            print_demo_summary(file_path, demo_key)


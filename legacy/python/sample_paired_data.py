#!/usr/bin/env python3
"""
Sample real paired data for benchmarking.
Creates a 10% sample of the Pacific 2014-2015 dataset.
"""

import numpy as np
from pathlib import Path
import random
import sys

def sample_paired_data(data_root: str, sample_fraction: float = 0.1, seed: int = 42):
    """Sample files from paired data directory."""
    
    data_root = Path(data_root)
    moments_dir = data_root / 'moments'
    scalars_dir = data_root / 'scalars'
    
    if not moments_dir.exists() or not scalars_dir.exists():
        raise FileNotFoundError(f"Expected moments/ and scalars/ subdirs in {data_root}")
    
    moment_files = sorted(list(moments_dir.glob('*.npz')))
    scalar_files = sorted(list(scalars_dir.glob('*.npz')))
    
    if not moment_files or not scalar_files:
        raise FileNotFoundError(f"No .npz files found in {moments_dir} or {scalars_dir}")
    
    print(f"Total files: {len(moment_files)} moments, {len(scalar_files)} scalars")
    
    # Sample files
    random.seed(seed)
    num_sample = max(1, int(len(moment_files) * sample_fraction))
    sample_moment_files = random.sample(sorted(moment_files), num_sample)
    
    print(f"Sampling {sample_fraction*100:.0f}%: {num_sample} files")
    
    # Load and combine sampled data
    all_moments = []
    all_targets = []
    
    for i, moment_file in enumerate(sample_moment_files):
        scalar_file = scalars_dir / moment_file.name
        if not scalar_file.exists():
            continue
        
        with np.load(moment_file, allow_pickle=True) as m_data:
            with np.load(scalar_file, allow_pickle=True) as s_data:
                # Extract moment features (skip wavelengths)
                moment_features = []
                for key in m_data.files:
                    if key not in ("__header__", "__version__", "__allow_pickle__", "SWwavlngs", "LWwavlngs"):
                        moment_features.append(m_data[key])
                
                # Extract scalar features (skip Fsw and Flw)
                scalar_features = []
                target = None
                for key in s_data.files:
                    if key not in ("__header__", "__version__", "__allow_pickle__"):
                        arr = s_data[key]
                        if key == "Fsw":
                            target = arr
                        elif key != "Flw":
                            if arr.ndim == 1:
                                arr = arr.reshape(-1, 1)
                            scalar_features.append(arr)
                
                if target is not None and moment_features and scalar_features:
                    combined = np.concatenate(moment_features + scalar_features, axis=-1)
                    all_moments.append(combined)
                    all_targets.append(target)
        
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{num_sample} files...")
    
    if not all_moments:
        raise ValueError("No valid moment/scalar pairs found")
    
    X = np.concatenate(all_moments, axis=0).astype(np.float32)
    y = np.concatenate(all_targets, axis=0).astype(np.float32)
    
    # Clean data
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\n✓ Sampled data: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Memory: ~{(X.nbytes + y.nbytes) / (1024**2):.1f} MB")
    
    return X, y


if __name__ == "__main__":
    import json
    import os
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/paired/Pacific_2014-2015"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "benchmark_outputs/sampled_data.npz"
    
    # Create output dir
    os.makedirs(Path(output_file).parent, exist_ok=True)
    
    # Sample and save
    X, y = sample_paired_data(data_root, sample_fraction=0.1)
    
    np.savez(output_file, X=X, y=y)
    print(f"✓ Saved to {output_file}")

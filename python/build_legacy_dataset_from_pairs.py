#!/usr/bin/env python3
"""Build legacy NPZ+PKL training bundle from paired moments/scalars files.

Generates files expected by:
- data/legacy_code/new_codes/train_Allsky_moments_narrow_values.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


WAVELENGTHS = ["0.47", "0.66", "0.86", "1.375", "2.13", "3.96", "4.05", "4.5", "6.7", "7.3", "9.7", "11.0", "12.0"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build legacy PKL/NPZ bundle from paired moments/scalars data")
    parser.add_argument(
        "--data-root",
        default=str(repo_root / "data" / "paired" / "Pacific_2014-2015"),
        help="Root that contains moments/ and scalars/ folders",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(repo_root / "projects" / "F1km" / "ALLSKY_newDat" / "mrchsepjunedec_2014-2015_Pacific"),
        help="Output file prefix (without extension)",
    )
    return parser.parse_args()


def _nearest_channel_indices(all_wavelengths: np.ndarray) -> List[int]:
    target = np.asarray([float(x) for x in WAVELENGTHS], dtype=np.float32)
    return [int(np.argmin(np.abs(all_wavelengths - wl))) for wl in target]


def _extract_feature_arrays(moments_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mean_spec = np.concatenate((moments_data["mSWspec"], moments_data["mLWspec"]), axis=1)
    std_spec = np.concatenate((moments_data["stdSWspec"], moments_data["stdLWspec"]), axis=1)
    skew_spec = np.concatenate((moments_data["skewSWspec"], moments_data["skewLWspec"]), axis=1)
    kurt_spec = np.concatenate((moments_data["kurtSWspec"], moments_data["kurtLWspec"]), axis=1)
    return {
        "m": mean_spec,
        "s": std_spec,
        "sk": skew_spec,
        "k": kurt_spec,
    }


def build_dataframe(data_root: Path) -> pd.DataFrame:
    moments_dir = data_root / "moments"
    scalars_dir = data_root / "scalars"

    if not moments_dir.is_dir() or not scalars_dir.is_dir():
        raise FileNotFoundError(f"Expected moments/ and scalars/ under {data_root}")

    rows: List[Dict[str, float]] = []
    used_pairs = 0

    for moment_file in sorted(moments_dir.glob("*.npz")):
        scalar_file = scalars_dir / moment_file.name
        if not scalar_file.exists():
            continue

        with np.load(moment_file, allow_pickle=True) as moments_data, np.load(scalar_file, allow_pickle=True) as scalars_data:
            required_m = ("SWwavlngs", "LWwavlngs", "mSWspec", "mLWspec", "stdSWspec", "stdLWspec", "skewSWspec", "skewLWspec", "kurtSWspec", "kurtLWspec")
            required_s = ("SZA", "VZA", "RAA", "meanCTH", "stdCTH", "Fsw")
            if not all(k in moments_data.files for k in required_m):
                continue
            if not all(k in scalars_data.files for k in required_s):
                continue

            all_wavelengths = np.concatenate((moments_data["SWwavlngs"], moments_data["LWwavlngs"]), axis=0).astype(np.float32)
            channel_indices = _nearest_channel_indices(all_wavelengths)
            feature_specs = _extract_feature_arrays(moments_data)

            sza = np.asarray(scalars_data["SZA"], dtype=np.float32)
            vza = np.asarray(scalars_data["VZA"], dtype=np.float32)
            raa = np.asarray(scalars_data["RAA"], dtype=np.float32)
            mcth = np.asarray(scalars_data["meanCTH"], dtype=np.float32)
            scth = np.asarray(scalars_data["stdCTH"], dtype=np.float32)
            flux = np.asarray(scalars_data["Fsw"], dtype=np.float32)

            sample_count = min(
                len(sza),
                len(vza),
                len(raa),
                len(mcth),
                len(scth),
                len(flux),
                *(arr.shape[0] for arr in feature_specs.values()),
            )
            if sample_count < 1:
                continue

            for i in range(sample_count):
                row: Dict[str, float] = {
                    "SZA": float(sza[i]),
                    "VZA": float(vza[i]),
                    "RAA": float(raa[i]),
                    "mCTH": float(mcth[i]),
                    "stdCTH": float(scth[i]),
                    "Flux": float(flux[i]),
                }
                for wl_name, ch in zip(WAVELENGTHS, channel_indices):
                    row[f"m{wl_name}"] = float(feature_specs["m"][i, ch])
                    row[f"s{wl_name}"] = float(feature_specs["s"][i, ch])
                    row[f"sk{wl_name}"] = float(feature_specs["sk"][i, ch])
                    row[f"k{wl_name}"] = float(feature_specs["k"][i, ch])
                rows.append(row)

            used_pairs += 1

    if not rows:
        raise ValueError(f"No valid paired moments/scalars rows were found under {data_root}")

    print(f"Used {used_pairs} paired NPZ files")
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(data_root)

    before_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    after_rows = len(df)
    print(f"Dropped {before_rows - after_rows} non-finite rows; remaining {after_rows}")

    feature_cols = [c for c in df.columns if c not in ("Flux",)]
    df_mean = df[feature_cols].replace([np.inf, -np.inf], np.nan).mean().fillna(0.0).to_numpy(dtype=np.float32)
    df_std = df[feature_cols].replace([np.inf, -np.inf], np.nan).std().fillna(0.0).to_numpy(dtype=np.float32)

    legacy_column_names = [c for c in df.columns if c != "Flux"]

    np.savez(
        str(output_prefix) + ".npz",
        column_names=np.array(legacy_column_names, dtype=str),
        label_normalization_factor=np.array([1.0], dtype=np.float32),
        SWnorm=np.array([1.0], dtype=np.float32),
        MIXnorm=np.array([1.0], dtype=np.float32),
        LWnorm=np.array([1.0], dtype=np.float32),
        Norms=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        df_mean=df_mean,
        df_std=df_std,
    )

    df.to_pickle(str(output_prefix) + ".pkl")
    print(f"Saved {len(df)} rows to {output_prefix}.npz and {output_prefix}.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

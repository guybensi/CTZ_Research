import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_data_root = script_dir / "Data_toGuy" / "test"

    parser = argparse.ArgumentParser(description="Inspect and visualize NPZ benchmark files.")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--file-type", choices=["images", "scalars", "moments"], default="images")
    parser.add_argument("--file-name", default="A2014060.0245.npz")
    parser.add_argument("--image-index", type=int, default=2)
    parser.add_argument(
        "--sw-channels",
        default="1,10",
        help="Comma-separated SW channel indices to plot (for images mode).",
    )
    parser.add_argument("--lw-channel", type=int, default=11)
    parser.add_argument(
        "--all-wavelengths",
        action="store_true",
        help="Plot all SW and all LW channels for the selected image index.",
    )
    parser.add_argument("--list-files", action="store_true", help="List available NPZ files for the selected type.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="If provided, save plot to this path instead of opening an interactive window.",
    )
    return parser.parse_args()


def parse_channel_list(value):
    channels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not channels:
        raise ValueError("--sw-channels must include at least one integer index.")
    return channels


def maybe_wavelength_label(data, key, channel_idx):
    if key not in data.files:
        return ""

    wl = data[key]
    if wl.ndim != 1:
        return ""
    if channel_idx < 0 or channel_idx >= wl.shape[0]:
        return ""

    return f" ({wl[channel_idx]:.4g} microns)"


def show_or_save_plot(save_path):
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def main():
    args = parse_args()
    data_dir = args.data_root / args.file_type

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.list_files:
        files = sorted(p.name for p in data_dir.glob("*.npz"))
        print(f"Available files in {data_dir}:")
        for f in files:
            print(f" - {f}")
        return

    npz_path = data_dir / args.file_name
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)

    print(f"File: {npz_path}\n")
    print(f"{'Variable':<25} {'Shape':<20} {'Dtype':<12} {'Size (MB)':<12}")
    print("-" * 70)

    for var_name in data.files:
        arr = data[var_name]
        size_mb = arr.nbytes / 1e6
        shape_str = str(arr.shape)
        print(f"{var_name:<25} {shape_str:<20} {str(arr.dtype):<12} {size_mb:>10.2f}")

    print("-" * 70)
    total_size = sum(data[var].nbytes for var in data.files)
    print(f"Total size: {total_size / 1e6:.2f} MB")

    if args.file_type == "scalars":
        fsw = data["Fsw"]
        isw = data["Isw"]
        cvar = data["glintf"]

        plt.figure()
        plt.scatter(isw, fsw, c=cvar)
        plt.colorbar()

        nsamp = np.sum(~np.isnan(fsw))
        print(f"Number of samples: {nsamp:.0f}")
        show_or_save_plot(args.save_path)

    if args.file_type == "images":
        sw = data["ImSW"]
        lw = data["ImLW"]
        if args.all_wavelengths:
            sw_channels = list(range(sw.shape[1]))
            lw_channels = list(range(lw.shape[1]))
        else:
            sw_channels = parse_channel_list(args.sw_channels)
            lw_channels = [args.lw_channel]
        im_ind = args.image_index

        if im_ind < 0 or im_ind >= sw.shape[0]:
            raise IndexError(f"image-index {im_ind} is out of range [0, {sw.shape[0] - 1}]")

        n_rows = len(sw_channels) + len(lw_channels)
        plt.figure(figsize=(8, 3.2 * n_rows))

        subplot_idx = 1
        for ch in sw_channels:
            if ch < 0 or ch >= sw.shape[1]:
                raise IndexError(f"SW channel {ch} is out of range [0, {sw.shape[1] - 1}]")
            plt.subplot(n_rows, 1, subplot_idx)
            plt.imshow(np.squeeze(sw[im_ind, ch, :, :]))
            plt.colorbar()
            label = maybe_wavelength_label(data, "SWwavlngs", ch)
            plt.title(f"SW channel {ch}{label}")
            subplot_idx += 1

        for ch in lw_channels:
            if ch < 0 or ch >= lw.shape[1]:
                raise IndexError(f"LW channel {ch} is out of range [0, {lw.shape[1] - 1}]")
            plt.subplot(n_rows, 1, subplot_idx)
            plt.imshow(np.squeeze(lw[im_ind, ch, :, :]))
            plt.colorbar()
            lw_label = maybe_wavelength_label(data, "LWwavlngs", ch)
            plt.title(f"LW channel {ch}{lw_label}")
            subplot_idx += 1
        show_or_save_plot(args.save_path)

    if args.file_type == "moments":
        print("solar wavelengths:", data["SWwavlngs"])
        print("thermal wavelengths:", data["LWwavlngs"])


if __name__ == "__main__":
    main()

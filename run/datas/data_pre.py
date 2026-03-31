import h5py
import torch
import numpy as np
from scipy.io import savemat, loadmat
import os
import torch.nn.functional as F

try:
    from torchvision import transforms
except Exception:
    transforms = None


def _resolve_path(file_path: str) -> str:
    """Resolve dataset path robustly across different working directories.
    Tries:
    - absolute path if provided
    - current working directory
    - project root (two levels up from this file)
    - path variant with 'run/datas' replaced by 'datas'
    Returns the first existing path; raises FileNotFoundError with candidates otherwise.
    """
    candidates = []
    if os.path.isabs(file_path):
        candidates.append(file_path)
    else:
        candidates.append(file_path)
        candidates.append(os.path.join(os.getcwd(), file_path))
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        candidates.append(os.path.join(project_root, file_path))
        # Try mapping 'run/datas' -> 'datas'
        if file_path.startswith("run/datas/"):
            mapped = file_path.replace("run/datas/", "datas/", 1)
            candidates.append(mapped)
            candidates.append(os.path.join(os.getcwd(), mapped))
            candidates.append(os.path.join(project_root, mapped))
        # Also try mapping to run1/datas for run1 workflows
        if file_path.startswith("run/datas/"):
            mapped_run1 = file_path.replace("run/datas/", "run1/datas/", 1)
            candidates.append(mapped_run1)
            candidates.append(os.path.join(os.getcwd(), mapped_run1))
            candidates.append(os.path.join(project_root, mapped_run1))
        # Try direct basename in common data folders under project root
        base = os.path.basename(file_path)
        candidates.append(os.path.join(project_root, "datas", base))
        candidates.append(os.path.join(project_root, "run", "datas", base))
        candidates.append(os.path.join(project_root, "run1", "datas", base))
    for p in candidates:
        if os.path.exists(p):
            return p
    # Last resort: search by filename under project root
    for root, dirs, files in os.walk(project_root):
        if base in files:
            return os.path.join(root, base)
    raise FileNotFoundError(f"Dataset file not found. Tried: {candidates}")


def load_DE_data1(hc_path, mdd_path, normalize=True):
    """
    Load and process DE feature data for HC and MDD, and generate labels.
    """

    # Load MAT files and check data structure
    def load_mat_data(file_path):
        resolved = _resolve_path(file_path)
        with h5py.File(resolved, "r") as f:
            data_key = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)][0]
            data = f[data_key][:]
            return data

    # Load data
    HC_data = load_mat_data(hc_path)
    MDD_data = load_mat_data(mdd_path)

    # Rearrange dimensions (6,300,128,29)->(29,6,128,300)
    HC_data = np.transpose(HC_data, (3, 0, 2, 1))
    MDD_data = np.transpose(MDD_data, (3, 0, 2, 1))

    # Create labels (MDD=0, HC=1)
    HC_labels = np.ones(HC_data.shape[0], dtype=np.int64)
    MDD_labels = np.zeros(MDD_data.shape[0], dtype=np.int64)

    # Merge data and labels
    all_data = np.concatenate((MDD_data, HC_data), axis=0)
    all_labels = np.concatenate((MDD_labels, HC_labels), axis=0)

    # Convert to PyTorch tensors
    data = torch.from_numpy(all_data).float()
    labels = torch.from_numpy(all_labels).long()

    # # Split data and flatten into chunks
    split_data = torch.split(
        data, 100, dim=-1
    )  # Get 3 chunks, each with shape: (53, 6, 128, 100)
    data = torch.cat(split_data, dim=0)  # shape: (159, 6, 128, 100)
    labels = labels.repeat_interleave(3)  # shape: (159,)

    data = data[:, :-1, :, :]

    if normalize:
        data = global_normalize(data)
        return data, labels
    return data, labels


def load_DE_data2(file_path, normalize=True):
    resolved_path = _resolve_path(file_path)
    with h5py.File(resolved_path, "r") as f:
        # Get DE_features group and labels dataset
        de_group = f["DE_features"]
        labels = f["labels"][:].squeeze()

        # Define all frequency bands to be loaded
        bands = ["alpha", "beta", "delta", "gamma", "theta"]
        band_data = []
        for band in bands:
            data = de_group[band][:]  # Get band data
            # Transpose to [samples, features, time] format
            data = data.transpose(
                2, 1, 0
            )  # Assuming original shape is (features, time, samples)
            band_data.append(data)

        # Stack all band data along the channel dimension (channels=5)
        all_data = np.stack(
            band_data, axis=1
        )  # Final shape [samples, 5, features, time]

        # Convert to PyTorch tensors
        data_tensor = torch.tensor(all_data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

    print(f"Data shape: {data_tensor.shape}")
    print(f"Label shape: {labels_tensor.shape}")

    if normalize:
        data_tensor = global_normalize(data_tensor)
        return data_tensor, labels_tensor
    return data_tensor, labels_tensor


def save_sample_data(data, labels, file_path="run/datas/sample_data.mat"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Create dictionary to save
    save_dict = {"data": data, "labels": labels}
    # Save to .mat file
    savemat(file_path, save_dict)
    print(f"Data saved to {file_path}")


def load_sample_data(file_path="run/datas/sample_data.mat", num=0):
    # Load .mat file
    loaded_data = loadmat(file_path)
    # Extract data and labels
    data = loaded_data["data"]
    labels = loaded_data["labels"]
    print(f"Data loaded from {file_path}")
    # Convert to tensors
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    # Squeeze possible singleton dimensions in labels (e.g. (1,N) or (N,1))
    labels = labels.squeeze()

    # Shuffle data and labels together to ensure random sampling
    total_samples = data.shape[0]
    idx = torch.randperm(total_samples)
    data = data[idx]
    labels = labels[idx]

    # If num>0, slice both data and labels so they stay aligned
    if isinstance(num, int) and num > 0:
        data = data[:num]
        labels = labels[:num]

    # Ensure correct dtypes
    data = data.float()
    labels = labels.long()

    return data, labels


def merging_data(data, labels, data_sample, labels_sample):
    data_sample = data_sample.to(data.dtype)
    labels_sample = labels_sample.to(labels.dtype)

    data = torch.cat((data, data_sample), dim=0)
    labels = torch.cat((labels, labels_sample), dim=0)

    return data, labels


def get_min_max(dataset="MODMA", device="cpu"):
    """
    Get the global min and max values of the dataset (returns PyTorch Tensor scalars)

    Args:
        dataset (str): dataset name ('MODMA' or others)
        device (str): device for returned Tensor ('cpu' or 'cuda')

    Returns:
        min_val (torch.Tensor), max_val (torch.Tensor): Tensor scalars
    """
    # 1. Load data
    try:
        if dataset == "MODMA":
            data, _ = load_DE_data1("datas/all_HC_DE.mat", "datas/all_MDD_DE.mat")
        else:
            data, _ = load_DE_data2("datas/all_DE.mat")
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")

    # 2. Ensure data is Tensor and move to specified device
    if not torch.is_tensor(data):
        data = torch.from_numpy(np.asarray(data)).float()
    data = data.to(device)

    # 3. Compute global min and max (return Tensor scalars)
    min_val = torch.min(data)
    max_val = torch.max(data)

    return min_val, max_val


def denormalize_global(normalized_data, original_min, original_max):
    """
    Denormalize data from [-1,1] back to the original range
    normalized_data: normalized data (range[-1,1])
    original_min/max: original min/max value (NumPy or Python scalar)
    """
    return (normalized_data + 1) / 2 * (original_max - original_min) + original_min


def global_normalize(data_tensor):
    """
    Global normalization to [-1, 1] range

    Args:
        data_tensor: input data (PyTorch Tensor)

    Returns:
        normalized data (Tensor)
    """
    # Ensure input is Tensor and float type
    if not torch.is_tensor(data_tensor):
        data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
    else:
        data_tensor = data_tensor.float()
    # Compute global min and max
    min_val = torch.min(data_tensor)
    max_val = torch.max(data_tensor)
    # Linear normalization to [-1, 1]
    normalized_data = (data_tensor - min_val) / (max_val - min_val) * 2 - 1
    return normalized_data


def gaussian_noise_augment(
    data: torch.Tensor,
    label: torch.Tensor,
    expand_factor: int = 2,
    noise_std: float = 0.1,
) -> tuple:
    """
    Add Gaussian noise and augment samples.
    Args:
        data: input EEG data (num, 5, channel, times)
        label: labels (num,)
        expand_factor: augmentation factor (e.g., 2 means doubling the samples)
        noise_std: noise standard deviation
    Returns:
        augmented_data: augmented data (num * expand_factor, 5, channel, times)
        augmented_label: augmented labels (num * expand_factor,)
    """
    data.shape[0]
    augmented_data = [data]
    augmented_label = [label]

    for _ in range(expand_factor - 1):
        noise = torch.randn_like(data) * noise_std
        augmented_data.append(data + noise)
        augmented_label.append(label)

    augmented_data = torch.cat(augmented_data, dim=0)
    augmented_label = torch.cat(augmented_label, dim=0)
    return augmented_data, augmented_label


def time_masking_augment(
    data: torch.Tensor,
    label: torch.Tensor,
    expand_factor: int = 2,
    mask_ratio: float = 0.1,
    fill_value: float = 0.0,
) -> tuple:
    """
    Randomly mask time segments and augment samples.
    Args:
        mask_ratio: ratio of time steps to mask (e.g., 0.1 means mask 10% of time steps)
        fill_value: value to fill in masked parts (default 0)
    """
    num, _, channel, times = data.shape
    augmented_data = [data]
    augmented_label = [label]

    for _ in range(expand_factor - 1):
        mask_length = int(times * mask_ratio)
        mask_start = torch.randint(0, times - mask_length + 1, (num,))

        masked_data = data.clone()
        for i in range(num):
            masked_data[i, :, :, mask_start[i] : mask_start[i] + mask_length] = (
                fill_value
            )

        augmented_data.append(masked_data)
        augmented_label.append(label)

    augmented_data = torch.cat(augmented_data, dim=0)
    augmented_label = torch.cat(augmented_label, dim=0)
    return augmented_data, augmented_label


def phase_shuffling_augment(
    data: torch.Tensor,
    label: torch.Tensor,
    expand_factor: int = 2,
    shuffle_ratio: float = 0.1,
) -> tuple:
    """
    Shuffle local phase and augment samples.
    Args:
        shuffle_ratio: ratio of frequency bands to shuffle (e.g., 0.1 means shuffle 10% of bands)
    """
    num, _, channel, times = data.shape
    augmented_data = [data]
    augmented_label = [label]

    for _ in range(expand_factor - 1):
        shuffled_data = []
        for i in range(num):
            # Fourier transform (process each 5D signal independently)
            signal = data[i]  # (5, channel, times)
            fft = torch.fft.rfft(signal, dim=-1)  # (5, channel, freq_bins)
            magnitude = torch.abs(fft)
            phase = torch.angle(fft)

            # Randomly shuffle part of the phase
            shuffle_bins = int(phase.shape[-1] * shuffle_ratio)
            shuffle_start = torch.randint(0, phase.shape[-1] - shuffle_bins, (1,))
            shuffled_phase = phase.clone()
            idx = torch.randperm(shuffle_bins)
            shuffled_phase[..., shuffle_start : shuffle_start + shuffle_bins] = phase[
                ..., shuffle_start : shuffle_start + shuffle_bins
            ][..., idx]

            # Inverse transform
            reconstructed = torch.fft.irfft(
                magnitude * torch.exp(1j * shuffled_phase), n=times
            )  # (5, channel, times)
            shuffled_data.append(reconstructed)

        shuffled_data = torch.stack(shuffled_data, dim=0)  # (num, 5, channel, times)
        augmented_data.append(shuffled_data)
        augmented_label.append(label)

    augmented_data = torch.cat(augmented_data, dim=0)
    augmented_label = torch.cat(augmented_label, dim=0)
    return augmented_data, augmented_label


def random_crop_augment(
    data: torch.Tensor,
    label: torch.Tensor,
    expand_factor: int = 2,
    crop_ratio: float = 0.8,
    interpolation_mode: str = "nearest",
    padding_mode: str = "constant",
    fill: float = 0.0,
) -> tuple:
    """
    Augmentation with shape-preserving random cropping and interpolation.
    Args:
        data: input EEG data (num, 5, channel, times)
        label: labels (num,)
        expand_factor: augmentation factor (original data + expand_factor times augmented data)
        crop_ratio: crop length ratio (0.0~1.0)
        interpolation_mode: interpolation mode ('linear' or 'nearest')
        padding_mode: padding mode ('constant', 'edge', 'reflect')
        fill: fill value when padding_mode='constant'
    """
    num, _, channel, original_length = data.shape
    crop_length = int(original_length * crop_ratio)

    # Initialize RandomCrop (operate on time dimension) if torchvision is available
    crop = None
    if transforms is not None:
        crop = transforms.RandomCrop(
            size=(channel, crop_length),
            padding=0,
            pad_if_needed=True,
            padding_mode=padding_mode,
            fill=fill,
        )

    augmented_data = [data]
    augmented_label = [label]

    for _ in range(expand_factor - 1):
        cropped_data = []
        for i in range(num):
            # Process each sample's 5D signal independently (keep time alignment)
            signals = []
            for j in range(5):
                signal = data[i, j]  # (channel, original_length)
                if crop is not None:
                    signal_cropped = crop(signal.unsqueeze(0)).squeeze(
                        0
                    )  # (channel, crop_length)
                else:
                    # Fallback without torchvision: random temporal slice
                    max_start = max(1, original_length - crop_length + 1)
                    start = torch.randint(0, max_start, (1,)).item()
                    end = start + crop_length
                    signal_cropped = signal[:, start:end]
                    if signal_cropped.shape[-1] < crop_length:
                        pad_len = crop_length - signal_cropped.shape[-1]
                        signal_cropped = F.pad(
                            signal_cropped, (0, pad_len), mode="constant", value=fill
                        )
                # Resize back to original length
                signal_resized = F.interpolate(
                    signal_cropped.unsqueeze(0).unsqueeze(
                        0
                    ),  # (1, 1, channel, crop_length)
                    size=(channel, original_length),
                    mode=interpolation_mode,
                    align_corners=False if interpolation_mode == "linear" else None,
                ).squeeze()  # (channel, original_length)
                signals.append(signal_resized)

            # Reassemble 5D signal
            cropped_sample = torch.stack(
                signals, dim=0
            )  # (5, channel, original_length)
            cropped_data.append(cropped_sample.unsqueeze(0))

        # Concatenate augmented data
        cropped_data = torch.cat(
            cropped_data, dim=0
        )  # (num, 5, channel, original_length)
        augmented_data.append(cropped_data)
        augmented_label.append(label)

    # Merge all data
    augmented_data = torch.cat(augmented_data, dim=0)
    augmented_label = torch.cat(augmented_label, dim=0)
    return augmented_data, augmented_label


# Data augmentation functions
def augment_data(
    data: torch.Tensor,
    label: torch.Tensor,
    method: str = "gaussian_noise",
    expand_factor: int = 2,
    **kwargs,
) -> tuple:
    """
    Unified interface for four augmentation methods.
    Args:
        method: options "gaussian_noise", "time_masking", "phase_shuffling", "random_crop"
        expand_factor: augmentation factor (original data + expand_factor times augmented data)
        kwargs: method-specific parameters (e.g., noise_std, mask_ratio, etc.)
    """
    if method == "gaussian_noise":
        return gaussian_noise_augment(data, label, expand_factor, **kwargs)
    elif method == "time_masking":
        return time_masking_augment(data, label, expand_factor, **kwargs)
    elif method == "phase_shuffling":
        return phase_shuffling_augment(data, label, expand_factor, **kwargs)
    elif method == "random_crop":
        return random_crop_augment(data, label, expand_factor, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

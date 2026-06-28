"""Data loading utilities for Gaussian Splatting experiments."""

import random
from pathlib import Path
from typing import List, Union


def get_random_icons(
    n: int,
    dataset_path: Union[str, Path] = "data/arknights-pfp-dataset/all",
    seed: int = None
) -> List[Path]:
    """
    Randomly select N icon images from the Arknights dataset, excluding default.png files.
    
    Args:
        n: Number of icons to randomly select
        dataset_path: Path to the 'all' directory containing operator folders
        seed: Optional random seed for reproducibility
    
    Returns:
        List of Path objects pointing to the selected PNG files
        
    Raises:
        ValueError: If N is greater than the number of available non-default icons
        FileNotFoundError: If the dataset path doesn't exist
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Find all PNG files recursively, excluding default.png
    all_icons = [
        p for p in dataset_path.rglob("*.png")
        if p.name != "default.png"
    ]
    
    if not all_icons:
        raise ValueError(f"No non-default icons found in {dataset_path}")
    
    if n > len(all_icons):
        raise ValueError(
            f"Requested {n} icons but only {len(all_icons)} non-default icons available"
        )
    
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    return random.sample(all_icons, n)


def get_random_icons_as_strings(
    n: int,
    dataset_path: Union[str, Path] = "data/arknights-pfp-dataset/all",
    seed: int = None
) -> List[str]:
    """
    Same as get_random_icons but returns string paths instead of Path objects.
    
    Args:
        n: Number of icons to randomly select
        dataset_path: Path to the 'all' directory containing operator folders
        seed: Optional random seed for reproducibility
    
    Returns:
        List of string paths to the selected PNG files
    """
    return [str(p) for p in get_random_icons(n, dataset_path, seed)]


def get_operator_name(icon_path: Union[str, Path]) -> str:
    """
    Extract the operator name from an icon path.
    
    The operator name is the parent directory of the icon file.
    For example: 'data/arknights-pfp-dataset/all/silverash/silverash-elite-2-icon.png'
    returns 'silverash'
    
    Args:
        icon_path: Path to an icon file (string or Path object)
    
    Returns:
        Operator name as a string
    """
    return Path(icon_path).parent.name

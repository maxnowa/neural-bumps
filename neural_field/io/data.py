"""Data loading and saving utilities."""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from neural_field.io.paths import get_paths, Paths


def load_stable_params(
    path: Optional[Union[str, Path]] = None,
    paths: Optional[Paths] = None,
) -> pd.DataFrame:
    """Load stable kernel parameters from CSV.

    Args:
        path: Path to CSV file. If None, uses default location.
        paths: Paths instance for finding default location.

    Returns:
        DataFrame with stable kernel parameters.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        df = load_stable_params()
        print(f"Loaded {len(df)} stable kernels")
    """
    if path is None:
        if paths is None:
            paths = get_paths()
        path = paths.data_file("stable_params.csv")
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Stable params file not found: {path}")

    return pd.read_csv(path)


def load_candidates(
    path: Optional[Union[str, Path]] = None,
    paths: Optional[Paths] = None,
) -> pd.DataFrame:
    """Load candidate kernels from CSV.

    Args:
        path: Path to CSV file. If None, uses default location.
        paths: Paths instance for finding default location.

    Returns:
        DataFrame with candidate kernels.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if path is None:
        if paths is None:
            paths = get_paths()
        path = paths.data_file("candidates.csv")
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    return pd.read_csv(path)


def load_verified_bias_curves(
    path: Optional[Union[str, Path]] = None,
    paths: Optional[Paths] = None,
) -> pd.DataFrame:
    """Load verified bias curves from CSV.

    Args:
        path: Path to CSV file. If None, uses default location.
        paths: Paths instance for finding default location.

    Returns:
        DataFrame with verified bias curves.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if path is None:
        if paths is None:
            paths = get_paths()
        path = paths.data_file("verified_bias_curves.csv")
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Verified bias curves file not found: {path}")

    return pd.read_csv(path)


def save_results(
    df: pd.DataFrame,
    filename: str,
    path: Optional[Union[str, Path]] = None,
    paths: Optional[Paths] = None,
    overwrite: bool = False,
) -> Path:
    """Save results DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filename: Output filename (with .csv extension).
        path: Full path to save to. If None, uses data directory.
        paths: Paths instance for finding default location.
        overwrite: Whether to overwrite existing file.

    Returns:
        Path where file was saved.

    Raises:
        FileExistsError: If file exists and overwrite=False.
    """
    if path is None:
        if paths is None:
            paths = get_paths()
        path = paths.data_file(filename)
    else:
        path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Use overwrite=True to replace."
        )

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Saved to: {path}")
    return path


def save_figure(
    fig,
    filename: str,
    path: Optional[Union[str, Path]] = None,
    paths: Optional[Paths] = None,
    dpi: int = 300,
    overwrite: bool = False,
) -> Path:
    """Save matplotlib figure to file.

    Args:
        fig: Matplotlib figure object.
        filename: Output filename (with extension like .png or .pdf).
        path: Full path to save to. If None, uses plots directory.
        paths: Paths instance for finding default location.
        dpi: Resolution for raster formats.
        overwrite: Whether to overwrite existing file.

    Returns:
        Path where file was saved.

    Raises:
        FileExistsError: If file exists and overwrite=False.
    """
    if path is None:
        if paths is None:
            paths = get_paths()
        path = paths.plot_file(filename)
    else:
        path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Use overwrite=True to replace."
        )

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to: {path}")
    return path

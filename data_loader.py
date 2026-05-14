"""Sentinel-2 STAC extraction, degradation, and patch dataset utilities."""

from __future__ import annotations

import os

# Disable EC2 metadata lookup to prevent timeout on non-AWS machines
# This must happen before any boto3/rasterio imports
os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import GUIDE_BANDS_10M, PATCH, STAC, TARGET_BANDS_20M

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    rasterio = None
    Window = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None

try:
    from pystac_client import Client
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    Client = None
    _PYSTAC_IMPORT_ERROR = exc
else:
    _PYSTAC_IMPORT_ERROR = None

try:
    import planetary_computer
except ImportError:  # pragma: no cover
    planetary_computer = None


BAND_ASSET_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "B02": ("B02", "blue"),
    "B03": ("B03", "green"),
    "B04": ("B04", "red"),
    "B08": ("B08", "nir", "nir08"),
    "B05": ("B05", "rededge1"),
    "B06": ("B06", "rededge2"),
    "B07": ("B07", "rededge3"),
    "B8A": ("B8A", "B08A", "nir08", "nir_narrow"),
    "B11": ("B11", "swir16"),
    "B12": ("B12", "swir22"),
}


@dataclass(frozen=True)
class PatchIndex:
    """Location of a patch in a specific STAC item."""

    item_index: int
    row_10m: int
    col_10m: int


def _require_rasterio() -> None:
    if rasterio is None:
        raise ImportError("rasterio is required for data loading") from _RASTERIO_IMPORT_ERROR


def _require_pystac() -> None:
    if Client is None:
        raise ImportError("pystac-client is required for STAC search") from _PYSTAC_IMPORT_ERROR


def search_stac_items(
    bbox: Tuple[float, float, float, float] = STAC.bbox,
    datetime_range: str = STAC.datetime_range,
    max_cloud_cover: int = STAC.max_cloud_cover,
    max_items: int = STAC.max_items,
    url: str = STAC.url,
    collection: str = STAC.collection,
    max_retries: int = 3,
) -> List[object]:
    """Query Earth Search for Sentinel-2 L2A items with retry logic.
    
    Uses planetary-computer module if available for better AWS connectivity.
    """

    _require_pystac()
    
    last_error = None
    for attempt in range(max_retries):
        try:
            client = Client.open(url)
            search = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                max_items=max_items,
            )
            items = list(search.items())
            
            # Sign items with planetary_computer if available
            if planetary_computer is not None and items:
                try:
                    items = [planetary_computer.sign(item) for item in items]
                except Exception as e:
                    if False:  # Log but don't fail if signing fails
                        print(f"Warning: planetary_computer signing failed: {e}")
            
            if items:
                print(f"Successfully found {len(items)} Sentinel-2 scenes")
            return items
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait_seconds = 2 ** attempt
                print(f"STAC search failed (attempt {attempt + 1}/{max_retries}): {exc}")
                print(f"Retrying in {wait_seconds}s...")
                import time
                time.sleep(wait_seconds)
            else:
                print(f"STAC search failed after {max_retries} attempts")
    
    raise RuntimeError(
        f"Failed to query STAC catalog after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Verify network connectivity or try using --synthetic flag to test locally."
    ) from last_error


def _resolve_asset_href(item: object, band_name: str) -> str:
    assets = getattr(item, "assets")
    candidates = BAND_ASSET_CANDIDATES[band_name]
    for candidate in candidates:
        if candidate in assets:
            return assets[candidate].href
    available = ", ".join(sorted(assets.keys()))
    raise KeyError(f"Unable to resolve band {band_name}; available assets: {available}")


def _build_start_positions(image_length: int, patch_size: int, stride: int) -> List[int]:
    if image_length <= patch_size:
        return [0]
    starts = list(range(0, image_length - patch_size + 1, stride))
    final_start = image_length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    unique_starts = sorted(set(max(0, start - (start % 2)) for start in starts))
    return unique_starts


def _read_window_stack(
    asset_hrefs: Sequence[str],
    window: Window,
    normalize: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    import time
    _require_rasterio()
    bands: List[np.ndarray] = []
    for idx, href in enumerate(asset_hrefs):
        t0 = time.perf_counter()
        try:
            with rasterio.open(href) as src:
                array = src.read(1, window=window, boundless=False).astype(np.float32)
            t1 = time.perf_counter()
            if verbose:
                print(f"    band {idx} ({href[:80]}...) read in {t1-t0:.3f}s")
        except Exception as e:
            t1 = time.perf_counter()
            print(f"    band {idx} read failed after {t1-t0:.3f}s: {e}")
            raise
        if normalize:
            array = array / 10000.0
        bands.append(array)
    stacked = np.stack(bands, axis=0)
    return torch.from_numpy(stacked)


def _downsample(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def _upsample(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


class WaldProtocolDataset(Dataset):
    """Patch-level dataset implementing Wald's protocol on Sentinel-2 L2A COGs."""

    def __init__(
        self,
        items: Optional[Sequence[object]] = None,
        bbox: Tuple[float, float, float, float] = STAC.bbox,
        datetime_range: str = STAC.datetime_range,
        max_cloud_cover: int = STAC.max_cloud_cover,
        max_items: int = STAC.max_items,
        patch_size_10m: int = PATCH.patch_size_10m,
        patch_size_20m: int = PATCH.patch_size_20m,
        overlap_10m: int = PATCH.overlap_10m,
        stac_url: str = STAC.url,
        collection: str = STAC.collection,
        max_patches_per_item: Optional[int] = None,
        max_items_to_use: Optional[int] = None,
        max_total_patches: Optional[int] = None,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.patch_size_10m = patch_size_10m
        self.patch_size_20m = patch_size_20m
        self.overlap_10m = overlap_10m
        self.stride_10m = patch_size_10m - overlap_10m
        self.max_patches_per_item = max_patches_per_item
        self.max_items_to_use = max_items_to_use
        self.max_total_patches = max_total_patches
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.seed = seed
        self.verbose = verbose

        if items is None:
            items = search_stac_items(
                bbox=bbox,
                datetime_range=datetime_range,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                url=stac_url,
                collection=collection,
            )
        self.items = list(items)
        if self.max_items_to_use is not None and self.max_items_to_use > 0:
            self.items = self.items[: self.max_items_to_use]
        if self.verbose:
            print(f"Building patch index from {len(self.items)} scenes...")
        self.patch_index: List[PatchIndex] = self._build_patch_index()
        if self.verbose:
            print(f"Patch index ready with {len(self.patch_index)} samples")

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_patch_index(self) -> List[PatchIndex]:
        _require_rasterio()
        patch_index: List[PatchIndex] = []
        for item_index, item in enumerate(self.items):
            if self.verbose:
                print(f"Indexing scene {item_index + 1}/{len(self.items)}")
            guide_href = _resolve_asset_href(item, "B02")
            with rasterio.open(guide_href) as src:
                width = src.width
                height = src.height
            starts_row = _build_start_positions(height, self.patch_size_10m, self.stride_10m)
            starts_col = _build_start_positions(width, self.patch_size_10m, self.stride_10m)
            item_positions = [
                PatchIndex(item_index=item_index, row_10m=row_10m, col_10m=col_10m)
                for row_10m in starts_row
                for col_10m in starts_col
            ]
            if self.max_patches_per_item is not None and len(item_positions) > self.max_patches_per_item:
                rng = random.Random(self.seed + item_index)
                item_positions = rng.sample(item_positions, self.max_patches_per_item)
            patch_index.extend(item_positions)
            if self.verbose:
                print(f"  Added {len(item_positions)} patches (running total: {len(patch_index)})")
            if self.max_total_patches is not None and self.max_total_patches > 0 and len(patch_index) >= self.max_total_patches:
                if self.verbose:
                    print(f"Reached max total patches: {self.max_total_patches}. Stopping indexing.")
                return patch_index[: self.max_total_patches]
        return patch_index

    def __len__(self) -> int:
        return len(self.patch_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        _require_rasterio()
        patch = self.patch_index[index]
        item = self.items[patch.item_index]

        # try loading from cache if available
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"patch_{patch.item_index}_{patch.row_10m}_{patch.col_10m}.npz"
            if cache_file.exists():
                data = np.load(str(cache_file))
                return {
                    "input": torch.from_numpy(data["input"]).float(),
                    "target": torch.from_numpy(data["target"]).float(),
                    "guide_10m": torch.from_numpy(data["guide_10m"]).float(),
                    "guide_20m": torch.from_numpy(data["guide_20m"]).float(),
                    "blurry_20m": torch.from_numpy(data["blurry_20m"]).float(),
                    "metadata": torch.tensor([patch.item_index, patch.row_10m, patch.col_10m], dtype=torch.int64),
                }

        guide_hrefs = [_resolve_asset_href(item, band) for band in GUIDE_BANDS_10M]
        target_hrefs = [_resolve_asset_href(item, band) for band in TARGET_BANDS_20M]

        guide_window_10m = Window(patch.col_10m, patch.row_10m, self.patch_size_10m, self.patch_size_10m)
        guide_10m = _read_window_stack(guide_hrefs, guide_window_10m, verbose=False)

        row_20m = patch.row_10m // 2
        col_20m = patch.col_10m // 2
        target_window_20m = Window(col_20m, row_20m, self.patch_size_20m, self.patch_size_20m)

        guide_20m = _downsample(guide_10m, (self.patch_size_20m, self.patch_size_20m))
        target_20m = _read_window_stack(target_hrefs, target_window_20m, verbose=False)
        blurry_40m = _downsample(target_20m, (self.patch_size_20m // 2, self.patch_size_20m // 2))
        blurry_20m = _upsample(blurry_40m, (self.patch_size_20m, self.patch_size_20m))

        model_input = torch.cat([guide_20m, blurry_20m], dim=0)

        # save to cache if requested
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"patch_{patch.item_index}_{patch.row_10m}_{patch.col_10m}.npz"
            np.savez_compressed(
                str(cache_file),
                input=model_input.numpy(),
                target=target_20m.numpy(),
                guide_10m=guide_10m.numpy(),
                guide_20m=guide_20m.numpy(),
                blurry_20m=blurry_20m.numpy(),
            )

        return {
            "input": model_input.float(),
            "target": target_20m.float(),
            "guide_10m": guide_10m.float(),
            "guide_20m": guide_20m.float(),
            "blurry_20m": blurry_20m.float(),
            "metadata": torch.tensor([patch.item_index, patch.row_10m, patch.col_10m], dtype=torch.int64),
        }

    def prefetch_to_cache(self, cache_dir: str, max_files: Optional[int] = None) -> int:
        """Prefetch patch files and store them in cache_dir as compressed npz files.

        Returns the number of files written.
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        written = 0
        for idx, patch in enumerate(self.patch_index):
            if max_files is not None and written >= max_files:
                break
            cache_file = cache_path / f"patch_{patch.item_index}_{patch.row_10m}_{patch.col_10m}.npz"
            if cache_file.exists():
                written += 1
                continue
            try:
                sample = self.__getitem__(idx)
            except Exception:
                continue
            np.savez_compressed(
                str(cache_file),
                input=sample["input"].numpy(),
                target=sample["target"].numpy(),
                guide_10m=sample["guide_10m"].numpy(),
                guide_20m=sample["guide_20m"].numpy(),
                blurry_20m=sample["blurry_20m"].numpy(),
            )
            written += 1
        return written


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def split_dataset(dataset: Dataset, validation_split: float, seed: int = 42) -> Tuple[Dataset, Dataset]:
    validation_size = max(1, int(len(dataset) * validation_split))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, validation_size], generator=generator)

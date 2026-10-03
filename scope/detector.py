"""Optional TorchVision COCO instance masks for recorded RGB-D frames."""

import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy import ndimage

from .data import RgbdFrame
from .objects import ObjectObservation2D


def clean_instance_mask(mask: np.ndarray, min_pixels: int = 24) -> np.ndarray:
    """Remove isolated mask islands while preserving separate visible parts."""
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    threshold = max(min_pixels, int(sizes[1:].max() * .015))
    keep = np.flatnonzero(sizes >= threshold)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def _ensure_official_weights(torch, weights) -> None:
    """Keep weights in the user cache and check the upstream filename hash."""
    filename = weights.url.rsplit("/", 1)[-1]
    prefix = filename.rsplit("-", 1)[-1].split(".")[0]
    destination = Path(torch.hub.get_dir()) / "checkpoints" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    def matches(path: Path) -> bool:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().startswith(prefix)
    if destination.exists():
        if matches(destination):
            return
        raise RuntimeError(f"Cached detector weights failed SHA-256 check: {destination}")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is needed to download the official detector weights")
    temporary = destination.with_suffix(".download")
    try:
        print(f"Downloading official TorchVision weights to {destination}", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                        "--connect-timeout", "15", "--max-time", "180",
                        "--output", str(temporary), weights.url], check=True)
        if not matches(temporary):
            raise RuntimeError("Downloaded detector weights failed SHA-256 check")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class TorchvisionMaskDetector:
    """COCO Mask R-CNN v2; weights remain in PyTorch's user cache."""

    name = "torchvision-maskrcnn-v2-coco"

    def __init__(self, score_threshold: float = .55, min_pixels: int = 24,
                 model_width: int = 480,
                 include_classes: set[str] | None = None):
        try:
            import torch
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2)
        except ImportError as exc:
            raise RuntimeError("Install the optional detector: pip install -e '.[detector]'") from exc
        self.torch = torch
        self.score_threshold = score_threshold
        self.min_pixels = min_pixels
        self.include_classes = include_classes
        self.categories = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"]
        self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        _ensure_official_weights(torch, self.weights)
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=self.weights, min_size=model_width, max_size=model_width * 2)
        self.model.eval()
        self.model.to("cpu")
        torch.set_num_threads(min(4, torch.get_num_threads()))

    def detect(self, frame: RgbdFrame) -> list[ObjectObservation2D]:
        image = self.torch.from_numpy(frame.rgb.copy()).permute(2, 0, 1).float() / 255.
        with self.torch.inference_mode():
            predictions = self.model([image])[0]
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        masks = predictions["masks"].cpu().numpy()[:, 0]
        result = []
        for index, (score, label_id, mask_score) in enumerate(zip(scores, labels, masks)):
            if score < self.score_threshold:
                continue
            mask = clean_instance_mask(mask_score >= .5, self.min_pixels)
            if mask.sum() < self.min_pixels:
                continue
            label = self.categories[int(label_id)]
            if label == "person" and self.include_classes is None:
                continue  # Human understanding is a later milestone.
            if self.include_classes is not None and label not in self.include_classes:
                continue
            result.append(ObjectObservation2D(
                f"{frame.frame_id}:det-{index:02d}", frame.frame_id, mask,
                {label: float(score), "unknown": float(1. - score)},
                float(score), self.name))
        return result

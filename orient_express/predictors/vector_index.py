import os
import json
from dataclasses import dataclass
import warnings
from typing import Any

import yaml
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .predictor import Predictor
from ..utils.paths import get_metadata_path


@dataclass
class SearchResult:
    labels: list
    score: float


@dataclass
class CropSpec:
    """An image path with a bounding box to crop before feature extraction.

    bbox is (x1, y1, x2, y2) in pixel coordinates, as expected by PIL.Image.crop.
    """

    path: str
    bbox: tuple[int, int, int, int]


class VectorIndex(Predictor):
    model_type = "vector-index"
    ARTIFACT_FILENAME = "vectors.npz"

    def __init__(
        self,
        vectors: np.ndarray,
        labels: list[int | str | tuple],
        normalize: bool = False,
    ):
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-dimensional (M, D), got shape {vectors.shape}"
            )
        if len(labels) != vectors.shape[0]:
            raise ValueError(
                f"labels length ({len(labels)}) must match number of vectors ({vectors.shape[0]})"
            )
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms = np.where(norms == 0, 1, norms)
            vectors = vectors / norms

        self.vectors = vectors
        self.labels = labels
        self.label_to_idx = {}

        self.multi_label = False

        for i, label in enumerate(labels):
            if label not in self.label_to_idx:
                self.label_to_idx[label] = []
            self.label_to_idx[label].append(i)

        self.model_path = None

    def search(self, query: np.ndarray, k: int = 1) -> list[SearchResult]:
        query = query.reshape(1, -1)
        similarities = (query @ self.vectors.T).squeeze(0)
        k = min(k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            SearchResult(labels=self.labels[idx], score=float(similarities[idx]))
            for idx in top_indices
        ]

    def search_batch(self, queries: np.ndarray, k: int = 1) -> list[list[SearchResult]]:
        similarities = queries @ self.vectors.T
        k = min(k, similarities.shape[1])

        all_results = []
        for i in range(similarities.shape[0]):
            row = similarities[i]
            top_indices = np.argpartition(row, -k)[-k:]
            top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
            all_results.append(
                [
                    SearchResult(labels=self.labels[idx], score=float(row[idx]))
                    for idx in top_indices
                ]
            )
        return all_results

    def get_by_idx(self, idx: int) -> np.ndarray:
        return self.vectors[idx].ex

    def get_by_idxs(self, idxs: list[int]) -> np.ndarray:
        return np.expand_dims(self.vectors[idxs], axis=0)

    def get_by_label(self, label: Any) -> np.ndarray:
        return self.vectors[self.label_to_idx[label]]

    def get_by_labels(self, labels: list[Any]) -> np.ndarray:
        idxs = []
        for label in labels:
            idxs.extend(self.label_to_idx[label])
        return self.vectors[idxs]

    def aggregate(self, per_label: bool = False) -> "VectorIndex":
        """
        Aggregate the vectors into a single centroid per unique label group.

        Args:
            per_label: if True, create one centroid per individual label
                (splitting label groups (tuples) apart). If False (default),
                create one centroid per unique label group.
        """
        key_to_indices: dict = {}
        for i, label in enumerate(self.labels):
            if per_label and isinstance(label, tuple):
                keys = label
            else:
                keys = [label]
            for key in keys:
                if key not in key_to_indices:
                    key_to_indices[key] = []
                key_to_indices[key].append(i)

        sorted_keys = sorted(key_to_indices.keys(), key=str)
        centroids = np.empty(
            (len(sorted_keys), self.vectors.shape[1]), dtype=self.vectors.dtype
        )

        new_labels = []
        for i, key in enumerate(sorted_keys):
            indices = key_to_indices[key]
            centroid = self.vectors[indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[i] = centroid
            new_labels.append(key)

        return VectorIndex(vectors=centroids, labels=new_labels)

    def get_serving_container_image_uri(self) -> str:
        warnings.warn(
            "VectorIndex does not support serving via a container. Returning incompatible image URI."
        )
        return "us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx:v2.1.2"

    def get_serving_container_health_route(self, model_name) -> str:
        return f"/v1/models/{model_name}"

    def get_serving_container_predict_route(self, model_name) -> str:
        return f"/v1/models/{model_name}:predict"

    def dump(self, dir: str) -> list[str]:
        artifact_path = os.path.join(dir, self.ARTIFACT_FILENAME)
        np.savez(
            artifact_path,
            vectors=self.vectors,
            labels_json=json.dumps(self.labels),
        )

        metadata = {
            "model_type": self.model_type,
            "model_file": self.ARTIFACT_FILENAME,
        }
        metadata_path = get_metadata_path(dir)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

        return [metadata_path, artifact_path]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def __repr__(self) -> str:
        return (
            f"VectorIndex({len(self)} vectors, dim={self.dim}, "
            f"{len(set(self.labels))} unique labels)"
        )


class _CropDataset:
    def __init__(self, crops: list[Image.Image | str | CropSpec]):
        self.crops = crops

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        if isinstance(crop, CropSpec):
            img = Image.open(crop.path).convert("RGB")
            return img.crop(crop.bbox)
        if isinstance(crop, str):
            return Image.open(crop).convert("RGB")
        if isinstance(crop, Image.Image):
            return crop
        raise TypeError(
            f"Each crop must be a PIL Image, a file path string, or a CropSpec, got {type(crop)}"
        )


def _collate_pil(batch):
    return list(batch)


def build_vector_index(
    crops: list[Image.Image | str | CropSpec],
    labels: list[int | str | tuple],
    feature_extractor,
    batch_size: int = 128,
    normalize: bool = True,
    num_workers: int = 0,
) -> VectorIndex:
    """Build a VectorIndex by extracting features from crops.

    Args:
        crops: images to extract features from. Each element can be a PIL
            Image (used directly), a file path string (loaded as a whole
            image), or a CropSpec (loaded and cropped to the specified bbox).
        labels: one label per crop. labels can be any hashable type, includeing
            iterables like tuples.
        feature_extractor: anything with a .predict(list[Image]) method that
            returns objects with a .feature attribute (e.g. FeatureExtractionPredictor).
        batch_size: number of crops to process at once.
        normalize: whether to L2-normalize the resulting feature vectors.
        multi_label: if True, labels are already lists of labels per crop.
        num_workers: number of DataLoader workers for parallel image loading.
    """
    if len(crops) != len(labels):
        raise ValueError(
            f"crops length ({len(crops)}) must match labels length ({len(labels)})"
        )

    loader = DataLoader(
        _CropDataset(crops),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_pil,
    )

    all_features = []
    for batch in tqdm(loader):
        results = feature_extractor.predict(batch)
        all_features.extend([r.feature for r in results])

    vectors = np.vstack(all_features)
    return VectorIndex(vectors=vectors, labels=labels, normalize=normalize)

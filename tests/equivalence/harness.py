"""Core of the golden equivalence suite.

Version-agnostic on purpose: this module must run unmodified against both the
pre-refactor (2.4.x) and current APIs, because goldens are generated on the
old code and enforced on every branch after it. Keep imports and predictor
interactions to surface area that exists in both.

Prediction always runs one image per predict() call: some deployed ONNX
graphs have a fixed batch dimension of 1, and per-image calls match how the
production pipelines and Vertex endpoints invoke the models.
"""

import hashlib
import io
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import numpy as np
import yaml
from PIL import Image

MANIFEST_ENV = "ORIENT_EXPRESS_TEST_MANIFEST"
DOCKER_IMAGE_ENV = "ORIENT_EXPRESS_TEST_DOCKER_IMAGE"
GOLDENS_DIR_ENV = "ORIENT_EXPRESS_TEST_GOLDENS_DIR"  # local override (else GCS)
REPORT_ENV = "ORIENT_EXPRESS_TEST_REPORT"  # report path override
DEVICE_ENV = "ORIENT_EXPRESS_TEST_DEVICE"  # predictor device (default cpu)

DEFAULT_TOLERANCES = {
    "score_atol": 1e-3,
    "bbox_atol": 0.5,
    "mask_iou_min": 0.999,
    "pixel_mismatch_max": 1e-4,
    "cosine_min": 0.9999,
    "class_scores_atol": 1e-3,
}


# ----------------------------------------------------------------- manifest


def manifest_location() -> str | None:
    return os.environ.get(MANIFEST_ENV)


def cache_root() -> str:
    return os.environ.get(
        "ORIENT_EXPRESS_TEST_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "orient-express-test-assets"),
    )


def _read_gcs_text(gs_url: str) -> str:
    from google.cloud import storage

    bucket_name, path = gs_url.replace("gs://", "").split("/", 1)
    client = storage.Client()
    return client.bucket(bucket_name).blob(path).download_as_text()


def load_manifest() -> dict:
    location = manifest_location()
    if location is None:
        raise RuntimeError(f"{MANIFEST_ENV} is not set")
    if location.startswith("gs://"):
        text = _read_gcs_text(location)
    else:
        with open(location) as f:
            text = f.read()
    manifest = yaml.safe_load(text)
    manifest.setdefault("default_tolerances", {})
    return manifest


def case_tolerances(manifest: dict, case_cfg: dict) -> dict:
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances.update(manifest.get("default_tolerances", {}))
    tolerances.update(case_cfg.get("tolerances", {}))
    return tolerances


def case_variants(case_cfg: dict) -> list[dict]:
    return case_cfg.get("variants", [{"name": "default", "predict_params": {}}])


def fetch_prefix(gs_prefix: str, local_dir: str):
    """Download every blob under gs_prefix into local_dir (skip existing)."""
    from google.cloud import storage

    bucket_name, prefix = gs_prefix.replace("gs://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    client = storage.Client()
    os.makedirs(local_dir, exist_ok=True)
    found = False
    for blob in client.bucket(bucket_name).list_blobs(prefix=prefix):
        relative = blob.name[len(prefix) :]
        if not relative:
            continue
        found = True
        destination = os.path.join(local_dir, relative)
        if os.path.exists(destination) and os.path.getsize(destination) == blob.size:
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        blob.download_to_filename(destination)
    if not found:
        raise FileNotFoundError(f"no blobs under {gs_prefix}")


def fetch_case_assets(manifest: dict, case: str) -> str:
    """Download cases/<case>/{model,images} to the local cache; return dir."""
    case_dir = os.path.join(cache_root(), "cases", case)
    fetch_prefix(f"{manifest['gcs_prefix']}/cases/{case}", case_dir)
    return case_dir


def fetch_golden(manifest: dict, name: str, goldens_dir: str | None = None):
    """Load a golden JSON from a local dir (if given) or the GCS prefix."""
    if goldens_dir is None:
        goldens_dir = os.environ.get(GOLDENS_DIR_ENV)
    if goldens_dir is not None:
        path = os.path.join(goldens_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    try:
        text = _read_gcs_text(f"{manifest['gcs_prefix']}/goldens/{name}.json")
    except Exception:
        return None
    return json.loads(text)


def golden_annotated_dir(
    manifest: dict, case: str, goldens_dir: str | None = None, kind: str = "annotated"
) -> str | None:
    """Local dir holding the golden's annotated images (fetched if remote).

    kind: "annotated" (predictor level) or "docker-annotated" (serving level).
    """
    if goldens_dir is None:
        goldens_dir = os.environ.get(GOLDENS_DIR_ENV)
    if goldens_dir is not None:
        path = os.path.join(goldens_dir, case, kind)
        return path if os.path.isdir(path) else None
    local = os.path.join(cache_root(), "goldens", case, kind)
    try:
        fetch_prefix(f"{manifest['gcs_prefix']}/goldens/{case}/{kind}", local)
    except FileNotFoundError:
        return None
    return local


# ------------------------------------------------------------ case running


def load_case_images(case_dir: str) -> dict[str, Image.Image]:
    images_dir = os.path.join(case_dir, "images")
    images = {}
    if not os.path.isdir(images_dir):
        return images
    for name in sorted(os.listdir(images_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            images[name] = Image.open(os.path.join(images_dir, name)).convert("RGB")
    return images


def load_case_predictor(case_dir: str):
    from orient_express.predictors import get_predictor

    device = os.environ.get(DEVICE_ENV, "cpu")
    return get_predictor(os.path.join(case_dir, "model"), device)


def case_model_type(case_dir: str) -> str:
    with open(os.path.join(case_dir, "model", "metadata.yaml")) as f:
        return yaml.safe_load(f)["model_type"]


def rle_encode(mask: np.ndarray) -> dict:
    """Run-length encode a boolean mask (row-major, starting with False runs)."""
    flat = np.asarray(mask, dtype=bool).ravel()
    if flat.size == 0:
        return {"shape": list(mask.shape), "counts": []}
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate([[0], changes, [flat.size]])
    counts = np.diff(boundaries).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"shape": list(mask.shape), "counts": counts}


def rle_decode(encoded: dict) -> np.ndarray:
    total = int(np.prod(encoded["shape"])) if encoded["shape"] else 0
    flat = np.zeros(total, dtype=bool)
    position = 0
    value = False
    for count in encoded["counts"]:
        if value:
            flat[position : position + count] = True
        position += count
        value = not value
    return flat.reshape(encoded["shape"])


def _full_res_mask(prediction, image: Image.Image) -> np.ndarray:
    """Instance-seg mask at image resolution, across library versions."""
    if hasattr(prediction, "resized_mask"):  # >= 3.0: model-res mask + resize
        return prediction.resized_mask(image)
    return np.asarray(prediction.mask, dtype=bool)  # <= 2.4.x: already full-res


def _predict_one(predictor, image: Image.Image, predict_params: dict):
    """One image per call: fixed-batch-1 graphs + production traffic shape."""
    return predictor.predict([image], **predict_params)[0]


def evaluate_variant(
    predictor, model_type: str, images, predict_params: dict, annotate: bool = True
):
    """Predict once per image; return (outputs, annotated JPEG bytes or None)."""
    outputs = {}
    annotated = {}
    for name, image in images.items():
        pred = _predict_one(predictor, image, predict_params)
        outputs[name] = _extract_prediction(pred, model_type, image)
        if annotate and model_type in ANNOTATABLE_TYPES:
            rendered = predictor.get_annotated_image(image, pred)
            buffer = io.BytesIO()
            rendered.convert("RGB").save(buffer, format="JPEG", quality=85)
            annotated[name] = buffer.getvalue()
        else:
            annotated[name] = None
    return outputs, annotated


def extract_outputs(predictor, model_type: str, images, predict_params: dict):
    """Run predict per image and reduce results to version-agnostic JSON."""
    return evaluate_variant(
        predictor, model_type, images, predict_params, annotate=False
    )[0]


def _extract_prediction(pred, model_type: str, image):
    if model_type == "classification-onnx":
        return {
            "class": pred.clss,
            "score": float(pred.score),
            "class_scores": {k: float(v) for k, v in pred.class_scores.items()},
        }
    if model_type == "multi-label-classification-onnx":
        return {
            "classes": sorted(pred.classes),
            "class_scores": {k: float(v) for k, v in pred.class_scores.items()},
        }
    if model_type == "object-detection-onnx":
        return {
            "detections": [
                {
                    "class": d.clss,
                    "score": float(d.score),
                    "bbox": [float(v) for v in d.bbox],
                }
                for d in pred
            ]
        }
    if model_type == "instance-segmentation-onnx":
        return {
            "detections": [
                {
                    "class": d.clss,
                    "score": float(d.score),
                    "bbox": [float(v) for v in d.bbox],
                    "mask_rle": rle_encode(_full_res_mask(d, image)),
                }
                for d in pred
            ]
        }
    if model_type == "semantic-segmentation-onnx":
        return {
            "class_mask_rle_per_class": {
                str(class_id): rle_encode(pred.class_mask == class_id)
                for class_id in np.unique(pred.class_mask).tolist()
            },
            "valid_mask_rle": rle_encode(np.asarray(pred.valid_mask, dtype=bool)),
        }
    if model_type == "feature-extraction-onnx":
        return {"embedding": [float(v) for v in np.ravel(pred.feature)]}
    raise ValueError(f"no extractor for model_type '{model_type}'")


ANNOTATABLE_TYPES = {
    "object-detection-onnx",
    "instance-segmentation-onnx",
    "semantic-segmentation-onnx",
}


def render_annotated(predictor, model_type: str, images, predict_params: dict):
    """Annotated JPEG bytes per image (None-valued for non-visual models)."""
    return evaluate_variant(predictor, model_type, images, predict_params)[1]


def _plain_labels(labels):
    if isinstance(labels, (list, tuple)):
        return list(labels)
    return labels


def run_vector_index_case(index, query_embeddings: dict, top_k: int):
    """query_embeddings: image name -> embedding list (from another case)."""
    outputs = {}
    for name, embedding in query_embeddings.items():
        query = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        results = index.search(query, k=top_k)
        outputs[name] = {
            "results": [
                {"labels": _plain_labels(r.labels), "score": float(r.score)}
                for r in results
            ]
        }
    return outputs


# -------------------------------------------------------------- comparison


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _record(records, field, golden, got, ok, detail=""):
    records.append(
        {"field": field, "golden": golden, "got": got, "ok": bool(ok), "detail": detail}
    )


def compare_outputs(got: dict, golden: dict, tolerances: dict) -> list[dict]:
    """Compare per-image outputs; returns structured records (passing + failing).

    Record fields: field (path-like), golden, got, ok, detail. The failing
    subset is what gates the tests; the full set feeds the HTML report.
    """
    records = []
    for name in golden:
        if name not in got:
            _record(records, name, "present", "missing", False)
            continue
        for r in _compare_image(got[name], golden[name], tolerances):
            r["field"] = f"{name}: {r['field']}"
            records.append(r)
    for name in got:
        if name not in golden:
            _record(records, name, "absent", "present", False, "not in golden")
    return records


def failures(records: list[dict]) -> list[str]:
    return [
        f"{r['field']}: golden={r['golden']} got={r['got']} {r['detail']}".strip()
        for r in records
        if not r["ok"]
    ]


def _bbox_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_detections(expected: list, actual: list) -> list[tuple]:
    """Greedy order-independent matching (same class, best IoU >= 0.5).

    Detection order is not part of the semantic contract: near-tied scores
    (deltas ~1e-6) swap TopK order across onnxruntime versions.
    """
    pairs = []
    used = set()
    for i, g in enumerate(expected):
        best, best_iou = None, 0.5
        for j, a in enumerate(actual):
            if j in used or a["class"] != g["class"]:
                continue
            iou = _bbox_iou(g["bbox"], a["bbox"])
            if iou >= best_iou:
                best, best_iou = j, iou
        if best is not None:
            used.add(best)
            pairs.append((i, g, actual[best]))
    return pairs


def _compare_image(got: dict, golden: dict, tol: dict) -> list[dict]:
    records = []
    if "detections" in golden:
        expected, actual = golden["detections"], got.get("detections", [])
        _record(
            records,
            "detection count",
            len(expected),
            len(actual),
            len(actual) == len(expected),
        )
        if len(actual) != len(expected):
            return records
        pairs = _match_detections(expected, actual)
        _record(
            records,
            "detections matched (class + IoU)",
            len(expected),
            len(pairs),
            len(pairs) == len(expected),
        )
        for i, g, a in pairs:
            score_delta = abs(a["score"] - g["score"])
            _record(
                records,
                f"det[{i}].score",
                round(g["score"], 5),
                round(a["score"], 5),
                score_delta <= tol["score_atol"],
                f"Δ{score_delta:.2e} (tol {tol['score_atol']})",
            )
            box_delta = max(
                abs(x - y) for x, y in zip(a["bbox"], g["bbox"], strict=True)
            )
            _record(
                records,
                f"det[{i}].bbox",
                [round(v, 1) for v in g["bbox"]],
                [round(v, 1) for v in a["bbox"]],
                box_delta <= tol["bbox_atol"],
                f"maxΔ{box_delta:.3f}px (tol {tol['bbox_atol']})",
            )
            if "mask_rle" in g:
                iou = _mask_iou(rle_decode(a["mask_rle"]), rle_decode(g["mask_rle"]))
                _record(
                    records,
                    f"det[{i}].mask",
                    "—",
                    f"IoU {iou:.6f}",
                    iou >= tol["mask_iou_min"],
                    f"(min {tol['mask_iou_min']})",
                )
    if "class" in golden:
        _record(
            records,
            "class",
            golden["class"],
            got.get("class"),
            got.get("class") == golden["class"],
        )
        delta = abs(got.get("score", 0) - golden["score"])
        _record(
            records,
            "score",
            round(golden["score"], 5),
            round(got.get("score", 0), 5),
            delta <= tol["score_atol"],
            f"Δ{delta:.2e}",
        )
    if "classes" in golden:  # multi-label
        same = sorted(got.get("classes", [])) == sorted(golden["classes"])
        _record(records, "classes", golden["classes"], got.get("classes"), same)
    if "class_scores" in golden:
        worst_key, worst_delta = None, -1.0
        ok = True
        for key, expected in golden["class_scores"].items():
            actual = got.get("class_scores", {}).get(key)
            if actual is None:
                ok = False
                worst_key, worst_delta = key, float("inf")
                break
            delta = abs(actual - expected)
            if delta > worst_delta:
                worst_key, worst_delta = key, delta
            if delta > tol["class_scores_atol"]:
                ok = False
        _record(
            records,
            "class_scores",
            "—",
            f"worst Δ{worst_delta:.2e} [{worst_key}]",
            ok,
            f"(tol {tol['class_scores_atol']})",
        )
    if "class_mask_rle_per_class" in golden:
        for class_id, encoded in golden["class_mask_rle_per_class"].items():
            expected_mask = rle_decode(encoded)
            actual_encoded = got.get("class_mask_rle_per_class", {}).get(class_id)
            actual_mask = (
                rle_decode(actual_encoded)
                if actual_encoded
                else np.zeros(expected_mask.shape, dtype=bool)
            )
            mismatch = float(np.mean(expected_mask != actual_mask))
            _record(
                records,
                f"class_mask[{class_id}]",
                "—",
                f"pixel mismatch {mismatch:.2e}",
                mismatch <= tol["pixel_mismatch_max"],
                f"(max {tol['pixel_mismatch_max']})",
            )
    if "valid_mask_rle" in golden:
        mismatch = float(
            np.mean(
                rle_decode(got["valid_mask_rle"])
                != rle_decode(golden["valid_mask_rle"])
            )
        )
        _record(
            records,
            "valid_mask",
            "—",
            f"pixel mismatch {mismatch:.2e}",
            mismatch <= tol["pixel_mismatch_max"],
        )
    if "embedding" in golden:
        expected = np.asarray(golden["embedding"], dtype=np.float64)
        actual = np.asarray(got.get("embedding", []), dtype=np.float64)
        if actual.shape != expected.shape:
            _record(
                records,
                "embedding",
                list(expected.shape),
                list(actual.shape),
                False,
                "shape mismatch",
            )
        else:
            denominator = np.linalg.norm(actual) * np.linalg.norm(expected)
            cosine = float(actual @ expected / denominator) if denominator else 1.0
            _record(
                records,
                "embedding",
                "—",
                f"cosine {cosine:.6f}",
                cosine >= tol["cosine_min"],
                f"(min {tol['cosine_min']})",
            )
    if "results" in golden:  # vector index
        expected, actual = golden["results"], got.get("results", [])
        same_labels = [r["labels"] for r in actual] == [r["labels"] for r in expected]
        _record(
            records,
            "top-k labels",
            [r["labels"] for r in expected],
            [r["labels"] for r in actual],
            same_labels,
        )
        if same_labels:
            for i, (g, a) in enumerate(zip(expected, actual, strict=True)):
                delta = abs(a["score"] - g["score"])
                _record(
                    records,
                    f"result[{i}].score",
                    round(g["score"], 5),
                    round(a["score"], 5),
                    delta <= tol["score_atol"],
                    f"Δ{delta:.2e}",
                )
    return records


# ------------------------------------------------------------- provenance


def provenance() -> dict:
    try:
        library_version = package_version("orient_express")
    except PackageNotFoundError:
        library_version = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "library_version": library_version,
        "git_commit": commit,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def stable_digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

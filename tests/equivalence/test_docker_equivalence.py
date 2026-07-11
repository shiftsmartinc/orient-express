"""Serving-level golden equivalence: boots the serving container per case and
compares HTTP responses against docker goldens.

Opt-in: requires ORIENT_EXPRESS_TEST_DOCKER_IMAGE (an image tag, usually built
locally from the branch under test) in addition to the manifest env var.
"""

import base64
import io
import os
import subprocess
import time

import numpy as np
import pytest
import requests
from PIL import Image

from . import harness
from .conftest import collect_image_block

HOST_PORT = int(os.environ.get("ORIENT_EXPRESS_TEST_DOCKER_PORT", "18093"))
ADC_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


def _docker_cases():
    if not harness.manifest_location():
        return []
    manifest = harness.load_manifest()
    return sorted(
        case for case, cfg in manifest["cases"].items() if cfg.get("docker", False)
    )


def _start_container(image: str, manifest: dict, case: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            f"{ADC_PATH}:/root/.config/gcloud/application_default_credentials.json:ro",
            "-e",
            "GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json",
            "-e",
            f"GOOGLE_CLOUD_PROJECT={os.environ.get('GOOGLE_CLOUD_PROJECT', '')}",
            "-e",
            f"AIP_STORAGE_URI={manifest['gcs_prefix']}/cases/{case}/model/",
            "-e",
            f"MODEL_NAME={case}",
            "-p",
            f"{HOST_PORT}:8080",
            image,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _wait_ready(case: str, container_id: str, timeout_s: int = 300):
    deadline = time.time() + timeout_s
    url = f"http://localhost:{HOST_PORT}/v1/models/{case}"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        alive = (
            subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
            ).returncode
            == 0
        )
        if not alive:
            raise RuntimeError(f"container for '{case}' exited during startup")
        time.sleep(3)
    raise TimeoutError(f"server for '{case}' not ready within {timeout_s}s")


def _mask_string(key: str) -> bool:
    return key in ("class_mask", "valid_mask")


def _png_mismatch(a_b64: str, b_b64: str) -> float:
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(a_b64))))
    b = np.asarray(Image.open(io.BytesIO(base64.b64decode(b_b64))))
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(a != b))


def _bbox_dict_iou(a, b) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_detection_list(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, dict) and "bbox" in v and "class" in v for v in value)
    )


def _align_detections(golden_list, got_list):
    """Reorder got_list to match golden_list (greedy class + IoU >= 0.5).

    Detection order is not part of the serving contract: near-tied scores
    swap TopK order across onnxruntime versions. Unmatched leftovers are
    appended so count mismatches still surface.
    """
    aligned = []
    used = set()
    for g in golden_list:
        best, best_iou = None, 0.5
        for j, a in enumerate(got_list):
            if j in used or a.get("class") != g.get("class"):
                continue
            iou = _bbox_dict_iou(g["bbox"], a["bbox"])
            if iou >= best_iou:
                best, best_iou = j, iou
        if best is not None:
            used.add(best)
            aligned.append(got_list[best])
    aligned.extend(a for j, a in enumerate(got_list) if j not in used)
    return aligned


def _rec(records, path, golden, got, ok, detail=""):
    records.append(
        {"field": path, "golden": golden, "got": got, "ok": bool(ok), "detail": detail}
    )


def compare_response(got, golden, tol, path=""):
    """Generic tolerant comparison of response JSON; structured records."""
    records = []
    if isinstance(golden, dict):
        if not isinstance(got, dict):
            _rec(records, path, "object", type(got).__name__, False)
            return records
        for key, expected in golden.items():
            if key == "debug_image":
                continue
            if key not in got:
                _rec(records, f"{path}.{key}", "present", "missing", False)
                continue
            if _mask_string(key) and isinstance(expected, str):
                mismatch = _png_mismatch(got[key], expected)
                _rec(
                    records,
                    f"{path}.{key}",
                    "—",
                    f"pixel mismatch {mismatch:.2e}",
                    mismatch <= tol["pixel_mismatch_max"],
                    f"(max {tol['pixel_mismatch_max']})",
                )
            else:
                got_value = got[key]
                if _is_detection_list(expected) and _is_detection_list(got_value):
                    got_value = _align_detections(expected, got_value)
                records.extend(
                    compare_response(got_value, expected, tol, f"{path}.{key}")
                )
        for key in got:
            if key != "debug_image" and key not in golden:
                _rec(records, f"{path}.{key}", "absent", "present", False)
    elif isinstance(golden, list):
        if not isinstance(got, list) or len(got) != len(golden):
            got_len = len(got) if isinstance(got, list) else "?"
            _rec(records, f"{path} length", len(golden), got_len, False)
            return records
        for i, (a, g) in enumerate(zip(got, golden, strict=True)):
            records.extend(compare_response(a, g, tol, f"{path}[{i}]"))
    elif isinstance(golden, bool) or golden is None or isinstance(golden, str):
        _rec(records, path, golden, got, got == golden)
    elif isinstance(golden, (int, float)):
        atol = (
            tol["bbox_atol"]
            if path.endswith(("x1", "y1", "x2", "y2"))
            else tol["score_atol"]
        )
        delta = abs(float(got) - float(golden))
        _rec(
            records,
            path,
            round(float(golden), 5),
            round(float(got), 5),
            delta <= atol,
            f"Δ{delta:.2e} (tol {atol})",
        )
    return records


@pytest.mark.parametrize("case", _docker_cases())
def test_docker_case_matches_golden(manifest, docker_image, case):
    golden = harness.fetch_golden(manifest, f"{case}.docker")
    if golden is None:
        pytest.fail(f"no docker golden for '{case}'")
    tolerances = harness.case_tolerances(manifest, manifest["cases"][case])

    case_dir = harness.fetch_case_assets(manifest, case)
    images_dir = os.path.join(case_dir, "images")
    names = sorted(os.listdir(images_dir))

    container_id = _start_container(docker_image, manifest, case)
    try:
        _wait_ready(case, container_id)
        failures = []
        for variant in harness.case_variants(manifest["cases"][case]):
            golden_variant = golden["variants"].get(variant["name"])
            if golden_variant is None:
                failures.append(f"[{variant['name']}] missing from docker golden")
                continue
            for name in names:
                # one image per request: fixed-batch-1 graphs + production shape
                with open(os.path.join(images_dir, name), "rb") as f:
                    instance = {"image": base64.b64encode(f.read()).decode()}
                response = requests.post(
                    f"http://localhost:{HOST_PORT}/v1/models/{case}:predict",
                    json={
                        "instances": [instance],
                        "parameters": variant.get("predict_params", {}),
                    },
                    timeout=600,
                )
                response.raise_for_status()
                prediction = response.json()["predictions"][0]
                after_b64 = prediction.pop("debug_image", None)
                records = compare_response(prediction, golden_variant[name], tolerances)
                before = None
                golden_annotated = harness.golden_annotated_dir(
                    manifest, case, kind="docker-annotated"
                )
                if golden_annotated:
                    path = os.path.join(
                        golden_annotated, variant["name"], f"{name}.jpg"
                    )
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            before = f.read()
                collect_image_block(
                    f"{case} — serving (docker)",
                    f"{name} / {variant['name']}",
                    records,
                    before=before,
                    after=base64.b64decode(after_b64) if after_b64 else None,
                    captions=("before (golden server)", "after (current server)"),
                )
                failures.extend(
                    f"[{variant['name']}] {name}{r['field']}: "
                    f"golden={r['golden']} got={r['got']} {r['detail']}"
                    for r in records
                    if not r["ok"]
                )
        assert not failures, "\n".join(failures)
    finally:
        subprocess.run(["docker", "stop", container_id], capture_output=True)

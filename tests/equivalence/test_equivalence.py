"""Predictor-level golden equivalence: get_predictor -> predict -> outputs.

Each test also feeds the end-of-session HTML report (see conftest) with
before/after annotated images and per-field diff records.
"""

import os

import pytest

from . import harness
from .conftest import collect_image_block


def _cases():
    if not harness.manifest_location():
        return []
    manifest = harness.load_manifest()
    return sorted(manifest["cases"].keys())


def evaluate_case(manifest, case, annotate=True):
    """Run one case on the current code.

    Returns (got_by_variant, annotated_by_variant, golden).
    """
    case_cfg = manifest["cases"][case]
    case_dir = harness.fetch_case_assets(manifest, case)
    golden = harness.fetch_golden(manifest, case)

    if "query_case" in case_cfg:  # vector index, chained from embeddings
        query_dir = harness.fetch_case_assets(manifest, case_cfg["query_case"])
        query_predictor = harness.load_case_predictor(query_dir)
        query_outputs = harness.extract_outputs(
            query_predictor,
            harness.case_model_type(query_dir),
            harness.load_case_images(query_dir),
            {},
        )
        embeddings = {k: v["embedding"] for k, v in query_outputs.items()}
        index = harness.load_case_predictor(case_dir)
        got = harness.run_vector_index_case(index, embeddings, case_cfg.get("top_k", 5))
        return {"default": got}, {"default": {}}, golden

    predictor = harness.load_case_predictor(case_dir)
    model_type = harness.case_model_type(case_dir)
    images = harness.load_case_images(case_dir)
    assert images, f"case '{case}' has no images"

    got_by_variant = {}
    annotated_by_variant = {}
    for variant in harness.case_variants(case_cfg):
        outputs, annotated = harness.evaluate_variant(
            predictor,
            model_type,
            images,
            variant.get("predict_params", {}),
            annotate=annotate,
        )
        got_by_variant[variant["name"]] = outputs
        annotated_by_variant[variant["name"]] = annotated
    return got_by_variant, annotated_by_variant, golden


def _golden_annotated(manifest, case, variant_name, image_name, kind="annotated"):
    directory = harness.golden_annotated_dir(manifest, case, kind=kind)
    if directory is None:
        return None
    path = os.path.join(directory, variant_name, f"{image_name}.jpg")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("case", _cases())
def test_case_matches_golden(manifest, case):
    got_by_variant, annotated_by_variant, golden = evaluate_case(manifest, case)
    if golden is None:
        pytest.fail(
            f"no golden for case '{case}' — generate with generate_goldens.py, "
            "review, and upload manually"
        )
    tolerances = harness.case_tolerances(manifest, manifest["cases"][case])

    all_failures = []
    for variant_name, got in sorted(got_by_variant.items()):
        golden_variant = golden["variants"].get(variant_name)
        if golden_variant is None:
            all_failures.append(f"[{variant_name}] missing from golden file")
            continue
        for image_name in sorted(golden_variant):
            records = harness.compare_outputs(
                {image_name: got.get(image_name, {})},
                {image_name: golden_variant[image_name]},
                tolerances,
            )
            collect_image_block(
                case,
                f"{image_name} / {variant_name}",
                records,
                before=_golden_annotated(manifest, case, variant_name, image_name),
                after=annotated_by_variant[variant_name].get(image_name),
            )
            all_failures.extend(
                f"[{variant_name}] {failure}" for failure in harness.failures(records)
            )
    assert not all_failures, "\n".join(all_failures)

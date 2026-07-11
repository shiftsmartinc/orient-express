"""Generate golden files for the equivalence suite.

Writes ONLY to a local directory — uploading goldens to GCS is a deliberate,
manually-reviewed act (there is no automated regeneration):

    python tests/equivalence/generate_goldens.py --out /tmp/goldens
    # review the JSON + annotated images, then:
    # gcloud storage cp -r /tmp/goldens/* <prefix>/goldens/

Alongside each <case>.json this writes <case>/annotated/<variant>/<image>.jpg
(the golden code's rendering of its own predictions) for the HTML report's
before/after view.

Serving-level goldens are captured from a running container (boot it
yourself against the case's model dir):

    python tests/equivalence/generate_goldens.py --out /tmp/goldens \
        --case <case> --docker-url http://localhost:8080

Requests are sent one image per call (fixed-batch-1 graphs; matches
production traffic shape).
"""

import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/

from equivalence import harness  # noqa: E402


def generate_predictor_golden(manifest, case, out_dir):
    case_cfg = manifest["cases"][case]
    case_dir = harness.fetch_case_assets(manifest, case)

    if "query_case" in case_cfg:
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
        outputs = harness.run_vector_index_case(
            index, embeddings, case_cfg.get("top_k", 5)
        )
        return {
            "case": case,
            "generated_by": harness.provenance(),
            "variants": {"default": outputs},
        }

    predictor = harness.load_case_predictor(case_dir)
    model_type = harness.case_model_type(case_dir)
    images = harness.load_case_images(case_dir)
    variants = {}
    for variant in harness.case_variants(case_cfg):
        params = variant.get("predict_params", {})
        variants[variant["name"]] = harness.extract_outputs(
            predictor, model_type, images, params
        )
        annotated = harness.render_annotated(predictor, model_type, images, params)
        for image_name, jpeg in annotated.items():
            if jpeg is None:
                continue
            path = os.path.join(
                out_dir, case, "annotated", variant["name"], f"{image_name}.jpg"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(jpeg)
    return {"case": case, "generated_by": harness.provenance(), "variants": variants}


def generate_docker_golden(manifest, case, server_url, out_dir):
    import requests

    case_cfg = manifest["cases"][case]
    case_dir = harness.fetch_case_assets(manifest, case)
    images_dir = os.path.join(case_dir, "images")

    variants = {}
    for variant in harness.case_variants(case_cfg):
        by_image = {}
        for name in sorted(os.listdir(images_dir)):
            with open(os.path.join(images_dir, name), "rb") as f:
                instance = {"image": base64.b64encode(f.read()).decode()}
            payload = {
                "instances": [instance],
                "parameters": variant.get("predict_params", {}),
            }
            response = requests.post(
                f"{server_url}/v1/models/{case}:predict", json=payload, timeout=600
            )
            response.raise_for_status()
            prediction = response.json()["predictions"][0]
            # stripped from the JSON (huge), but kept as sibling files for the
            # HTML report's before/after view
            debug_b64 = prediction.pop("debug_image", None)
            if debug_b64:
                path = os.path.join(
                    out_dir, case, "docker-annotated", variant["name"], f"{name}.jpg"
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(debug_b64))
            by_image[name] = prediction
        variants[variant["name"]] = by_image
    return {"case": case, "generated_by": harness.provenance(), "variants": variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="local output directory")
    parser.add_argument("--case", action="append", help="subset of cases (default all)")
    parser.add_argument("--docker-url", help="capture serving goldens from this URL")
    args = parser.parse_args()

    manifest = harness.load_manifest()
    cases = args.case or sorted(manifest["cases"].keys())
    os.makedirs(args.out, exist_ok=True)

    for case in cases:
        if args.docker_url:
            if not manifest["cases"][case].get("docker", False):
                print(f"{case}: docker disabled in manifest, skipping")
                continue
            golden = generate_docker_golden(manifest, case, args.docker_url, args.out)
            path = os.path.join(args.out, f"{case}.docker.json")
        else:
            golden = generate_predictor_golden(manifest, case, args.out)
            path = os.path.join(args.out, f"{case}.json")
        with open(path, "w") as f:
            json.dump(golden, f, indent=1, sort_keys=True)
        print(
            f"{case}: wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB, "
            f"digest {harness.stable_digest(golden['variants'])})"
        )

    print("\nReview the files, then upload manually:")
    print(f"  gcloud storage cp -r {args.out}/* {manifest['gcs_prefix']}/goldens/")


if __name__ == "__main__":
    main()

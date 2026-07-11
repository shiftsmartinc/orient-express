# Golden equivalence suite

Runs real models on real photos and compares outputs against reviewed golden
files, so any change that shifts model outputs is caught before merge — at
two levels:

- **Predictor level** (`test_equivalence.py`): the library in-process —
  `get_predictor(model_dir)` → `predict(images)` → extracted outputs.
- **Serving level** (`test_docker_equivalence.py`, opt-in): boots the actual
  Docker serving image per case and checks the HTTP responses.

Models, photos, and goldens are **private** — they live in a GCS prefix,
never in this repo, and this code contains no references to specific models.
The suite auto-skips unless configured, so public CI and external
contributors never see it run.

**Every run writes an HTML report** to `test-output/equivalence_report.html`:
before/after annotated images per photo plus a per-field diff table of every
comparison, including within-tolerance deltas (yellow rows). A test fails if
and only if its image shows DIFFERS in the report. Always inspect the report
before merging. It embeds internal photos — treat it as private; never
publish or attach it anywhere public.

## Environment variables

| variable | required | meaning |
|---|---|---|
| `ORIENT_EXPRESS_TEST_MANIFEST` | yes | `gs://…/manifest.yaml` (or a local path). Without it, every test skips. |
| `ORIENT_EXPRESS_TEST_DOCKER_IMAGE` | for serving tests | image tag to boot (usually built locally from your branch). |
| `GOOGLE_CLOUD_PROJECT` | for serving tests | passed into the container so its GCS client can resolve a project (production Vertex provides this itself). |
| `ORIENT_EXPRESS_TEST_GOLDENS_DIR` | no | local goldens dir instead of GCS — for evaluating **candidate** goldens before uploading. |
| `ORIENT_EXPRESS_TEST_REPORT` | no | report output path (default `test-output/equivalence_report.html`). |
| `ORIENT_EXPRESS_TEST_CACHE` | no | asset cache dir (default `~/.cache/orient-express-test-assets`). |

GCP application-default credentials with read access to the prefix are
required (`gcloud auth application-default login`).

## Running (uv)

```bash
export ORIENT_EXPRESS_TEST_MANIFEST=gs://<internal-prefix>/manifest.yaml

make equivalence          # predictor level + HTML report
make equivalence-docker   # builds the local serving image, then both levels
```

Or directly: `uv run pytest tests/equivalence -v`. First run downloads the
model artifacts (~100 MB per case) into the local cache; later runs reuse it.

## Testing a feature branch — the full flow

```bash
make install                  # uv sync
make lint                     # ruff check + format check (same as CI)
make test                     # unit tests
export ORIENT_EXPRESS_TEST_MANIFEST=gs://<internal-prefix>/manifest.yaml
make equivalence              # golden gate + report
# if the branch touches serving code, Dockerfiles, or dependencies:
make equivalence-docker
# then LOOK at test-output/equivalence_report.html before merging
```

## Golden lifecycle

Goldens are the enshrined reference outputs. There is **no automated
regeneration** — creating or replacing a golden is a deliberate, reviewed,
manual act.

> **No-take-back warning:** uploading goldens overwrites the previous
> reference in place — there is no versioning and no undo. From that moment,
> every branch is judged against the new goldens, and the old baseline is
> gone unless you kept the local copy. Only upload after you have reviewed
> the report and the golden files themselves, and keep the local directory
> until you're certain.

### 1. Generate candidates (locally only)

```bash
# predictor level (writes <case>.json + <case>/annotated/ images):
uv run python tests/equivalence/generate_goldens.py --out /tmp/candidate-goldens

# serving level (writes <case>.docker.json + <case>/docker-annotated/):
# boot the reference server image against a case's model dir, then:
uv run python tests/equivalence/generate_goldens.py --out /tmp/candidate-goldens \
    --case <case> --docker-url http://localhost:8080
```

### 2. Review the candidates

Run the suite against them and inspect the report — this shows exactly what
would change relative to your current code:

```bash
ORIENT_EXPRESS_TEST_GOLDENS_DIR=/tmp/candidate-goldens uv run pytest tests/equivalence -v
```

Also open the golden JSONs and the `annotated/` JPEGs directly — they are
the reference you're about to enshrine.

### 3. Upload (the point of no return)

```bash
gcloud storage cp -r /tmp/candidate-goldens/* gs://<internal-prefix>/goldens/
```

### 4. Verify from GCS

```bash
unset ORIENT_EXPRESS_TEST_GOLDENS_DIR
uv run pytest tests/equivalence -v
```

## Tolerances

Comparison is tolerance-based, not bit-exact — BLAS/onnxruntime/pillow
versions and OSes introduce float-level wiggle (score jitter across
onnxruntime session boots and image dependency bumps is real and measured).
Defaults live in `harness.DEFAULT_TOLERANCES`; the manifest can override per
case, with comments documenting why. Class/argmax equality is always strict.
Detection lists are matched order-independently (class + IoU): near-tied
scores legitimately swap TopK order across onnxruntime versions.

## Manifest

See `manifest.example.yaml` for the structure. The real manifest lives next
to the assets in GCS and defines the cases, predict-parameter variants,
docker enablement, and tolerance overrides.

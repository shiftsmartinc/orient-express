"""Fixtures for the golden equivalence suite.

Everything here auto-skips unless ORIENT_EXPRESS_TEST_MANIFEST is set, so the
suite is invisible to normal test runs, public CI, and external contributors.

Every pytest run that executes at least one equivalence test writes an HTML
report (before/after images + per-field diff tables) — by default to
test-output/equivalence_report.html, overridable via
ORIENT_EXPRESS_TEST_REPORT.
"""

import os

import pytest

from . import harness, htmlreport

DEFAULT_REPORT_PATH = os.path.join("test-output", "equivalence_report.html")

# case name -> {"blocks": [(title, html, failed)], "level": ...} accumulated
# across tests, rendered at session end
_collected: dict[str, list] = {}


def pytest_collection_modifyitems(config, items):
    if harness.manifest_location():
        return
    skip = pytest.mark.skip(
        reason=f"{harness.MANIFEST_ENV} not set (private equivalence suite)"
    )
    for item in items:
        if "equivalence" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def manifest():
    return harness.load_manifest()


@pytest.fixture(scope="session")
def docker_image():
    image = os.environ.get(harness.DOCKER_IMAGE_ENV)
    if not image:
        pytest.skip(f"{harness.DOCKER_IMAGE_ENV} not set")
    return image


def collect_image_block(
    section: str, title, records, before=None, after=None, captions=None
):
    """Called by tests to feed the end-of-session HTML report."""
    block_html, failed = htmlreport.image_block(title, records, before, after, captions)
    _collected.setdefault(section, []).append((block_html, failed))


def pytest_sessionfinish(session, exitstatus):
    if not _collected:
        return
    sections = []
    for section_name in sorted(_collected):
        blocks = _collected[section_name]
        fail = sum(1 for _, failed in blocks if failed)
        ok = len(blocks) - fail
        section_html = f"<h2>{section_name}</h2>" + "".join(b for b, _ in blocks)
        sections.append((section_name, section_html, ok, fail))
    page = htmlreport.build_page(sections, harness.provenance())
    path = os.environ.get(harness.REPORT_ENV, DEFAULT_REPORT_PATH)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(page)
    print(f"\nequivalence report: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    _collected.clear()

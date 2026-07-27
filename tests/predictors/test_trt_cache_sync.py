"""Tests for the TRT engine cache GCS sync (mocked GCS)."""

import time
from threading import Event
from unittest.mock import ANY, MagicMock, patch

from orient_express.predictors.runtime import _TrtCacheGcsSync

SCOPE = "abc123-ort1.27.0-trt10.16/fp32"  # what create_session appends


def make_sync(tmp_path):
    return _TrtCacheGcsSync(f"gs://bucket/trt-cache/{SCOPE}", str(tmp_path))


def test_upload_new_pushes_only_new_files(tmp_path):
    sync = make_sync(tmp_path)
    with patch.object(sync, "_gs") as gs:
        sync.upload_new()  # nothing to upload
        gs.upload_file.assert_not_called()

        (tmp_path / "graph_sm89.engine").write_bytes(b"engine")
        sync.upload_new()
        gs.upload_file.assert_called_once_with(
            str(tmp_path / "graph_sm89.engine"),
            f"gs://bucket/trt-cache/{SCOPE}/graph_sm89.engine",
            timeout=sync._timeout,
            retry=ANY,
            chunk_size=sync._UPLOAD_CHUNK_BYTES,
        )

        gs.upload_file.reset_mock()
        sync.upload_new()  # already synced, dir unchanged
        gs.upload_file.assert_not_called()

        # ORT rewrites cache files on every session load without changing
        # content; identical bytes with a fresh mtime must NOT re-upload
        time.sleep(0.01)
        (tmp_path / "graph_sm89.engine").write_bytes(b"engine")
        sync.upload_new()
        gs.upload_file.assert_not_called()


def test_upload_reuploads_changed_files(tmp_path):
    sync = make_sync(tmp_path)
    with patch.object(sync, "_gs") as gs:
        f = tmp_path / "cache_sm89.timing"
        f.write_bytes(b"v1")
        sync.upload_new()
        time.sleep(0.01)
        f.write_bytes(b"v2, rebuilt")
        f.touch()
        sync.upload_new()
        assert gs.upload_file.call_count == 2


def test_upload_errors_do_not_raise(tmp_path):
    sync = make_sync(tmp_path)
    with patch.object(sync, "_gs") as gs:
        gs.upload_file.side_effect = OSError("no network")
        (tmp_path / "x.engine").write_bytes(b"engine")
        sync.upload_new()  # logs a warning, never raises


def test_upload_failure_does_not_abandon_sweep(tmp_path):
    # per-file try: a failed (e.g. too-slow) engine upload must not starve
    # the profile/timing files behind it, and gets retried next sweep
    sync = make_sync(tmp_path)
    with patch.object(sync, "_gs") as gs:
        (tmp_path / "a.engine").write_bytes(b"a")
        (tmp_path / "b.timing").write_bytes(b"b")
        gs.upload_file.side_effect = [OSError("timeout"), None]
        sync.upload_new()
        assert gs.upload_file.call_count == 2  # second file still attempted

        gs.upload_file.reset_mock()
        gs.upload_file.side_effect = None
        sync.upload_new()  # only the failed file is retried
        assert gs.upload_file.call_count == 1


def _gcs_mock(blob_names):
    blobs = []
    for name in blob_names:
        blob = MagicMock()
        blob.name = name
        blob.download_to_filename.side_effect = lambda path, **kw: open(
            path, "wb"
        ).write(b"engine")
        blobs.append(blob)
    bucket = MagicMock()
    bucket.list_blobs.return_value = blobs
    client = MagicMock()
    client.bucket.return_value = bucket
    return client, bucket, blobs


def test_download_populates_missing_files(tmp_path):
    sync = make_sync(tmp_path)
    client, bucket, blobs = _gcs_mock([f"trt-cache/{SCOPE}/graph_sm89.engine"])

    with (
        patch("google.cloud.storage.Client", return_value=client),
        patch("orient_express.predictors.runtime._local_sm_tags", return_value=None),
    ):
        sync.download()

    assert (tmp_path / "graph_sm89.engine").read_bytes() == b"engine"
    # every GCS call is capped by the timeout INCLUDING retries — the
    # client's default policy would otherwise retry an outage for 120s
    # of cold-start stall
    _, list_kwargs = bucket.list_blobs.call_args
    assert list_kwargs["timeout"] == sync._timeout
    assert list_kwargs["retry"]._timeout == sync._timeout
    _, dl_kwargs = blobs[0].download_to_filename.call_args
    assert dl_kwargs["timeout"] == sync._timeout
    assert dl_kwargs["retry"]._timeout == sync._timeout


def test_download_skips_other_gpu_architectures(tmp_path):
    # ORT keys engine/timing filenames by SM arch; another generation's
    # engine can never load here, so it must not cost cold-start bandwidth.
    # Untagged files and unknown local SM always download (safe fallback).
    sync = make_sync(tmp_path)
    client, bucket, blobs = _gcs_mock(
        [
            f"trt-cache/{SCOPE}/graph_sm89.engine",
            f"trt-cache/{SCOPE}/graph_sm120.engine",
            f"trt-cache/{SCOPE}/cache_sm89.timing",
            f"trt-cache/{SCOPE}/untagged.txt",
        ]
    )

    with (
        patch("google.cloud.storage.Client", return_value=client),
        patch(
            "orient_express.predictors.runtime._local_sm_tags",
            return_value={"sm120"},
        ),
    ):
        sync.download()

    downloaded = sorted(p.name for p in tmp_path.iterdir())
    assert downloaded == ["graph_sm120.engine", "untagged.txt"]


def test_schedule_upload_runs_on_daemon_worker(tmp_path):
    sync = make_sync(tmp_path)
    done = Event()
    with patch.object(sync, "upload_new", side_effect=done.set):
        sync.schedule_upload()
        assert done.wait(5)
    assert sync._worker.daemon  # a hung upload must never delay process exit


def test_schedule_upload_coalesces_pending_wakes(tmp_path):
    sync = make_sync(tmp_path)
    gate = Event()
    calls = []

    def sweep():
        calls.append(1)
        gate.wait(5)

    with patch.object(sync, "upload_new", side_effect=sweep):
        sync.schedule_upload()
        deadline = time.time() + 5
        while not calls and time.time() < deadline:  # first sweep is running
            time.sleep(0.01)
        sync.schedule_upload()
        sync.schedule_upload()
        sync.schedule_upload()  # all pending wakes collapse into ONE sweep
        gate.set()
        deadline = time.time() + 5
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)  # would-be third sweep gets a chance to appear
        assert len(calls) == 2


def test_create_session_schedules_upload_for_eager_builds(tmp_path, monkeypatch):
    # explicit profiles (mandatory for TRT) build the engine during session
    # init, not first predict — the sweep must be scheduled at creation so a
    # worker that dies before predicting still populates the shared cache
    from orient_express.predictors import runtime

    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_GCS", "gs://bucket/prefix")
    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    session = MagicMock()
    session.get_providers.return_value = ["TensorrtExecutionProvider"]
    sync = MagicMock()
    profile = {
        "trt_profile_min_shapes": "images:1x64x64x3",
        "trt_profile_opt_shapes": "images:1x64x64x3",
        "trt_profile_max_shapes": "images:1x64x64x3",
    }
    with (
        patch.object(runtime.ort, "InferenceSession", return_value=session),
        patch.object(runtime, "_TrtCacheGcsSync", return_value=sync),
        patch.object(runtime, "_preload_gpu_dlls"),
        patch.object(runtime, "_preload_tensorrt_libs"),
    ):
        _, returned = runtime.create_session(str(model), "tensorrt", profile)
    assert returned is sync
    sync.download.assert_called_once()
    sync.schedule_upload.assert_called_once()


def test_upload_timeout_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_TIMEOUT", "7.5")
    assert make_sync(tmp_path)._timeout == 7.5


def test_cache_gc_evicts_lru_scopes(tmp_path, monkeypatch):
    import os

    from orient_express.predictors.runtime import trt_engine_cache_dir

    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES", "1000")

    oldest = tmp_path / "aaaaaaaaaaaaaaaa-ort1-trt1" / "fp32"
    oldest.mkdir(parents=True)
    (oldest / "x.engine").write_bytes(b"x" * 600)
    os.utime(oldest / "x.engine", (500, 500))
    newer = tmp_path / "bbbbbbbbbbbbbbbb-ort1-trt1" / "fp32"
    newer.mkdir(parents=True)
    (newer / "y.engine").write_bytes(b"y" * 600)
    os.utime(newer / "y.engine", (1000, 1000))

    # resolving a scope prunes: 1200 > 1000, oldest goes first, then under cap
    current = trt_engine_cache_dir("cccccccccccccccc-ort1-trt1/fp32")
    assert not oldest.exists()
    assert newer.exists()

    # the in-use scope is never evicted, however old it looks
    (tmp_path / "cccccccccccccccc-ort1-trt1" / "fp32" / "z.engine").write_bytes(
        b"z" * 600
    )
    os.utime(tmp_path / "cccccccccccccccc-ort1-trt1" / "fp32" / "z.engine", (100, 100))
    trt_engine_cache_dir("cccccccccccccccc-ort1-trt1/fp32")
    assert (tmp_path / "cccccccccccccccc-ort1-trt1" / "fp32" / "z.engine").exists()
    assert not newer.exists()  # 1200 > 1000 again; the other scope went

    # 0 disables GC entirely
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES", "0")
    big = tmp_path / "dddddddddddddddd-ort1-trt1" / "fp32"
    big.mkdir(parents=True)
    (big / "w.engine").write_bytes(b"w" * 5000)
    trt_engine_cache_dir("cccccccccccccccc-ort1-trt1/fp32")
    assert big.exists()

    assert current == str(tmp_path / "cccccccccccccccc-ort1-trt1" / "fp32")


def test_trt_cache_scope_keys(tmp_path):
    from orient_express.predictors.runtime import trt_cache_scope

    model_a = tmp_path / "a.onnx"
    model_a.write_bytes(b"weights-a")
    model_b = tmp_path / "b.onnx"
    model_b.write_bytes(b"weights-b")
    profile = {"trt_profile_min_shapes": "images:1x64x64x3"}

    base = trt_cache_scope(str(model_a), None, fp16=False)
    assert base == trt_cache_scope(str(model_a), None, fp16=False)  # stable
    assert base.endswith("/fp32")
    assert base != trt_cache_scope(str(model_b), None, fp16=False)  # model
    assert base != trt_cache_scope(str(model_a), profile, fp16=False)  # profile
    assert base != trt_cache_scope(str(model_a), None, fp16=True)  # precision

    # cache plumbing options never split the scope...
    plumbing = {"trt_engine_cache_path": "/x", "trt_timing_cache_enable": True}
    assert base == trt_cache_scope(str(model_a), plumbing, fp16=False)
    # ...but any other option is assumed to change the compiled engine
    fallback = {"trt_layer_norm_fp32_fallback": True}
    assert base != trt_cache_scope(str(model_a), fallback, fp16=False)
    assert trt_cache_scope(str(model_a), fallback, fp16=False) != trt_cache_scope(
        str(model_a), {"trt_builder_optimization_level": 5}, fp16=False
    )


def test_trt_cache_scope_keys_tf32_env(tmp_path, monkeypatch):
    # TRT bakes TF32 tactics into engines and a mismatched cached engine
    # hard-fails to load, so the override must split the scope
    from orient_express.predictors.runtime import trt_cache_scope

    model = tmp_path / "a.onnx"
    model.write_bytes(b"weights")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    base = trt_cache_scope(str(model), None, fp16=False)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    assert base != trt_cache_scope(str(model), None, fp16=False)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    assert trt_cache_scope(str(model), None, fp16=False) != base


def test_cache_gc_never_touches_foreign_content(tmp_path, monkeypatch):
    # eviction candidates are recognized by the exact scope-dir shape this
    # library mints; anything else under a (possibly shared) cache root —
    # loose files, foreign dirs, the root itself — must survive GC
    import os

    from orient_express.predictors.runtime import trt_engine_cache_dir

    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES", "1000")

    loose = tmp_path / "notes.txt"
    loose.write_bytes(b"n" * 5000)  # over cap on its own, but not ours
    os.utime(loose, (10, 10))
    foreign = tmp_path / "some-user-dir"
    foreign.mkdir()
    (foreign / "data.bin").write_bytes(b"d" * 5000)
    os.utime(foreign / "data.bin", (10, 10))

    evictable = tmp_path / ("e" * 16 + "-ort1-trt1") / "fp32"
    evictable.mkdir(parents=True)
    (evictable / "x.engine").write_bytes(b"x" * 600)
    os.utime(evictable / "x.engine", (500, 500))
    other = tmp_path / ("f" * 16 + "-ort1-trt1") / "fp32"
    other.mkdir(parents=True)
    (other / "y.engine").write_bytes(b"y" * 600)
    os.utime(other / "y.engine", (1000, 1000))

    trt_engine_cache_dir("0" * 16 + "-ort1-trt1/fp32")

    assert tmp_path.exists()
    assert loose.exists()
    assert (foreign / "data.bin").exists()
    # scope-shaped dirs still obey the LRU cap (1200 > 1000)
    assert not evictable.exists()
    assert other.exists()


def test_download_is_atomic(tmp_path):
    # a worker killed mid-download must not leave a truncated file at the
    # final path: it would be trusted forever and re-uploaded over the good
    # GCS copy. Downloads land in a .part temp and are renamed on success.
    sync = make_sync(tmp_path)
    client, bucket, blobs = _gcs_mock([f"trt-cache/{SCOPE}/graph_sm89.engine"])
    paths_used = []

    def record_path(path, **kw):
        paths_used.append(path)
        open(path, "wb").write(b"engine")

    blobs[0].download_to_filename.side_effect = record_path
    with (
        patch("google.cloud.storage.Client", return_value=client),
        patch("orient_express.predictors.runtime._local_sm_tags", return_value=None),
    ):
        sync.download()

    assert paths_used == [str(tmp_path / "graph_sm89.engine.part")]
    assert (tmp_path / "graph_sm89.engine").read_bytes() == b"engine"
    assert not (tmp_path / "graph_sm89.engine.part").exists()


def test_interrupted_download_leaves_no_final_file(tmp_path):
    sync = make_sync(tmp_path)
    client, bucket, blobs = _gcs_mock([f"trt-cache/{SCOPE}/graph_sm89.engine"])

    def die_midway(path, **kw):
        open(path, "wb").write(b"eng")  # partial bytes
        raise TimeoutError("connection lost")

    blobs[0].download_to_filename.side_effect = die_midway
    with (
        patch("google.cloud.storage.Client", return_value=client),
        patch("orient_express.predictors.runtime._local_sm_tags", return_value=None),
    ):
        sync.download()  # best-effort: logs, doesn't raise

    assert not (tmp_path / "graph_sm89.engine").exists()  # nothing to trust
    # and the leftover temp is never swept up to GCS
    with patch.object(sync, "_gs") as gs:
        sync.upload_new()
        gs.upload_file.assert_not_called()


_TEST_PROFILE = {
    "trt_profile_min_shapes": "images:1x64x64x3",
    "trt_profile_opt_shapes": "images:1x64x64x3",
    "trt_profile_max_shapes": "images:1x64x64x3",
}


def _create_session_with_failures(tmp_path, monkeypatch, failures, error):
    """create_session against a mock ORT that fails `failures` times."""
    from orient_express.predictors import runtime

    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ORIENT_EXPRESS_TRT_CACHE_GCS", raising=False)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    good = MagicMock()
    good.get_providers.return_value = ["TensorrtExecutionProvider"]
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return good

    with (
        patch.object(runtime.ort, "InferenceSession", side_effect=flaky),
        patch.object(runtime, "_preload_gpu_dlls"),
        patch.object(runtime, "_preload_tensorrt_libs"),
    ):
        session, _ = runtime.create_session(str(model), "tensorrt", _TEST_PROFILE)
    return session, good, calls


def test_corrupt_engine_cache_rebuilds_once(tmp_path, monkeypatch):
    # ORT hard-fails on a corrupt cached engine and never rebuilds on its
    # own; create_session clears the scope's cache and retries exactly once
    from orient_express.predictors.runtime import trt_cache_scope

    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    scope = trt_cache_scope(str(model), _TEST_PROFILE, fp16=False)
    cache_dir = tmp_path / scope
    cache_dir.mkdir(parents=True)
    (cache_dir / "graph_sm89.engine").write_bytes(b"torn")

    error = RuntimeError(
        "TensorRT EP could not deserialize engine from cache: graph_sm89.engine"
    )
    session, good, calls = _create_session_with_failures(
        tmp_path, monkeypatch, failures=1, error=error
    )
    assert session is good
    assert len(calls) == 2
    assert not (cache_dir / "graph_sm89.engine").exists()  # cleared


def test_corrupt_engine_cache_fails_for_real_on_second_failure(tmp_path, monkeypatch):
    import pytest

    error = RuntimeError("TensorRT EP could not deserialize engine from cache: x")
    with pytest.raises(RuntimeError, match="deserialize engine"):
        _create_session_with_failures(tmp_path, monkeypatch, failures=2, error=error)


def test_non_cache_session_errors_do_not_retry(tmp_path, monkeypatch):
    import pytest

    error = RuntimeError("TensorRT EP failed: unsupported op FooBar")
    with pytest.raises(RuntimeError, match="FooBar"):
        _, _, calls = _create_session_with_failures(
            tmp_path, monkeypatch, failures=2, error=error
        )

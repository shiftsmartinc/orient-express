"""Tests for the TRT engine cache GCS sync (mocked GCS)."""

import time
from threading import Event
from unittest.mock import MagicMock, patch

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
        )

        gs.upload_file.reset_mock()
        sync.upload_new()  # already synced, dir unchanged
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


def test_download_populates_missing_files(tmp_path):
    sync = make_sync(tmp_path)

    blob = MagicMock()
    blob.name = f"trt-cache/{SCOPE}/graph_sm89.engine"
    blob.download_to_filename.side_effect = lambda path, **kw: open(path, "wb").write(
        b"engine"
    )
    bucket = MagicMock()
    bucket.list_blobs.return_value = [blob]
    client = MagicMock()
    client.bucket.return_value = bucket

    with patch("google.cloud.storage.Client", return_value=client):
        sync.download()

    assert (tmp_path / "graph_sm89.engine").read_bytes() == b"engine"


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


def test_upload_timeout_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_TIMEOUT", "7.5")
    assert make_sync(tmp_path)._timeout == 7.5


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

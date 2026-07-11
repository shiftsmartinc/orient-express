"""Tests for utils.paths cache-dir resolution."""

import platformdirs

from orient_express.utils.paths import get_cache_dir


def test_env_var_overrides_cache_dir(monkeypatch):
    monkeypatch.setenv("ORIENT_EXPRESS_CACHE", "/custom/cache/dir")
    assert get_cache_dir() == "/custom/cache/dir"


def test_defaults_to_platform_user_cache(monkeypatch):
    monkeypatch.delenv("ORIENT_EXPRESS_CACHE", raising=False)
    assert get_cache_dir() == platformdirs.user_cache_dir("orient-express")

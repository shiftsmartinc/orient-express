import os

import platformdirs


def get_metadata_path(dir: str):
    metadata_path = os.path.join(dir, "metadata.yaml")
    return metadata_path


def get_cache_dir() -> str:
    """Local directory for downloaded model artifacts.

    Defaults to the platform user cache (e.g. ~/.cache/orient-express on
    Linux, ~/Library/Caches/orient-express on macOS). Set the
    ORIENT_EXPRESS_CACHE environment variable (before importing
    orient_express) to override — e.g. to a known mount point in a serving
    container.
    """
    env_dir = os.environ.get("ORIENT_EXPRESS_CACHE")
    if env_dir:
        return env_dir
    return platformdirs.user_cache_dir("orient-express")

import os


def get_metadata_path(dir: str):
    metadata_path = os.path.join(dir, "metadata.yaml")
    return metadata_path

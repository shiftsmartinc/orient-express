import logging
import os
from urllib.parse import urlparse, unquote

from google.api_core.retry import Retry
from google.cloud import storage


def get_default_retry_policy():
    return Retry(initial=1.0, maximum=10.0, multiplier=2.0)


def parse_gcs_url(gs_url):
    """
    Parse URL of a file, located in Google Cloud Storage bucket.
    """
    parsed = urlparse(gs_url)
    if not parsed.scheme == "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URL format for {gs_url}")
    bucket_name = parsed.netloc
    file_path = parsed.path.lstrip("/")  # Remove leading slash for consistency
    return bucket_name, file_path


def get_gcs_from_http_url(http_url):
    """
    If HTTP URL belongs to Google Storage Bucket, extract GCS URI, otherwise return None

    :param http_url:
    :return: gcs_uri
    """
    if not http_url.startswith("https://storage.googleapis.com/"):
        return None

    return unquote(http_url.replace("https://storage.googleapis.com/", "gs://"))


def exists(gs_url):
    """
    Check whether a GCS URL exists
    """
    bucket_name, path = parse_gcs_url(gs_url)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.exists()


def upload_file(
    file_path, gs_url, read_mode="rb", content_type="application/octet-stream"
):
    """
    Upload file to Google Cloud Storage

    Args:
        file_path (str): Path to the local file to upload
        gs_url (str): Google Cloud Storage URL where the file should be uploaded
        content_type (str): MIME type of the file being uploaded
    """
    bucket_name, gs_file_path = parse_gcs_url(gs_url)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gs_file_path)

    # Open file in read binary mode for uploading
    with open(file_path, read_mode) as file_obj:
        blob.upload_from_file(
            file_obj, content_type=content_type, retry=get_default_retry_policy()
        )

    logging.info(f"File uploaded to {gs_url}")


def download_file(gs_url, file_path):
    """
    Download file from Google Cloud Storage to local filesystem.

    Args:
        gs_url (str): Google Cloud Storage URL (e.g., 'gs://bucket-name/path/to/file.txt')
        file_path (str): Local file path where the file should be saved

    Raises:
        ValueError: If gs_url is not a valid GCS URL
        google.cloud.exceptions.NotFound: If the file doesn't exist in GCS
        IOError: If there's an issue writing to the local file path

    Example:
        download_file('gs://my-bucket/data/file.csv', '/local/path/file.csv')
    """
    bucket_name, gs_file_path = parse_gcs_url(gs_url)

    # Create local directory if it doesn't exist
    local_dir = os.path.dirname(file_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gs_file_path)

    # Download the file
    blob.download_to_filename(file_path, retry=get_default_retry_policy())

    logging.info(f"File downloaded from {gs_url} to {file_path}")


def read_file_bytes(gs_url):
    """
    Read file, located in Google Storage bucket, as bytes
    """
    bucket_name, path = parse_gcs_url(gs_url)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.download_as_bytes()
